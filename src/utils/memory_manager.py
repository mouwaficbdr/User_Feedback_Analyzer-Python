"""Memory management utilities for the sentiment analysis engine."""

import gc
import psutil
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class MemoryManager:
    """
    Manages memory usage and provides utilities for memory-efficient processing.

    Monitors memory usage and provides warnings when memory consumption is high.
    """

    def __init__(self, max_memory_percent: float = 80.0):
        """
        Initialize memory manager.

        Args:
            max_memory_percent: Maximum memory usage percentage before warnings
        """
        self.max_memory_percent = max_memory_percent
        self.logger = logging.getLogger(__name__)
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage statistics
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()

            return {
                "process_memory_mb": memory_info.rss / (1024 * 1024),
                "process_memory_percent": process.memory_percent(),
                "system_memory_percent": system_memory.percent,
                "system_available_mb": system_memory.available / (1024 * 1024),
                "system_total_mb": system_memory.total / (1024 * 1024),
            }
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return {
                "process_memory_mb": 0,
                "process_memory_percent": 0,
                "system_memory_percent": 0,
                "system_available_mb": 0,
                "system_total_mb": 0,
            }

    def check_memory_constraints(self) -> List[str]:
        """
        Check current memory usage against constraints.

        Returns:
            List of memory warnings/errors
        """
        warnings = []
        memory_info = self.get_memory_usage()

        # Check system memory usage
        if memory_info["system_memory_percent"] > self.max_memory_percent:
            warnings.append(
                f"High system memory usage: {memory_info['system_memory_percent']:.1f}% "
                f"(threshold: {self.max_memory_percent}%)"
            )

        # Check available memory
        if memory_info["system_available_mb"] < 100:  # Less than 100MB available
            warnings.append(
                f"Low available memory: {memory_info['system_available_mb']:.1f}MB"
            )

        # Check process memory usage
        if (
            memory_info["process_memory_percent"] > 50
        ):  # Process using more than 50% of system memory
            warnings.append(
                f"High process memory usage: {memory_info['process_memory_mb']:.1f}MB "
                f"({memory_info['process_memory_percent']:.1f}% of system memory)"
            )

        return warnings

    def estimate_processing_memory(
        self, num_reviews: int, avg_text_length: int = 200
    ) -> float:
        """
        Estimate memory requirements for processing reviews.

        Args:
            num_reviews: Number of reviews to process
            avg_text_length: Average text length per review

        Returns:
            Estimated memory usage in MB
        """
        # Rough estimation based on:
        # - Review objects: ~1KB per review
        # - Text processing: ~2x text size for preprocessing
        # - Sentiment analysis: ~0.5KB per review for results
        # - Overhead: 50% buffer

        review_memory = num_reviews * 1024  # 1KB per review object
        text_memory = num_reviews * avg_text_length * 2  # 2x for preprocessing
        analysis_memory = num_reviews * 512  # 0.5KB per analysis result

        total_bytes = (
            review_memory + text_memory + analysis_memory
        ) * 1.5  # 50% overhead
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def suggest_batch_size(self, num_reviews: int, available_memory_mb: float) -> int:
        """
        Suggest optimal batch size based on available memory.

        Args:
            num_reviews: Total number of reviews
            available_memory_mb: Available memory in MB

        Returns:
            Suggested batch size
        """
        # Reserve 25% of available memory for other operations
        usable_memory = available_memory_mb * 0.75

        # Estimate memory per review (conservative estimate)
        memory_per_review = 0.5  # 0.5MB per review (conservative)

        # Calculate batch size
        suggested_batch_size = max(1, int(usable_memory / memory_per_review))

        # Cap at reasonable limits
        suggested_batch_size = min(suggested_batch_size, 1000)  # Max 1000 per batch
        suggested_batch_size = min(
            suggested_batch_size, num_reviews
        )  # Don't exceed total

        return suggested_batch_size

    @contextmanager
    def memory_monitor(self, operation_name: str):
        """
        Context manager to monitor memory usage during an operation.

        Args:
            operation_name: Name of the operation being monitored
        """
        start_memory = self.get_memory_usage()
        self.logger.debug(
            f"Starting {operation_name} - Memory: {start_memory['process_memory_mb']:.1f}MB"
        )

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            memory_delta = (
                end_memory["process_memory_mb"] - start_memory["process_memory_mb"]
            )

            self.logger.debug(
                f"Completed {operation_name} - Memory: {end_memory['process_memory_mb']:.1f}MB "
                f"(Δ{memory_delta:+.1f}MB)"
            )

            # Check for memory warnings
            warnings = self.check_memory_constraints()
            for warning in warnings:
                self.logger.warning(
                    f"Memory warning during {operation_name}: {warning}"
                )

    def force_garbage_collection(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.

        Returns:
            Dictionary with garbage collection statistics
        """
        before_memory = self.get_memory_usage()

        # Force garbage collection
        collected = gc.collect()

        after_memory = self.get_memory_usage()
        memory_freed = (
            before_memory["process_memory_mb"] - after_memory["process_memory_mb"]
        )

        gc_stats = {
            "objects_collected": collected,
            "memory_before_mb": before_memory["process_memory_mb"],
            "memory_after_mb": after_memory["process_memory_mb"],
            "memory_freed_mb": memory_freed,
        }

        self.logger.debug(
            f"Garbage collection: {collected} objects, {memory_freed:.1f}MB freed"
        )

        return gc_stats

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage report.

        Returns:
            Dictionary with detailed memory information
        """
        current_memory = self.get_memory_usage()
        memory_delta = (
            current_memory["process_memory_mb"]
            - self.initial_memory["process_memory_mb"]
        )

        return {
            "current_usage": current_memory,
            "initial_usage": self.initial_memory,
            "memory_delta_mb": memory_delta,
            "warnings": self.check_memory_constraints(),
            "gc_stats": {
                "garbage_objects": len(gc.garbage),
                "gc_counts": gc.get_count(),
            },
        }


class ResourceConstraintHandler:
    """
    Handles resource constraints and provides fallback strategies.
    """

    def __init__(self):
        """Initialize resource constraint handler."""
        self.logger = logging.getLogger(__name__)
        self.memory_manager = MemoryManager()

    def handle_memory_constraint(self, num_reviews: int) -> Dict[str, Any]:
        """
        Handle memory constraints by suggesting processing strategies.

        Args:
            num_reviews: Number of reviews to process

        Returns:
            Dictionary with processing recommendations
        """
        memory_info = self.memory_manager.get_memory_usage()
        estimated_memory = self.memory_manager.estimate_processing_memory(num_reviews)

        recommendations = {
            "can_process_all": True,
            "suggested_batch_size": num_reviews,
            "use_streaming": False,
            "warnings": [],
            "memory_info": memory_info,
        }

        # Check if we have enough memory
        available_memory = memory_info["system_available_mb"]

        if (
            estimated_memory > available_memory * 0.8
        ):  # Need more than 80% of available memory
            recommendations["can_process_all"] = False
            recommendations["warnings"].append(
                f"Estimated memory usage ({estimated_memory:.1f}MB) exceeds available memory "
                f"({available_memory:.1f}MB)"
            )

            # Suggest batch processing
            suggested_batch_size = self.memory_manager.suggest_batch_size(
                num_reviews, available_memory
            )
            recommendations["suggested_batch_size"] = suggested_batch_size

            if suggested_batch_size < num_reviews:
                recommendations["warnings"].append(
                    f"Recommended batch processing with batch size: {suggested_batch_size}"
                )

        # Check if streaming is recommended
        if (
            num_reviews > 10000 or estimated_memory > 1000
        ):  # More than 10k reviews or 1GB
            recommendations["use_streaming"] = True
            recommendations["warnings"].append(
                "Large dataset detected. Consider streaming processing for better memory efficiency."
            )

        return recommendations

    def handle_disk_space_constraint(
        self, output_dir: str, estimated_output_size_mb: float = 10.0
    ) -> Dict[str, Any]:
        """
        Handle disk space constraints.

        Args:
            output_dir: Output directory path
            estimated_output_size_mb: Estimated output size in MB

        Returns:
            Dictionary with disk space information and recommendations
        """
        try:
            import shutil

            free_space = shutil.disk_usage(output_dir).free
            free_space_mb = free_space / (1024 * 1024)

            recommendations = {
                "sufficient_space": True,
                "free_space_mb": free_space_mb,
                "estimated_need_mb": estimated_output_size_mb,
                "warnings": [],
            }

            if (
                free_space_mb < estimated_output_size_mb * 2
            ):  # Need 2x the estimated size as buffer
                recommendations["sufficient_space"] = False
                recommendations["warnings"].append(
                    f"Low disk space: {free_space_mb:.1f}MB available, "
                    f"estimated need: {estimated_output_size_mb:.1f}MB"
                )

            return recommendations

        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return {
                "sufficient_space": True,  # Assume OK if we can't check
                "free_space_mb": 0,
                "estimated_need_mb": estimated_output_size_mb,
                "warnings": [f"Could not verify disk space: {e}"],
            }
