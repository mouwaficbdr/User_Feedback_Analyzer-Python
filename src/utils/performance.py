"""Performance monitoring and optimization utilities."""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    items_processed: int = 0
    
    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        if self.duration > 0 and self.items_processed > 0:
            return self.items_processed / self.duration
        return 0.0
    
    @property
    def memory_per_item(self) -> float:
        """Calculate memory usage per item."""
        if self.items_processed > 0 and self.memory_delta > 0:
            return self.memory_delta / self.items_processed
        return 0.0


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for the sentiment analysis pipeline.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.process = psutil.Process()
    
    @contextmanager
    def monitor_operation(self, operation_name: str, items_count: int = 0):
        """
        Context manager to monitor an operation's performance.
        
        Args:
            operation_name: Name of the operation being monitored
            items_count: Number of items being processed
            
        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        # Initial measurements
        start_time = time.time()
        memory_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_before = self.process.cpu_percent()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=0,
            duration=0,
            memory_before=memory_before,
            memory_after=0,
            memory_delta=0,
            cpu_percent=0,
            items_processed=items_count
        )
        
        try:
            yield metrics
        finally:
            # Final measurements
            end_time = time.time()
            memory_after = self.process.memory_info().rss / (1024 * 1024)  # MB
            cpu_after = self.process.cpu_percent()
            
            # Update metrics
            metrics.end_time = end_time
            metrics.duration = end_time - start_time
            metrics.memory_after = memory_after
            metrics.memory_delta = memory_after - memory_before
            metrics.cpu_percent = max(cpu_after, cpu_before)  # Take max as CPU% can be delayed
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log performance info
            self._log_performance(metrics)
    
    def _log_performance(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance [{metrics.operation_name}]: "
            f"{metrics.duration:.2f}s, "
            f"{metrics.memory_delta:+.1f}MB, "
            f"{metrics.items_per_second:.1f} items/s"
        )
        
        if metrics.duration > 30:  # Warn for operations > 30s
            self.logger.warning(
                f"Slow operation detected: {metrics.operation_name} "
                f"took {metrics.duration:.1f}s"
            )
        
        if metrics.memory_delta > 100:  # Warn for operations using > 100MB
            self.logger.warning(
                f"High memory usage: {metrics.operation_name} "
                f"used {metrics.memory_delta:.1f}MB"
            )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary with operation statistics
        """
        operation_metrics = [
            m for m in self.metrics_history 
            if m.operation_name == operation_name
        ]
        
        if not operation_metrics:
            return {"error": f"No metrics found for operation: {operation_name}"}
        
        durations = [m.duration for m in operation_metrics]
        memory_deltas = [m.memory_delta for m in operation_metrics]
        rates = [m.items_per_second for m in operation_metrics if m.items_per_second > 0]
        
        return {
            "operation_name": operation_name,
            "execution_count": len(operation_metrics),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations)
            },
            "memory_usage": {
                "min": min(memory_deltas),
                "max": max(memory_deltas),
                "avg": sum(memory_deltas) / len(memory_deltas),
                "total": sum(memory_deltas)
            },
            "processing_rate": {
                "min": min(rates) if rates else 0,
                "max": max(rates) if rates else 0,
                "avg": sum(rates) / len(rates) if rates else 0
            }
        }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.metrics_history:
            return {"error": "No performance metrics available"}
        
        total_duration = sum(m.duration for m in self.metrics_history)
        total_memory = sum(m.memory_delta for m in self.metrics_history)
        total_items = sum(m.items_processed for m in self.metrics_history)
        
        operations = list(set(m.operation_name for m in self.metrics_history))
        
        return {
            "total_operations": len(self.metrics_history),
            "unique_operations": len(operations),
            "total_duration": total_duration,
            "total_memory_used": total_memory,
            "total_items_processed": total_items,
            "overall_rate": total_items / total_duration if total_duration > 0 else 0,
            "operations": operations
        }
    
    def clear_history(self) -> None:
        """Clear performance metrics history."""
        self.metrics_history.clear()
        self.logger.debug("Performance metrics history cleared")


def performance_benchmark(func):
    """
    Decorator to benchmark function performance.
    
    Args:
        func: Function to benchmark
        
    Returns:
        Decorated function that logs performance metrics
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        
        # Try to estimate items count from arguments
        items_count = 0
        for arg in args:
            if hasattr(arg, '__len__'):
                items_count = len(arg)
                break
        
        with monitor.monitor_operation(func.__name__, items_count):
            result = func(*args, **kwargs)
        
        return result
    
    return wrapper


class ProgressTracker:
    """
    Tracks and reports progress for long-running operations.
    """
    
    def __init__(self, total_items: int, operation_name: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            operation_name: Name of the operation being tracked
        """
        self.total_items = total_items
        self.operation_name = operation_name
        self.processed_items = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.logger = logging.getLogger(__name__)
        
        # Progress reporting intervals
        self.report_interval = 5.0  # Report every 5 seconds
        self.percentage_milestones = [10, 25, 50, 75, 90]  # Report at these percentages
        self.reported_milestones = set()
    
    def update(self, items_processed: int = 1) -> None:
        """
        Update progress counter.
        
        Args:
            items_processed: Number of items processed since last update
        """
        self.processed_items += items_processed
        current_time = time.time()
        
        # Check if we should report progress
        should_report = (
            current_time - self.last_report_time >= self.report_interval or
            self._check_milestone_reached()
        )
        
        if should_report:
            self._report_progress()
            self.last_report_time = current_time
    
    def _check_milestone_reached(self) -> bool:
        """Check if a percentage milestone has been reached."""
        if self.total_items == 0:
            return False
        
        percentage = (self.processed_items / self.total_items) * 100
        
        for milestone in self.percentage_milestones:
            if (percentage >= milestone and 
                milestone not in self.reported_milestones):
                self.reported_milestones.add(milestone)
                return True
        
        return False
    
    def _report_progress(self) -> None:
        """Report current progress."""
        if self.total_items == 0:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        percentage = (self.processed_items / self.total_items) * 100
        
        # Calculate rate and ETA
        if elapsed_time > 0:
            rate = self.processed_items / elapsed_time
            remaining_items = self.total_items - self.processed_items
            eta_seconds = remaining_items / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.operation_name}: {self.processed_items}/{self.total_items} "
                f"({percentage:.1f}%) - {rate:.1f} items/s - "
                f"ETA: {self._format_time(eta_seconds)}"
            )
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def finish(self) -> Dict[str, Any]:
        """
        Mark processing as finished and return final statistics.
        
        Returns:
            Dictionary with final processing statistics
        """
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        stats = {
            "operation_name": self.operation_name,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "duration": total_duration,
            "average_rate": self.processed_items / total_duration if total_duration > 0 else 0,
            "completion_percentage": (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0
        }
        
        self.logger.info(
            f"{self.operation_name} completed: {self.processed_items} items "
            f"in {self._format_time(total_duration)} "
            f"({stats['average_rate']:.1f} items/s)"
        )
        
        return stats


class BatchProcessor:
    """
    Processes large datasets in optimized batches.
    """
    
    def __init__(self, batch_size: int = 100, max_memory_mb: float = 500.0):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            max_memory_mb: Maximum memory usage before reducing batch size
        """
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
    
    def process_in_batches(self, items: List[Any], processor_func, **kwargs) -> List[Any]:
        """
        Process items in optimized batches.
        
        Args:
            items: List of items to process
            processor_func: Function to process each batch
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processed results
        """
        if not items:
            return []
        
        total_items = len(items)
        results = []
        
        # Adaptive batch sizing
        current_batch_size = self._calculate_optimal_batch_size(total_items)
        
        # Progress tracking
        progress = ProgressTracker(total_items, "Batch Processing")
        
        # Process in batches
        for i in range(0, total_items, current_batch_size):
            batch = items[i:i + current_batch_size]
            
            # Monitor memory usage
            memory_before = self.process.memory_info().rss / (1024 * 1024)
            
            # Process batch
            try:
                batch_results = processor_func(batch, **kwargs)
                results.extend(batch_results)
                
                # Update progress
                progress.update(len(batch))
                
                # Check memory usage and adjust batch size if needed
                memory_after = self.process.memory_info().rss / (1024 * 1024)
                memory_used = memory_after - memory_before
                
                if memory_used > self.max_memory_mb * 0.8:  # 80% of max memory
                    current_batch_size = max(1, current_batch_size // 2)
                    self.logger.warning(
                        f"High memory usage detected ({memory_used:.1f}MB), "
                        f"reducing batch size to {current_batch_size}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//current_batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        # Finish progress tracking
        final_stats = progress.finish()
        
        self.logger.info(
            f"Batch processing completed: {len(results)}/{total_items} items processed"
        )
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """
        Calculate optimal batch size based on available memory and dataset size.
        
        Args:
            total_items: Total number of items to process
            
        Returns:
            Optimal batch size
        """
        # Get available memory
        memory_info = psutil.virtual_memory()
        available_mb = memory_info.available / (1024 * 1024)
        
        # Conservative estimate: use 25% of available memory
        usable_memory = available_mb * 0.25
        
        # Estimate memory per item (conservative: 1MB per item)
        estimated_memory_per_item = 1.0
        
        # Calculate batch size
        memory_based_batch_size = int(usable_memory / estimated_memory_per_item)
        
        # Apply constraints
        optimal_batch_size = min(
            self.batch_size,  # User-configured maximum
            memory_based_batch_size,  # Memory-based limit
            total_items,  # Don't exceed total items
            1000  # Absolute maximum
        )
        
        optimal_batch_size = max(1, optimal_batch_size)  # Minimum of 1
        
        self.logger.debug(
            f"Calculated optimal batch size: {optimal_batch_size} "
            f"(available memory: {available_mb:.1f}MB, total items: {total_items})"
        )
        
        return optimal_batch_size