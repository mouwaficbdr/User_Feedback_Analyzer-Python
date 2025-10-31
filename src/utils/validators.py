"""Validation utilities for the sentiment analysis engine."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from .logger import ErrorAggregator


class InputValidator:
    """Validates input files and parameters for the sentiment analysis engine."""

    def __init__(self):
        """Initialize the input validator."""
        self.logger = logging.getLogger(__name__)
        self.error_aggregator = ErrorAggregator(self.logger)

    def validate_input_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of input JSON file.

        Args:
            file_path: Path to the input file

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Basic file existence and type checks
            file_errors = self._validate_file_basic(file_path)
            errors.extend(file_errors)

            if file_errors:  # If basic validation fails, don't continue
                return False, errors

            # File size and accessibility checks
            size_errors = self._validate_file_size(file_path)
            errors.extend(size_errors)

            # JSON structure validation
            json_errors = self._validate_json_structure(file_path)
            errors.extend(json_errors)

            # Content validation
            content_errors = self._validate_json_content(file_path)
            errors.extend(content_errors)

        except Exception as e:
            errors.append(f"Unexpected error during validation: {e}")

        is_valid = len([e for e in errors if not e.startswith("Warning:")]) == 0
        return is_valid, errors

    def _validate_file_basic(self, file_path: str) -> List[str]:
        """Basic file validation (existence, type, permissions)."""
        errors = []

        if not file_path or not isinstance(file_path, str):
            errors.append("File path must be a non-empty string")
            return errors

        path_obj = Path(file_path)

        # Check existence
        if not path_obj.exists():
            errors.append(f"File does not exist: {file_path}")
            return errors

        # Check if it's a file
        if not path_obj.is_file():
            errors.append(f"Path is not a file: {file_path}")
            return errors

        # Check file extension
        if path_obj.suffix.lower() != ".json":
            errors.append(f"File must have .json extension, got: {path_obj.suffix}")

        # Check read permissions
        if not os.access(file_path, os.R_OK):
            errors.append(f"File is not readable: {file_path}")

        return errors

    def _validate_file_size(self, file_path: str) -> List[str]:
        """Validate file size and warn about potential issues."""
        errors = []

        try:
            file_size = Path(file_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Empty file check
            if file_size == 0:
                errors.append("File is empty")
                return errors

            # Very small file warning
            if file_size < 100:  # Less than 100 bytes
                errors.append(
                    "Warning: File is very small, may not contain valid review data"
                )

            # Large file warning
            if file_size_mb > 50:  # More than 50MB
                errors.append(
                    f"Warning: Large file ({file_size_mb:.1f}MB) may require significant processing time"
                )

            # Extremely large file error
            if file_size_mb > 500:  # More than 500MB
                errors.append(
                    f"File too large ({file_size_mb:.1f}MB). Maximum recommended size is 500MB"
                )

        except Exception as e:
            errors.append(f"Could not check file size: {e}")

        return errors

    def _validate_json_structure(self, file_path: str) -> List[str]:
        """Validate JSON file structure and format."""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for BOM
            if content.startswith("\ufeff"):
                errors.append("Warning: File contains BOM (Byte Order Mark)")

            # Try to parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format: {e}")
                return errors

            # Check basic structure
            if not isinstance(data, (list, dict)):
                errors.append(
                    f"JSON root must be a list or object, got: {type(data).__name__}"
                )
                return errors

            # If it's a dict, check for common wrapper patterns
            if isinstance(data, dict):
                if not any(key in data for key in ["reviews", "data", "items"]):
                    # Check if it looks like a single review
                    if not ("review_id" in data or "id" in data):
                        errors.append(
                            "Warning: JSON object doesn't contain expected review structure"
                        )

        except UnicodeDecodeError as e:
            errors.append(f"File encoding error: {e}")
        except Exception as e:
            errors.append(f"Error reading file: {e}")

        return errors

    def _validate_json_content(self, file_path: str) -> List[str]:
        """Validate the content of the JSON file for review data."""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract review data based on structure
            if isinstance(data, list):
                reviews_data = data
            elif isinstance(data, dict):
                if "reviews" in data:
                    reviews_data = data["reviews"]
                elif len(data) == 1 and isinstance(list(data.values())[0], list):
                    reviews_data = list(data.values())[0]
                else:
                    reviews_data = [data]  # Single review
            else:
                errors.append("Cannot extract review data from JSON structure")
                return errors

            if not isinstance(reviews_data, list):
                errors.append("Review data must be a list")
                return errors

            if len(reviews_data) == 0:
                errors.append("No reviews found in file")
                return errors

            # Validate individual reviews
            review_errors = self._validate_reviews_content(reviews_data)
            errors.extend(review_errors)

        except Exception as e:
            errors.append(f"Error validating JSON content: {e}")

        return errors

    def _validate_reviews_content(self, reviews_data: List[Any]) -> List[str]:
        """Validate individual review entries."""
        errors = []

        valid_reviews = 0
        empty_text_count = 0
        missing_id_count = 0

        for i, review in enumerate(reviews_data):
            if not isinstance(review, dict):
                errors.append(
                    f"Review at index {i} is not an object: {type(review).__name__}"
                )
                continue

            # Check for ID field
            has_id = any(field in review for field in ["review_id", "id", "reviewId"])
            if not has_id:
                missing_id_count += 1

            # Check for text field
            has_text = any(
                field in review
                for field in ["review_text", "text", "content", "review"]
            )
            if not has_text:
                errors.append(f"Review at index {i} missing text field")
                continue

            # Check if text is empty
            text_field = None
            for field in ["review_text", "text", "content", "review"]:
                if field in review:
                    text_field = review[field]
                    break

            if not text_field or (
                isinstance(text_field, str) and not text_field.strip()
            ):
                empty_text_count += 1

            valid_reviews += 1

        # Summary warnings
        if missing_id_count > 0:
            errors.append(
                f"Warning: {missing_id_count} reviews missing ID fields (will be auto-generated)"
            )

        if empty_text_count > 0:
            errors.append(
                f"Warning: {empty_text_count} reviews have empty text (will be assigned neutral sentiment)"
            )

        if valid_reviews == 0:
            errors.append("No valid reviews found in file")
        else:
            errors.append(f"Info: Found {valid_reviews} processable reviews")

        return errors

    def validate_output_directory(self, output_dir: str) -> Tuple[bool, List[str]]:
        """
        Validate output directory for write access and space.

        Args:
            output_dir: Path to output directory

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            output_path = Path(output_dir)

            # Check if path exists
            if output_path.exists():
                # Check if it's a directory
                if not output_path.is_dir():
                    errors.append(
                        f"Output path exists but is not a directory: {output_dir}"
                    )
                    return False, errors

                # Check write permissions
                if not os.access(output_path, os.W_OK):
                    errors.append(f"No write permission for directory: {output_dir}")
                    return False, errors
            else:
                # Try to create directory
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create output directory: {e}")
                    return False, errors

            # Check disk space
            try:
                free_space = shutil.disk_usage(output_path).free
                free_mb = free_space / (1024 * 1024)

                if free_mb < 10:  # Less than 10MB
                    errors.append(
                        f"Insufficient disk space: {free_mb:.1f}MB available, need at least 10MB"
                    )
                    return False, errors
                elif free_mb < 50:  # Less than 50MB
                    errors.append(f"Warning: Low disk space: {free_mb:.1f}MB available")

            except Exception as e:
                errors.append(f"Warning: Could not check disk space: {e}")

        except Exception as e:
            errors.append(f"Error validating output directory: {e}")
            return False, errors

        return True, errors

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Validate sentiment thresholds
            if "sentiment_thresholds" in config:
                thresholds = config["sentiment_thresholds"]

                pos_threshold = thresholds.get("positive", 0.05)
                neg_threshold = thresholds.get("negative", -0.05)

                if not isinstance(pos_threshold, (int, float)):
                    errors.append("Positive threshold must be a number")
                elif not -1.0 <= pos_threshold <= 1.0:
                    errors.append(
                        f"Positive threshold must be between -1.0 and 1.0, got: {pos_threshold}"
                    )

                if not isinstance(neg_threshold, (int, float)):
                    errors.append("Negative threshold must be a number")
                elif not -1.0 <= neg_threshold <= 1.0:
                    errors.append(
                        f"Negative threshold must be between -1.0 and 1.0, got: {neg_threshold}"
                    )

                if isinstance(pos_threshold, (int, float)) and isinstance(
                    neg_threshold, (int, float)
                ):
                    if pos_threshold <= neg_threshold:
                        errors.append(
                            f"Positive threshold ({pos_threshold}) must be greater than negative threshold ({neg_threshold})"
                        )

            # Validate output formats
            if "output" in config:
                output_config = config["output"]

                summary_format = output_config.get("summary_format", "json")
                if summary_format not in ["json", "txt"]:
                    errors.append(
                        f"Summary format must be 'json' or 'txt', got: {summary_format}"
                    )

                results_format = output_config.get("results_format", "csv")
                if results_format not in ["csv"]:
                    errors.append(
                        f"Results format must be 'csv', got: {results_format}"
                    )

            # Validate logging configuration
            if "logging" in config:
                logging_config = config["logging"]

                log_level = logging_config.get("level", "INFO")
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                if log_level not in valid_levels:
                    errors.append(
                        f"Log level must be one of {valid_levels}, got: {log_level}"
                    )

            # Validate processing configuration
            if "processing" in config:
                processing_config = config["processing"]

                batch_size = processing_config.get("batch_size", 100)
                if not isinstance(batch_size, int) or batch_size <= 0:
                    errors.append(
                        f"Batch size must be a positive integer, got: {batch_size}"
                    )
                elif batch_size > 10000:
                    errors.append(
                        f"Warning: Very large batch size ({batch_size}) may cause memory issues"
                    )

        except Exception as e:
            errors.append(f"Error validating configuration: {e}")

        is_valid = len([e for e in errors if not e.startswith("Warning:")]) == 0
        return is_valid, errors

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "total_errors": self.error_aggregator.get_error_count(),
            "errors": self.error_aggregator.get_errors(),
            "has_errors": self.error_aggregator.has_errors(),
        }
