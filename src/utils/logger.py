"""Logging utilities for the sentiment analysis engine."""

import logging
import logging.handlers
import os
from functools import wraps
from typing import Callable, Any, Optional
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str = "sentiment_analysis.log",
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with file rotation and console output.

    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler with rotation
    try:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")

    return logger


def handle_errors(
    logger: Optional[logging.Logger] = None,
    default_return: Any = None,
    reraise: bool = False,
) -> Callable:
    """
    Decorator for graceful error handling with logging.

    Args:
        logger: Logger instance to use for error logging
        default_return: Default value to return on error
        reraise: Whether to reraise the exception after logging

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"

                if logger:
                    logger.error(error_msg, exc_info=True)
                else:
                    # Fallback to print if no logger provided
                    print(f"ERROR: {error_msg}")

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance to use

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if logger:
                    logger.info(
                        f"{func.__name__} completed in {execution_time:.2f} seconds"
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                if logger:
                    logger.error(
                        f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}"
                    )

                raise

        return wrapper

    return decorator


class ErrorAggregator:
    """
    Collects and manages errors during processing.

    Allows for graceful error handling while maintaining a record
    of all issues encountered during processing.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error aggregator.

        Args:
            logger: Logger instance for error reporting
        """
        self.errors = []
        self.logger = logger or logging.getLogger(__name__)

    def add_error(self, error_message: str, context: str = "") -> None:
        """
        Add an error to the aggregator.

        Args:
            error_message: Description of the error
            context: Additional context about where the error occurred
        """
        full_message = f"{context}: {error_message}" if context else error_message
        self.errors.append(full_message)
        self.logger.error(full_message)

    def add_exception(self, exception: Exception, context: str = "") -> None:
        """
        Add an exception to the aggregator.

        Args:
            exception: Exception that occurred
            context: Additional context about where the exception occurred
        """
        error_message = f"{type(exception).__name__}: {str(exception)}"
        self.add_error(error_message, context)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0

    def get_errors(self) -> list:
        """Get list of all recorded errors."""
        return self.errors.copy()

    def get_error_count(self) -> int:
        """Get the total number of errors."""
        return len(self.errors)

    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()

    def log_summary(self) -> None:
        """Log a summary of all errors."""
        if self.has_errors():
            self.logger.warning(
                f"Processing completed with {self.get_error_count()} errors:"
            )
            for i, error in enumerate(self.errors, 1):
                self.logger.warning(f"  {i}. {error}")
        else:
            self.logger.info("Processing completed without errors")


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate a file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must already exist

    Returns:
        True if path is valid, False otherwise

    Raises:
        ValueError: If path is invalid
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")

    path = Path(file_path)

    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    if must_exist and not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check if parent directory exists for output files
    if not must_exist:
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create directory {parent_dir}: {e}")

    return True


def check_disk_space(file_path: str, required_mb: float = 10.0) -> bool:
    """
    Check if there's enough disk space for file operations.

    Args:
        file_path: Path to check disk space for
        required_mb: Required space in megabytes

    Returns:
        True if enough space is available

    Raises:
        RuntimeError: If insufficient disk space
    """
    try:
        import shutil

        path = Path(file_path).parent
        free_bytes = shutil.disk_usage(path).free
        free_mb = free_bytes / (1024 * 1024)

        if free_mb < required_mb:
            raise RuntimeError(
                f"Insufficient disk space. Required: {required_mb:.1f}MB, "
                f"Available: {free_mb:.1f}MB"
            )

        return True

    except Exception as e:
        # If we can't check disk space, log warning but don't fail
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not check disk space: {e}")
        return True
