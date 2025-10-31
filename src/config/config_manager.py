"""Configuration management for the sentiment analysis engine."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration settings for the sentiment analysis engine.

    Handles loading configuration from JSON files, providing defaults,
    and validating configuration values.
    """

    DEFAULT_CONFIG = {
        "sentiment_thresholds": {"positive": 0.05, "negative": -0.05},
        "output": {
            "summary_format": "json",
            "results_format": "csv",
            "summary_filename": "summary",
            "results_filename": "results",
        },
        "logging": {
            "level": "INFO",
            "file": "sentiment_analysis.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "processing": {
            "batch_size": 100,
            "encoding_fallbacks": ["utf-8", "latin-1", "cp1252"],
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or "config.json"
        self._config = self.DEFAULT_CONFIG.copy()

        # Load configuration if file exists
        if os.path.exists(self.config_path):
            self._load_config()
        else:
            # Create default config file
            self._create_default_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)

            # Merge user config with defaults
            self._config = self._merge_configs(self.DEFAULT_CONFIG, user_config)
            self._validate_config()

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
        except ValueError as e:
            raise e  # Re-raise ValueError as-is for proper test handling
        except Exception as e:
            raise RuntimeError(f"Error loading config file {self.config_path}: {e}")

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
        except Exception as e:
            # If we can't create config file, continue with defaults
            print(f"Warning: Could not create config file {self.config_path}: {e}")

    def _merge_configs(
        self, default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge user configuration with defaults.

        Args:
            default: Default configuration dictionary
            user: User configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = default.copy()

        for key, value in user.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate sentiment thresholds
        pos_threshold = self.get_positive_threshold()
        neg_threshold = self.get_negative_threshold()

        if pos_threshold <= neg_threshold:
            raise ValueError(
                f"Positive threshold ({pos_threshold}) must be greater than "
                f"negative threshold ({neg_threshold})"
            )

        if not -1.0 <= neg_threshold <= 1.0:
            raise ValueError(
                f"Negative threshold must be between -1.0 and 1.0, got {neg_threshold}"
            )

        if not -1.0 <= pos_threshold <= 1.0:
            raise ValueError(
                f"Positive threshold must be between -1.0 and 1.0, got {pos_threshold}"
            )

        # Validate output formats
        valid_summary_formats = ["json", "txt"]
        summary_format = self.get_summary_format()
        if summary_format not in valid_summary_formats:
            raise ValueError(
                f"Summary format must be one of {valid_summary_formats}, got {summary_format}"
            )

        valid_results_formats = ["csv"]
        results_format = self.get_results_format()
        if results_format not in valid_results_formats:
            raise ValueError(
                f"Results format must be one of {valid_results_formats}, got {results_format}"
            )

        # Validate logging level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = self.get_log_level()
        if log_level not in valid_log_levels:
            raise ValueError(
                f"Log level must be one of {valid_log_levels}, got {log_level}"
            )

    def get_positive_threshold(self) -> float:
        """Get the positive sentiment threshold."""
        return self._config["sentiment_thresholds"]["positive"]

    def get_negative_threshold(self) -> float:
        """Get the negative sentiment threshold."""
        return self._config["sentiment_thresholds"]["negative"]

    def get_summary_format(self) -> str:
        """Get the summary output format."""
        return self._config["output"]["summary_format"]

    def get_results_format(self) -> str:
        """Get the results output format."""
        return self._config["output"]["results_format"]

    def get_summary_filename(self) -> str:
        """Get the summary output filename (without extension)."""
        return self._config["output"]["summary_filename"]

    def get_results_filename(self) -> str:
        """Get the results output filename (without extension)."""
        return self._config["output"]["results_filename"]

    def get_log_level(self) -> str:
        """Get the logging level."""
        return self._config["logging"]["level"]

    def get_log_file(self) -> str:
        """Get the log file path."""
        return self._config["logging"]["file"]

    def get_log_format(self) -> str:
        """Get the log message format."""
        return self._config["logging"]["format"]

    def get_batch_size(self) -> int:
        """Get the processing batch size."""
        return self._config["processing"]["batch_size"]

    def get_encoding_fallbacks(self) -> list:
        """Get the list of encoding fallbacks."""
        return self._config["processing"]["encoding_fallbacks"]

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        self._config = self._merge_configs(self._config, updates)
        self._validate_config()

    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses current config_path.
        """
        save_path = path or self.config_path
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Error saving config to {save_path}: {e}")

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self._config, indent=2)
