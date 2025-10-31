"""Tests for configuration management."""

import json
import pytest
import tempfile
import os
from src.config.config_manager import ConfigManager


class TestConfigManager:
    """Test cases for ConfigManager."""

    def test_default_initialization(self):
        """Test initialization with default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            config = ConfigManager(config_path)

            # Should create default config file
            assert os.path.exists(config_path)

            # Check default values
            assert config.get_positive_threshold() == 0.05
            assert config.get_negative_threshold() == -0.05
            assert config.get_summary_format() == "json"
            assert config.get_results_format() == "csv"
            assert config.get_log_level() == "INFO"

    def test_load_existing_config(self):
        """Test loading existing configuration file."""
        custom_config = {
            "sentiment_thresholds": {"positive": 0.1, "negative": -0.1},
            "output": {"summary_format": "txt", "results_format": "csv"},
            "logging": {"level": "DEBUG"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(custom_config, f)
            config_path = f.name

        try:
            config = ConfigManager(config_path)

            assert config.get_positive_threshold() == 0.1
            assert config.get_negative_threshold() == -0.1
            assert config.get_summary_format() == "txt"
            assert config.get_log_level() == "DEBUG"
        finally:
            os.unlink(config_path)

    def test_invalid_json_config(self):
        """Test handling of invalid JSON configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid thresholds
        invalid_config = {
            "sentiment_thresholds": {
                "positive": -0.1,  # Should be greater than negative
                "negative": 0.1,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Positive threshold.*must be greater"):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_threshold_validation_bounds(self):
        """Test threshold validation bounds."""
        # Threshold out of bounds
        invalid_config = {
            "sentiment_thresholds": {
                "positive": 1.5,  # Out of bounds
                "negative": -0.05,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="must be between -1.0 and 1.0"):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_invalid_output_format(self):
        """Test validation of invalid output formats."""
        invalid_config = {"output": {"summary_format": "invalid_format"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Summary format must be"):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_invalid_log_level(self):
        """Test validation of invalid log levels."""
        invalid_config = {"logging": {"level": "INVALID_LEVEL"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Log level must be one of"):
                ConfigManager(config_path)
        finally:
            os.unlink(config_path)

    def test_config_merging(self):
        """Test configuration merging with defaults."""
        partial_config = {
            "sentiment_thresholds": {
                "positive": 0.2  # Only override positive threshold
            },
            "logging": {"level": "WARNING"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(partial_config, f)
            config_path = f.name

        try:
            config = ConfigManager(config_path)

            # Overridden values
            assert config.get_positive_threshold() == 0.2
            assert config.get_log_level() == "WARNING"

            # Default values should remain
            assert config.get_negative_threshold() == -0.05  # Default
            assert config.get_summary_format() == "json"  # Default
        finally:
            os.unlink(config_path)

    def test_update_config(self):
        """Test configuration updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            config = ConfigManager(config_path)

            # Update configuration
            updates = {"sentiment_thresholds": {"positive": 0.15, "negative": -0.15}}

            config.update_config(updates)

            assert config.get_positive_threshold() == 0.15
            assert config.get_negative_threshold() == -0.15

    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            config = ConfigManager(config_path)

            # Modify configuration
            config.update_config({"sentiment_thresholds": {"positive": 0.3}})

            # Save to new file
            new_config_path = os.path.join(temp_dir, "saved_config.json")
            config.save_config(new_config_path)

            # Load saved configuration
            with open(new_config_path, "r") as f:
                saved_config = json.load(f)

            assert saved_config["sentiment_thresholds"]["positive"] == 0.3

    def test_get_all_config_methods(self):
        """Test all configuration getter methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            config = ConfigManager(config_path)

            # Test all getter methods
            assert isinstance(config.get_positive_threshold(), float)
            assert isinstance(config.get_negative_threshold(), float)
            assert isinstance(config.get_summary_format(), str)
            assert isinstance(config.get_results_format(), str)
            assert isinstance(config.get_summary_filename(), str)
            assert isinstance(config.get_results_filename(), str)
            assert isinstance(config.get_log_level(), str)
            assert isinstance(config.get_log_file(), str)
            assert isinstance(config.get_log_format(), str)
            assert isinstance(config.get_batch_size(), int)
            assert isinstance(config.get_encoding_fallbacks(), list)
            assert isinstance(config.get_config(), dict)

    def test_string_representation(self):
        """Test string representation of configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            config = ConfigManager(config_path)

            config_str = str(config)

            # Should be valid JSON
            parsed_config = json.loads(config_str)
            assert isinstance(parsed_config, dict)
            assert "sentiment_thresholds" in parsed_config
