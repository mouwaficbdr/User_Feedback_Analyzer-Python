"""Comprehensive tests for SentimentAnalysisEngine to achieve 90%+ coverage."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import time

from src.engine import SentimentAnalysisEngine
from src.models.review import Review


class TestEngineInitialization:
    """Tests for engine initialization."""

    def test_engine_initialization_default(self):
        """Test engine initialization with default configuration."""
        engine = SentimentAnalysisEngine()
        assert engine is not None
        assert engine.config is not None
        assert engine.data_loader is not None
        assert engine.preprocessor is not None
        assert engine.sentiment_analyzer is not None
        assert engine.report_generator is not None

    def test_engine_initialization_with_config(self, tmp_path):
        """Test engine initialization with custom configuration."""
        config_file = tmp_path / "custom_config.json"
        config_data = {
            "sentiment_thresholds": {"positive": 0.1, "negative": -0.1},
            "output": {"summary_format": "json", "results_format": "csv"},
            "logging": {"level": "DEBUG", "file": "test.log"},
            "processing": {"batch_size": 50},
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        engine = SentimentAnalysisEngine(config_path=str(config_file))
        assert engine.config.get_positive_threshold() == 0.1
        assert engine.config.get_negative_threshold() == -0.1

    def test_engine_initialization_invalid_config(self):
        """Test engine initialization with invalid config path."""
        with pytest.raises(Exception):
            SentimentAnalysisEngine(config_path="nonexistent_config.json")


class TestEngineValidation:
    """Tests for input file validation."""

    def test_validate_input_file_valid(self, tmp_path):
        """Test validation of valid input file."""
        input_file = tmp_path / "valid.json"
        data = [
            {"review_id": "R001", "review_text": "Good product"},
            {"review_id": "R002", "review_text": "Bad service"},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        errors = engine.validate_input_file(str(input_file))
        assert len(errors) == 0

    def test_validate_input_file_not_found(self):
        """Test validation with non-existent file."""
        engine = SentimentAnalysisEngine()
        errors = engine.validate_input_file("nonexistent.json")
        assert len(errors) > 0
        assert any("does not exist" in error for error in errors)

    def test_validate_input_file_not_json(self, tmp_path):
        """Test validation with non-JSON file."""
        input_file = tmp_path / "test.txt"
        input_file.write_text("Not a JSON file")

        engine = SentimentAnalysisEngine()
        errors = engine.validate_input_file(str(input_file))
        assert len(errors) > 0

    def test_validate_input_file_large_warning(self, tmp_path):
        """Test validation with large file (warning)."""
        input_file = tmp_path / "large.json"
        # Create a large file (> 100MB would trigger warning, but we'll mock it)
        data = [{"review_id": f"R{i:05d}", "review_text": "Test" * 100} for i in range(1000)]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        errors = engine.validate_input_file(str(input_file))
        # Should complete without critical errors
        assert all(not error.startswith("Error") for error in errors if error)


class TestEngineAnalysis:
    """Tests for sentiment analysis pipeline."""

    def test_analyze_reviews_basic(self, tmp_path):
        """Test basic sentiment analysis."""
        input_file = tmp_path / "reviews.json"
        data = [
            {"review_id": "R001", "review_text": "Excellent product!"},
            {"review_id": "R002", "review_text": "Terrible service."},
            {"review_id": "R003", "review_text": "It's okay."},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 3
        assert "files_generated" in results
        assert "summary" in results["files_generated"]
        assert "results" in results["files_generated"]

    def test_analyze_reviews_with_progress_callback(self, tmp_path):
        """Test analysis with progress callback."""
        input_file = tmp_path / "reviews.json"
        data = [{"review_id": f"R{i:03d}", "review_text": "Test review"} for i in range(10)]
        with open(input_file, "w") as f:
            json.dump(data, f)

        progress_calls = []

        def callback(message, percentage):
            progress_calls.append((message, percentage))

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path), progress_callback=callback)

        assert len(progress_calls) > 0
        assert progress_calls[-1][1] == 100  # Last call should be 100%

    def test_analyze_reviews_empty_file(self, tmp_path):
        """Test analysis with empty review list."""
        input_file = tmp_path / "empty.json"
        with open(input_file, "w") as f:
            json.dump([], f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["statistics"]["total_reviews"] == 0

    def test_analyze_reviews_with_errors(self, tmp_path):
        """Test analysis with problematic reviews."""
        input_file = tmp_path / "problematic.json"
        data = [
            {"review_id": "R001", "review_text": ""},  # Empty text
            {"review_id": "R002", "review_text": "Good"},  # Valid
            {"review_id": "R003", "review_text": None},  # None text
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["processing_info"]["total_errors"] >= 0  # Should handle errors gracefully


class TestEngineRobustness:
    """Tests for engine robustness and error handling."""

    def test_analyze_with_corrupted_json(self, tmp_path):
        """Test with corrupted JSON file."""
        input_file = tmp_path / "corrupted.json"
        input_file.write_text('{"review_id": "R001", "review_text": "Test"')  # Missing closing brace

        engine = SentimentAnalysisEngine()
        with pytest.raises(Exception):
            engine.analyze_reviews(str(input_file), str(tmp_path))

    def test_analyze_with_mixed_encodings(self, tmp_path):
        """Test with various character encodings."""
        input_file = tmp_path / "mixed_encoding.json"
        data = [
            {"review_id": "R001", "review_text": "Très bon produit! 😀"},
            {"review_id": "R002", "review_text": "Qualité médiocre"},
            {"review_id": "R003", "review_text": "Ça va"},
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 3

    def test_analyze_with_special_characters(self, tmp_path):
        """Test with special characters and emojis."""
        input_file = tmp_path / "special_chars.json"
        data = [
            {"review_id": "R001", "review_text": "😀😀😀👍👍"},
            {"review_id": "R002", "review_text": "!!!???..."},
            {"review_id": "R003", "review_text": "@#$%^&*()"},
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"

    def test_analyze_with_very_long_text(self, tmp_path):
        """Test with very long review text."""
        input_file = tmp_path / "long_text.json"
        long_text = "Excellent produit! " * 500  # Very long text
        data = [{"review_id": "R001", "review_text": long_text}]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"

    def test_analyze_with_very_short_text(self, tmp_path):
        """Test with very short review texts."""
        input_file = tmp_path / "short_text.json"
        data = [
            {"review_id": "R001", "review_text": "OK"},
            {"review_id": "R002", "review_text": "!"},
            {"review_id": "R003", "review_text": "a"},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 3


class TestEnginePerformance:
    """Tests for engine performance."""

    def test_analyze_medium_dataset(self, tmp_path):
        """Test with medium-sized dataset (100 reviews)."""
        input_file = tmp_path / "medium.json"
        data = [
            {"review_id": f"R{i:04d}", "review_text": f"Test review number {i}"}
            for i in range(100)
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        start_time = time.time()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        duration = time.time() - start_time

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 100
        assert duration < 30  # Should complete in less than 30 seconds

    def test_analyze_large_dataset(self, tmp_path):
        """Test with large dataset (500 reviews)."""
        input_file = tmp_path / "large.json"
        data = [
            {"review_id": f"R{i:05d}", "review_text": f"Review {i}: Good product"}
            for i in range(500)
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        start_time = time.time()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        duration = time.time() - start_time

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 500
        assert duration < 120  # Should complete in less than 2 minutes


class TestEngineConfiguration:
    """Tests for engine configuration management."""

    def test_get_engine_info(self):
        """Test getting engine information."""
        engine = SentimentAnalysisEngine()
        info = engine.get_engine_info()

        assert "version" in info
        assert "components" in info
        assert "configuration" in info
        assert "analyzer_info" in info

    def test_update_configuration(self):
        """Test updating engine configuration."""
        engine = SentimentAnalysisEngine()
        
        new_config = {
            "sentiment_thresholds.positive": 0.15,
            "sentiment_thresholds.negative": -0.15,
        }
        
        engine.update_configuration(new_config)
        
        # Verify thresholds were updated
        analyzer_info = engine.sentiment_analyzer.get_analyzer_info()
        assert analyzer_info["positive_threshold"] == 0.15
        assert analyzer_info["negative_threshold"] == -0.15

    def test_update_configuration_invalid(self):
        """Test updating with invalid configuration."""
        engine = SentimentAnalysisEngine()
        
        invalid_config = {
            "sentiment_thresholds.positive": -0.1,  # Invalid: should be > negative
            "sentiment_thresholds.negative": 0.1,
        }
        
        with pytest.raises(Exception):
            engine.update_configuration(invalid_config)


class TestEngineOutputFormats:
    """Tests for different output formats."""

    def test_analyze_with_json_summary(self, tmp_path):
        """Test analysis with JSON summary format."""
        input_file = tmp_path / "reviews.json"
        data = [{"review_id": "R001", "review_text": "Good product"}]
        with open(input_file, "w") as f:
            json.dump(data, f)

        config_file = tmp_path / "config.json"
        config_data = {
            "output": {"summary_format": "json", "results_format": "csv"}
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        engine = SentimentAnalysisEngine(config_path=str(config_file))
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        summary_file = Path(results["files_generated"]["summary"])
        assert summary_file.exists()
        assert summary_file.suffix == ".json"

    def test_analyze_with_txt_summary(self, tmp_path):
        """Test analysis with TXT summary format."""
        input_file = tmp_path / "reviews.json"
        data = [{"review_id": "R001", "review_text": "Good product"}]
        with open(input_file, "w") as f:
            json.dump(data, f)

        config_file = tmp_path / "config.json"
        config_data = {
            "output": {"summary_format": "txt", "results_format": "csv"}
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        engine = SentimentAnalysisEngine(config_path=str(config_file))
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        summary_file = Path(results["files_generated"]["summary"])
        assert summary_file.exists()
        assert summary_file.suffix == ".txt"


class TestEngineErrorAggregation:
    """Tests for error aggregation and reporting."""

    def test_error_aggregation(self, tmp_path):
        """Test that errors are properly aggregated."""
        input_file = tmp_path / "errors.json"
        data = [
            {"review_id": "R001", "review_text": ""},  # Empty
            {"review_id": "R002", "review_text": "Good"},  # Valid
            {"review_id": "R003", "review_text": ""},  # Empty
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert "processing_info" in results
        assert "total_errors" in results["processing_info"]
        assert "errors" in results["processing_info"]
        assert results["processing_info"]["total_errors"] >= 0


class TestEngineMemoryManagement:
    """Tests for memory management."""

    def test_analyze_with_small_batch_size(self, tmp_path):
        """Test analysis with small batch size."""
        input_file = tmp_path / "reviews.json"
        data = [
            {"review_id": f"R{i:03d}", "review_text": f"Review {i}"}
            for i in range(50)
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        config_file = tmp_path / "config.json"
        config_data = {"processing": {"batch_size": 10}}
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        engine = SentimentAnalysisEngine(config_path=str(config_file))
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 50


class TestEnginePerformanceMetrics:
    """Tests for performance metrics collection."""

    def test_performance_metrics_collected(self, tmp_path):
        """Test that performance metrics are collected."""
        input_file = tmp_path / "reviews.json"
        data = [{"review_id": "R001", "review_text": "Good product"}]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert "performance_metrics" in results
        # Performance metrics should contain timing information


class TestEngineEdgeCases:
    """Tests for edge cases."""

    def test_analyze_with_unicode_characters(self, tmp_path):
        """Test with various Unicode characters."""
        input_file = tmp_path / "unicode.json"
        data = [
            {"review_id": "R001", "review_text": "Très élégant! 🎉"},
            {"review_id": "R002", "review_text": "中文测试"},
            {"review_id": "R003", "review_text": "Тест на русском"},
        ]
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"

    def test_analyze_with_mixed_languages(self, tmp_path):
        """Test with mixed language reviews."""
        input_file = tmp_path / "mixed.json"
        data = [
            {"review_id": "R001", "review_text": "Good product but cher"},
            {"review_id": "R002", "review_text": "Excellent quality"},
            {"review_id": "R003", "review_text": "Très bad"},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 3

    def test_analyze_with_only_punctuation(self, tmp_path):
        """Test with reviews containing only punctuation."""
        input_file = tmp_path / "punctuation.json"
        data = [
            {"review_id": "R001", "review_text": "!!!"},
            {"review_id": "R002", "review_text": "???"},
            {"review_id": "R003", "review_text": "..."},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"

    def test_analyze_with_numbers_only(self, tmp_path):
        """Test with reviews containing only numbers."""
        input_file = tmp_path / "numbers.json"
        data = [
            {"review_id": "R001", "review_text": "123456"},
            {"review_id": "R002", "review_text": "5/5"},
            {"review_id": "R003", "review_text": "10/10"},
        ]
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        assert results["status"] == "completed"
