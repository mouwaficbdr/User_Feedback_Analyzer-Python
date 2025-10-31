"""Integration tests for the complete sentiment analysis pipeline."""

import json
import csv
import pytest
import tempfile
import os
from pathlib import Path

from src.engine import SentimentAnalysisEngine


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def create_test_reviews_file(self, reviews_data, file_path):
        """Helper to create test reviews JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, ensure_ascii=False, indent=2)

    def test_complete_pipeline_success(self):
        """Test complete pipeline with valid data."""
        test_reviews = [
            {"review_id": "TEST001", "review_text": "Excellent produit !"},
            {"review_id": "TEST002", "review_text": "Service horrible."},
            {"review_id": "TEST003", "review_text": "Ça va."},
            {"review_id": "TEST004", "review_text": ""},  # Empty text
            {"review_id": "TEST005", "review_text": "Très bon ! 😀"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test input file
            input_file = os.path.join(temp_dir, "test_reviews.json")
            self.create_test_reviews_file(test_reviews, input_file)

            # Create config file
            config_file = os.path.join(temp_dir, "config.json")

            # Initialize engine
            engine = SentimentAnalysisEngine(config_path=config_file)

            # Run analysis
            results = engine.analyze_reviews(input_file, temp_dir)

            # Verify results structure
            assert "status" in results
            assert results["status"] == "completed"
            assert "statistics" in results
            assert "files_generated" in results
            assert "processing_info" in results

            # Verify statistics
            stats = results["statistics"]
            assert stats["total_reviews"] == 5
            assert (
                stats["positive_count"]
                + stats["negative_count"]
                + stats["neutral_count"]
                == 5
            )

            # Verify files were generated
            files = results["files_generated"]
            assert "summary" in files
            assert "results" in files
            assert os.path.exists(files["summary"])
            assert os.path.exists(files["results"])

            # Verify summary file content
            with open(files["summary"], "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            assert "analysis_summary" in summary_data
            assert summary_data["analysis_summary"]["total_reviews"] == 5
            assert "sentiment_distribution" in summary_data["analysis_summary"]

            # Verify results file content
            with open(files["results"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 5
            assert all("review_id" in row for row in rows)
            assert all("sentiment_final" in row for row in rows)
            assert all("sentiment_score" in row for row in rows)

    def test_pipeline_with_edge_cases(self):
        """Test pipeline with various edge cases."""
        test_reviews = [
            {"review_id": "EDGE001", "review_text": ""},  # Empty
            {"review_id": "EDGE002", "review_text": "   "},  # Whitespace only
            {"review_id": "EDGE003", "review_text": "🙄😒💔"},  # Emojis only
            {"review_id": "EDGE004", "review_text": "http://example.com"},  # URL only
            {"review_id": "EDGE005", "review_text": "A" * 1000},  # Very long text
            {
                "review_id": "EDGE006",
                "review_text": "Très très très bon produit ! ⭐⭐⭐⭐⭐",
            },  # Mixed content
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "edge_cases.json")
            self.create_test_reviews_file(test_reviews, input_file)

            engine = SentimentAnalysisEngine()
            results = engine.analyze_reviews(input_file, temp_dir)

            # Should complete successfully despite edge cases
            assert results["status"] == "completed"
            assert results["statistics"]["total_reviews"] == 6

            # Check that all reviews were processed
            files = results["files_generated"]
            with open(files["results"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 6

            # All reviews should have sentiment labels
            for row in rows:
                assert row["sentiment_final"] in ["Positive", "Negative", "Neutral"]

    def test_pipeline_with_different_json_structures(self):
        """Test pipeline with different JSON input structures."""
        # Test wrapped structure
        wrapped_data = {
            "reviews": [
                {"review_id": "WRAP001", "review_text": "Wrapped review positive"},
                {"review_id": "WRAP002", "review_text": "Wrapped review negative"},
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "wrapped.json")
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(wrapped_data, f)

            engine = SentimentAnalysisEngine()
            results = engine.analyze_reviews(input_file, temp_dir)

            assert results["status"] == "completed"
            assert results["statistics"]["total_reviews"] == 2

    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        test_reviews = [
            {"review_id": "CONFIG001", "review_text": "Slightly positive"},
            {"review_id": "CONFIG002", "review_text": "Slightly negative"},
        ]

        custom_config = {
            "sentiment_thresholds": {
                "positive": 0.3,  # Higher threshold
                "negative": -0.3,  # Lower threshold
            },
            "output": {"summary_format": "txt"},  # Text format instead of JSON
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = os.path.join(temp_dir, "config_test.json")
            self.create_test_reviews_file(test_reviews, input_file)

            # Create custom config file
            config_file = os.path.join(temp_dir, "custom_config.json")
            with open(config_file, "w") as f:
                json.dump(custom_config, f)

            engine = SentimentAnalysisEngine(config_path=config_file)
            results = engine.analyze_reviews(input_file, temp_dir)

            assert results["status"] == "completed"

            # Check that summary is in text format
            files = results["files_generated"]
            assert files["summary"].endswith(".txt")

            # Verify custom thresholds were used
            processing_info = results["processing_info"]
            config_info = processing_info["configuration"]
            assert config_info["positive_threshold"] == 0.3
            assert config_info["negative_threshold"] == -0.3

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with problematic data."""
        # Create file with some problematic entries
        problematic_data = [
            {"review_id": "PROB001", "review_text": "Normal review"},
            {"review_text": "Missing ID"},  # Missing review_id
            {"review_id": "PROB003"},  # Missing review_text
            {"review_id": "PROB004", "review_text": None},  # Null text
            123,  # Not a dictionary
            {"review_id": "PROB006", "review_text": "Final good review"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "problematic.json")
            with open(input_file, "w") as f:
                json.dump(problematic_data, f)

            engine = SentimentAnalysisEngine()
            results = engine.analyze_reviews(input_file, temp_dir)

            # Should complete despite errors
            assert results["status"] == "completed"

            # Should have processed some reviews
            assert results["statistics"]["total_reviews"] > 0

            # Should have recorded errors
            assert results["processing_info"]["total_errors"] > 0

    def test_pipeline_memory_efficiency(self):
        """Test pipeline with larger dataset for memory efficiency."""
        # Create a larger dataset
        large_dataset = []
        for i in range(200):  # 200 reviews
            sentiment_text = (
                "positive" if i % 3 == 0 else "negative" if i % 3 == 1 else "neutral"
            )
            large_dataset.append(
                {
                    "review_id": f"LARGE{i:03d}",
                    "review_text": f"This is a {sentiment_text} review number {i}",
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "large_dataset.json")
            self.create_test_reviews_file(large_dataset, input_file)

            engine = SentimentAnalysisEngine()
            results = engine.analyze_reviews(input_file, temp_dir)

            assert results["status"] == "completed"
            assert results["statistics"]["total_reviews"] == 200

            # Verify all reviews were processed
            files = results["files_generated"]
            with open(files["results"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 200

    def test_pipeline_validation_errors(self):
        """Test pipeline with validation errors."""
        engine = SentimentAnalysisEngine()

        # Test with non-existent file
        validation_errors = engine.validate_input_file("nonexistent.json")
        assert len(validation_errors) > 0
        assert any("does not exist" in error for error in validation_errors)

    def test_engine_info(self):
        """Test engine information retrieval."""
        engine = SentimentAnalysisEngine()
        info = engine.get_engine_info()

        assert "version" in info
        assert "components" in info
        assert "configuration" in info
        assert "analyzer_info" in info
        assert "report_info" in info

        # Verify component information
        components = info["components"]
        assert "data_loader" in components
        assert "preprocessor" in components
        assert "sentiment_analyzer" in components
        assert "report_generator" in components

    def test_configuration_updates(self):
        """Test engine configuration updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            engine = SentimentAnalysisEngine(config_path=config_file)

            # Update configuration
            config_updates = {
                "sentiment_thresholds": {"positive": 0.2, "negative": -0.2}
            }

            engine.update_configuration(config_updates)

            # Verify updates were applied
            info = engine.get_engine_info()
            analyzer_info = info["analyzer_info"]
            assert analyzer_info["positive_threshold"] == 0.2
            assert analyzer_info["negative_threshold"] == -0.2
