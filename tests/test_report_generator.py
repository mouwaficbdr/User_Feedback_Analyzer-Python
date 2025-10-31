"""Tests for report generation functionality."""

import json
import csv
import pytest
import tempfile
import os
from pathlib import Path

from src.reporting.report_generator import ReportGenerator
from src.models.review import Review, SentimentResult


class TestReportGenerator:
    """Test cases for ReportGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

        # Create sample reviews for testing
        self.sample_reviews = [
            Review(
                review_id="REV001",
                review_text="Excellent produit !",
                sentiment_score=0.8,
                sentiment_label="Positive",
            ),
            Review(
                review_id="REV002",
                review_text="Service horrible.",
                sentiment_score=-0.7,
                sentiment_label="Negative",
            ),
            Review(
                review_id="REV003",
                review_text="Ça va.",
                sentiment_score=0.0,
                sentiment_label="Neutral",
            ),
            Review(
                review_id="REV004",
                review_text="",
                sentiment_score=0.0,
                sentiment_label="Neutral",
                processing_errors=["Empty text"],
            ),
        ]

    def test_generator_initialization(self):
        """Test generator initialization with default parameters."""
        assert self.generator.summary_format == "json"
        assert self.generator.results_format == "csv"
        assert self.generator.summary_filename == "summary"
        assert self.generator.results_filename == "results"

    def test_generator_initialization_custom(self):
        """Test generator initialization with custom parameters."""
        generator = ReportGenerator(
            summary_format="txt",
            results_format="csv",
            summary_filename="custom_summary",
            results_filename="custom_results",
        )

        assert generator.summary_format == "txt"
        assert generator.results_format == "csv"
        assert generator.summary_filename == "custom_summary"
        assert generator.results_filename == "custom_results"

    def test_invalid_formats(self):
        """Test validation of invalid formats."""
        with pytest.raises(ValueError):
            ReportGenerator(summary_format="invalid")

        with pytest.raises(ValueError):
            ReportGenerator(results_format="invalid")

    def test_calculate_statistics(self):
        """Test sentiment statistics calculation."""
        result = self.generator.calculate_statistics(self.sample_reviews)

        assert result.total_reviews == 4
        assert result.positive_count == 1
        assert result.negative_count == 1
        assert result.neutral_count == 2
        assert result.positive_percentage == 25.0
        assert result.negative_percentage == 25.0
        assert result.neutral_percentage == 50.0

    def test_generate_json_summary(self):
        """Test JSON summary report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sentiment_result = SentimentResult.from_reviews(self.sample_reviews)

            analyzer_info = {
                "analyzer_type": "VADER",
                "positive_threshold": 0.05,
                "negative_threshold": -0.05,
                "threshold_justification": "Test justification",
            }

            summary_path = self.generator.generate_summary_report(
                sentiment_result, temp_dir, analyzer_info
            )

            # Verify file was created
            assert os.path.exists(summary_path)
            assert summary_path.endswith("summary.json")

            # Verify content
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "analysis_summary" in data
            summary = data["analysis_summary"]

            assert summary["total_reviews"] == 4
            assert summary["sentiment_distribution"]["positive"]["count"] == 1
            assert summary["sentiment_distribution"]["negative"]["count"] == 1
            assert summary["sentiment_distribution"]["neutral"]["count"] == 2
            assert summary["processing_info"]["errors_count"] == 1
            assert "configuration" in summary

    def test_generate_txt_summary(self):
        """Test text summary report generation."""
        generator = ReportGenerator(summary_format="txt")

        with tempfile.TemporaryDirectory() as temp_dir:
            sentiment_result = SentimentResult.from_reviews(self.sample_reviews)

            summary_path = generator.generate_summary_report(sentiment_result, temp_dir)

            # Verify file was created
            assert os.path.exists(summary_path)
            assert summary_path.endswith("summary.txt")

            # Verify content
            with open(summary_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "SENTIMENT ANALYSIS SUMMARY REPORT" in content
            assert "Total Reviews Analyzed: 4" in content
            assert "Positive: 1 reviews (25.0%)" in content
            assert "Negative: 1 reviews (25.0%)" in content
            assert "Neutral:  2 reviews (50.0%)" in content

    def test_generate_detailed_csv_report(self):
        """Test detailed CSV report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = self.generator.generate_detailed_report(
                self.sample_reviews, temp_dir
            )

            # Verify file was created
            assert os.path.exists(results_path)
            assert results_path.endswith("results.csv")

            # Verify content
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 4

            # Check headers
            expected_headers = [
                "review_id",
                "review_text",
                "sentiment_final",
                "sentiment_score",
                "processing_errors",
            ]
            assert list(rows[0].keys()) == expected_headers

            # Check first row
            first_row = rows[0]
            assert first_row["review_id"] == "REV001"
            assert first_row["sentiment_final"] == "Positive"
            assert first_row["sentiment_score"] == "0.8"
            assert first_row["processing_errors"] == ""

            # Check row with error
            error_row = next(row for row in rows if row["review_id"] == "REV004")
            assert error_row["processing_errors"] == "Empty text"

    def test_generate_reports_complete(self):
        """Test complete report generation (both summary and detailed)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer_info = {
                "analyzer_type": "VADER",
                "positive_threshold": 0.05,
                "negative_threshold": -0.05,
            }

            generated_files = self.generator.generate_reports(
                self.sample_reviews, temp_dir, analyzer_info
            )

            # Verify both files were generated
            assert "summary" in generated_files
            assert "results" in generated_files

            assert os.path.exists(generated_files["summary"])
            assert os.path.exists(generated_files["results"])

            assert generated_files["summary"].endswith("summary.json")
            assert generated_files["results"].endswith("results.csv")

    def test_clean_text_for_csv(self):
        """Test text cleaning for CSV output."""
        # Test normal text
        clean_text = self.generator._clean_text_for_csv("Normal text here")
        assert clean_text == "Normal text here"

        # Test text with newlines
        text_with_newlines = "Line 1\nLine 2\n\nLine 3"
        clean_text = self.generator._clean_text_for_csv(text_with_newlines)
        assert "\n" not in clean_text
        assert clean_text == "Line 1 Line 2 Line 3"

        # Test very long text
        long_text = "A" * 600
        clean_text = self.generator._clean_text_for_csv(long_text, max_length=100)
        assert len(clean_text) <= 100
        assert clean_text.endswith("...")

        # Test empty text
        clean_text = self.generator._clean_text_for_csv("")
        assert clean_text == ""

    def test_validate_reviews(self):
        """Test review validation."""
        # Test valid reviews
        errors = self.generator.validate_reviews(self.sample_reviews)
        assert len(errors) == 0

        # Test empty reviews list
        errors = self.generator.validate_reviews([])
        assert len(errors) == 1
        assert "No reviews provided" in errors[0]

        # Test invalid review (create manually to bypass Review validation)
        invalid_reviews = [
            Review(
                review_id="VALID001",
                review_text="Test",
                sentiment_score=2.0,
                sentiment_label="Invalid",
            )
        ]

        errors = self.generator.validate_reviews(invalid_reviews)
        assert len(errors) >= 1  # Should have validation errors

    def test_get_report_info(self):
        """Test report generator information."""
        info = self.generator.get_report_info()

        assert info["summary_format"] == "json"
        assert info["results_format"] == "csv"
        assert info["summary_filename"] == "summary"
        assert info["results_filename"] == "results"
        assert "supported_summary_formats" in info
        assert "supported_results_formats" in info

    def test_empty_reviews_handling(self):
        """Test handling of empty reviews list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle empty list gracefully
            generated_files = self.generator.generate_reports([], temp_dir)

            assert "summary" in generated_files
            assert "results" in generated_files

            # Check summary content
            with open(generated_files["summary"], "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["analysis_summary"]["total_reviews"] == 0

            # Check CSV content
            with open(generated_files["results"], "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 1  # Only header row

    def test_special_characters_in_text(self):
        """Test handling of special characters in review text."""
        special_reviews = [
            Review(
                review_id="SPECIAL001",
                review_text="Très bon produit! 🙂 Ça marche à 100%.",
                sentiment_score=0.5,
                sentiment_label="Positive",
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            generated_files = self.generator.generate_reports(special_reviews, temp_dir)

            # Verify files were created without errors
            assert os.path.exists(generated_files["summary"])
            assert os.path.exists(generated_files["results"])

            # Check that special characters are preserved in JSON
            with open(generated_files["summary"], "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["analysis_summary"]["total_reviews"] == 1

            # Check CSV handling
            with open(generated_files["results"], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert "Très bon" in rows[0]["review_text"]
