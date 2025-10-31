"""Tests for data loading functionality."""

import json
import pytest
import tempfile
import os
from pathlib import Path

from src.data.loader import JSONDataLoader
from src.models.review import Review


class TestJSONDataLoader:
    """Test cases for JSONDataLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = JSONDataLoader()

    def test_load_valid_reviews(self):
        """Test loading valid review data."""
        test_data = [
            {"review_id": "REV001", "review_text": "Great product!"},
            {"review_id": "REV002", "review_text": "Not so good."},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            reviews = self.loader.load_reviews(temp_path)

            assert len(reviews) == 2
            assert reviews[0].review_id == "REV001"
            assert reviews[0].review_text == "Great product!"
            assert reviews[1].review_id == "REV002"
            assert reviews[1].review_text == "Not so good."
        finally:
            os.unlink(temp_path)

    def test_load_empty_review_text(self):
        """Test handling of empty review text."""
        test_data = [
            {"review_id": "REV001", "review_text": ""},
            {"review_id": "REV002", "review_text": None},
            {"review_id": "REV003", "review_text": "   "},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            reviews = self.loader.load_reviews(temp_path)

            assert len(reviews) == 3
            assert all(
                review.review_text == "" or review.review_text.strip() == ""
                for review in reviews
            )
            assert self.loader.has_loading_errors()
        finally:
            os.unlink(temp_path)

    def test_missing_review_id(self):
        """Test handling of missing review IDs."""
        test_data = [
            {"review_text": "Review without ID"},
            {"id": "ALT001", "review_text": "Review with alternative ID field"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            reviews = self.loader.load_reviews(temp_path)

            assert len(reviews) == 2
            assert reviews[0].review_id == "REVIEW_001"  # Generated ID
            assert reviews[1].review_id == "ALT001"  # Alternative field
            assert self.loader.has_loading_errors()
        finally:
            os.unlink(temp_path)

    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Could not load JSON file"):
                self.loader.load_reviews(temp_path)
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_reviews("nonexistent_file.json")

    def test_different_json_structures(self):
        """Test handling of different JSON structures."""
        # Test wrapped structure
        wrapped_data = {
            "reviews": [{"review_id": "REV001", "review_text": "Wrapped review"}]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(wrapped_data, f)
            temp_path = f.name

        try:
            reviews = self.loader.load_reviews(temp_path)
            assert len(reviews) == 1
            assert reviews[0].review_text == "Wrapped review"
        finally:
            os.unlink(temp_path)

    def test_encoding_handling(self):
        """Test handling of different encodings."""
        # Create file with special characters
        test_data = [{"review_id": "REV001", "review_text": "Très bon produit! 🙂"}]

        # Test UTF-8 encoding
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            reviews = self.loader.load_reviews(temp_path)
            assert len(reviews) == 1
            assert "Très bon" in reviews[0].review_text
            assert "🙂" in reviews[0].review_text
        finally:
            os.unlink(temp_path)
