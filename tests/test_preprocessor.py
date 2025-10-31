"""Tests for text preprocessing functionality."""

import pytest
from src.preprocessing.preprocessor import ReviewPreprocessor
from src.models.review import Review


class TestReviewPreprocessor:
    """Test cases for ReviewPreprocessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = ReviewPreprocessor()

    def test_preprocess_normal_text(self):
        """Test preprocessing of normal text."""
        text = "This is a great product! I love it."
        result = self.preprocessor.preprocess_text(text)

        assert "great product" in result.lower()
        assert "love" in result.lower()
        assert len(result) > 0

    def test_preprocess_french_text(self):
        """Test preprocessing of French text with contractions."""
        text = "J'adore ce produit! C'est fantastique."
        result = self.preprocessor.preprocess_text(text)

        assert "je adore" in result or "j'adore" in result
        assert "ce est" in result or "c'est" in result

    def test_preprocess_emojis(self):
        """Test emoji processing."""
        text = "Great product! 😀👍⭐⭐⭐⭐⭐"
        result = self.preprocessor.preprocess_text(text)

        assert "happy" in result or "good" in result or "star" in result
        # Original emojis should be replaced or removed
        assert "😀" not in result
        assert "👍" not in result

    def test_preprocess_negative_emojis(self):
        """Test negative emoji processing."""
        text = "Terrible product 😞👎💔"
        result = self.preprocessor.preprocess_text(text)

        assert "sad" in result or "bad" in result or "heartbroken" in result

    def test_preprocess_urls_and_structured_data(self):
        """Test removal of URLs and structured data."""
        text = "Check http://example.com or email me at test@example.com. Code: XYZ-789"
        result = self.preprocessor.preprocess_text(text)

        assert "http://example.com" not in result
        assert "test@example.com" not in result
        assert "XYZ-789" not in result

    def test_preprocess_prices(self):
        """Test price handling."""
        text = "Bought for 19.99€ and it's worth it!"
        result = self.preprocessor.preprocess_text(text)

        assert "19.99€" not in result
        assert "prix" in result  # Should be replaced with French word for price

    def test_preprocess_empty_text(self):
        """Test handling of empty text."""
        review = Review(review_id="TEST001", review_text="")
        result = self.preprocessor.preprocess_review(review)

        assert result.review_text == ""
        assert result.has_errors()
        assert "empty" in result.processing_errors[0].lower()

    def test_preprocess_whitespace_only(self):
        """Test handling of whitespace-only text."""
        review = Review(review_id="TEST002", review_text="   \n\t  ")
        result = self.preprocessor.preprocess_review(review)

        assert result.review_text == ""
        assert result.has_errors()

    def test_preprocess_special_characters(self):
        """Test handling of special characters and encoding issues."""
        text = "Très bon produit! Ça marche parfaitement."
        result = self.preprocessor.preprocess_text(text)

        assert "très" in result.lower()
        assert "ça" in result.lower()
        assert len(result) > 0

    def test_normalize_punctuation(self):
        """Test punctuation normalization."""
        text = "Great!!!!! Really????"
        result = self.preprocessor.preprocess_text(text)

        # Multiple punctuation should be normalized
        assert "!!!!" not in result
        assert "????" not in result

    def test_preprocess_mixed_content(self):
        """Test preprocessing of mixed content with various elements."""
        text = (
            "🙄 Trop cher pour ce que c'est. Je suis déçu. (URL: http://site.com/bug)"
        )
        result = self.preprocessor.preprocess_text(text)

        assert "annoyed" in result or "🙄" not in result  # Emoji processed
        assert "ce est" in result or "c'est" in result  # Contraction handled
        assert "http://site.com/bug" not in result  # URL removed
        assert len(result) > 0

    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        text = "This is a test. It has two sentences!"
        stats = self.preprocessor.get_text_statistics(text)

        assert stats["word_count"] == 8
        assert stats["sentence_count"] == 2
        assert stats["character_count"] > 0

    def test_get_text_statistics_empty(self):
        """Test text statistics for empty text."""
        stats = self.preprocessor.get_text_statistics("")

        assert stats["word_count"] == 0
        assert stats["sentence_count"] == 0
        assert stats["character_count"] == 0

    def test_preprocess_reviews_batch(self):
        """Test batch preprocessing of multiple reviews."""
        reviews = [
            Review(review_id="REV001", review_text="Great product! 😀"),
            Review(review_id="REV002", review_text=""),
            Review(review_id="REV003", review_text="J'adore ce produit!"),
        ]

        results = self.preprocessor.preprocess_reviews(reviews)

        assert len(results) == 3
        assert results[0].review_text != "Great product! 😀"  # Should be processed
        assert results[1].has_errors()  # Empty text should have error
        assert (
            "je adore" in results[2].review_text or "j'adore" in results[2].review_text
        )
