"""Tests for data models."""

import pytest
from src.models.review import Review, SentimentResult


class TestReview:
    """Test cases for Review model."""

    def test_review_creation(self):
        """Test basic review creation."""
        review = Review(review_id="TEST001", review_text="Test review text")

        assert review.review_id == "TEST001"
        assert review.review_text == "Test review text"
        assert review.sentiment_score is None
        assert review.sentiment_label is None
        assert review.processing_errors == []

    def test_review_with_sentiment(self):
        """Test review creation with sentiment data."""
        review = Review(
            review_id="TEST002",
            review_text="Great product!",
            sentiment_score=0.8,
            sentiment_label="Positive",
        )

        assert review.sentiment_score == 0.8
        assert review.sentiment_label == "Positive"

    def test_review_validation(self):
        """Test review validation."""
        # Empty review_id should raise error
        with pytest.raises(ValueError, match="review_id cannot be empty"):
            Review(review_id="", review_text="Test")

        # None review_text should be converted to empty string
        review = Review(review_id="TEST003", review_text=None)
        assert review.review_text == ""

    def test_add_error(self):
        """Test adding processing errors."""
        review = Review(review_id="TEST004", review_text="Test")

        review.add_error("Test error")
        assert len(review.processing_errors) == 1
        assert "Test error" in review.processing_errors

        review.add_error("Another error")
        assert len(review.processing_errors) == 2

    def test_has_errors(self):
        """Test error checking."""
        review = Review(review_id="TEST005", review_text="Test")

        assert not review.has_errors()

        review.add_error("Test error")
        assert review.has_errors()

    def test_is_empty_text(self):
        """Test empty text detection."""
        # Empty string
        review1 = Review(review_id="TEST006", review_text="")
        assert review1.is_empty_text()

        # Whitespace only
        review2 = Review(review_id="TEST007", review_text="   \n\t  ")
        assert review2.is_empty_text()

        # Normal text
        review3 = Review(review_id="TEST008", review_text="Normal text")
        assert not review3.is_empty_text()


class TestSentimentResult:
    """Test cases for SentimentResult model."""

    def test_sentiment_result_creation(self):
        """Test basic sentiment result creation."""
        result = SentimentResult(
            total_reviews=10,
            positive_count=4,
            negative_count=3,
            neutral_count=3,
            positive_percentage=40.0,
            negative_percentage=30.0,
            neutral_percentage=30.0,
        )

        assert result.total_reviews == 10
        assert result.positive_count == 4
        assert result.negative_count == 3
        assert result.neutral_count == 3
        assert result.positive_percentage == 40.0
        assert result.negative_percentage == 30.0
        assert result.neutral_percentage == 30.0

    def test_sentiment_result_validation(self):
        """Test sentiment result validation."""
        # Negative total reviews
        with pytest.raises(ValueError, match="total_reviews cannot be negative"):
            SentimentResult(
                total_reviews=-1,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                positive_percentage=0.0,
                negative_percentage=0.0,
                neutral_percentage=0.0,
            )

        # Counts don't match total
        with pytest.raises(ValueError, match="don't match total reviews"):
            SentimentResult(
                total_reviews=10,
                positive_count=4,
                negative_count=3,
                neutral_count=2,  # Should be 3 to sum to 10
                positive_percentage=40.0,
                negative_percentage=30.0,
                neutral_percentage=20.0,
            )

        # Percentages don't sum to 100
        with pytest.raises(ValueError, match="don't sum to 100"):
            SentimentResult(
                total_reviews=10,
                positive_count=4,
                negative_count=3,
                neutral_count=3,
                positive_percentage=40.0,
                negative_percentage=30.0,
                neutral_percentage=20.0,  # Should be 30.0
            )

    def test_from_reviews_empty(self):
        """Test creating SentimentResult from empty reviews list."""
        result = SentimentResult.from_reviews([])

        assert result.total_reviews == 0
        assert result.positive_count == 0
        assert result.negative_count == 0
        assert result.neutral_count == 0
        assert result.positive_percentage == 0.0
        assert result.negative_percentage == 0.0
        assert result.neutral_percentage == 0.0

    def test_from_reviews_with_data(self):
        """Test creating SentimentResult from reviews with sentiment data."""
        reviews = [
            Review(review_id="R1", review_text="Great!", sentiment_label="Positive"),
            Review(review_id="R2", review_text="Bad", sentiment_label="Negative"),
            Review(review_id="R3", review_text="OK", sentiment_label="Neutral"),
            Review(review_id="R4", review_text="Good", sentiment_label="Positive"),
        ]

        result = SentimentResult.from_reviews(reviews)

        assert result.total_reviews == 4
        assert result.positive_count == 2
        assert result.negative_count == 1
        assert result.neutral_count == 1
        assert result.positive_percentage == 50.0
        assert result.negative_percentage == 25.0
        assert result.neutral_percentage == 25.0

    def test_from_reviews_with_errors(self):
        """Test creating SentimentResult from reviews with processing errors."""
        reviews = [
            Review(review_id="R1", review_text="Test", sentiment_label="Positive"),
            Review(
                review_id="R2",
                review_text="",
                sentiment_label="Neutral",
                processing_errors=["Empty text"],
            ),
        ]

        result = SentimentResult.from_reviews(reviews)

        assert result.total_reviews == 2
        assert len(result.processing_errors) == 1
        assert "Empty text" in result.processing_errors

    def test_percentage_rounding(self):
        """Test percentage calculation and rounding."""
        # Create reviews that would result in repeating decimals
        reviews = [
            Review(review_id="R1", review_text="1", sentiment_label="Positive"),
            Review(review_id="R2", review_text="2", sentiment_label="Positive"),
            Review(review_id="R3", review_text="3", sentiment_label="Negative"),
        ]

        result = SentimentResult.from_reviews(reviews)

        # Should handle rounding properly
        assert result.total_reviews == 3
        assert result.positive_count == 2
        assert result.negative_count == 1
        assert result.neutral_count == 0

        # Percentages should sum to exactly 100.0
        total_percentage = (
            result.positive_percentage
            + result.negative_percentage
            + result.neutral_percentage
        )
        assert total_percentage == 100.0
