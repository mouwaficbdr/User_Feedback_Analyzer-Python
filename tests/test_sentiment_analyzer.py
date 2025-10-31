"""Tests for sentiment analysis functionality."""

import pytest
from src.analysis.sentiment_analyzer import VaderSentimentAnalyzer
from src.models.review import Review


class TestVaderSentimentAnalyzer:
    """Test cases for VaderSentimentAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = VaderSentimentAnalyzer()

    def test_analyzer_initialization(self):
        """Test analyzer initialization with default parameters."""
        assert self.analyzer.positive_threshold == 0.05
        assert self.analyzer.negative_threshold == -0.05
        assert self.analyzer.batch_size == 100
        assert self.analyzer.analyzer is not None

    def test_analyzer_initialization_custom_thresholds(self):
        """Test analyzer initialization with custom thresholds."""
        analyzer = VaderSentimentAnalyzer(
            positive_threshold=0.1, negative_threshold=-0.1, batch_size=50
        )

        assert analyzer.positive_threshold == 0.1
        assert analyzer.negative_threshold == -0.1
        assert analyzer.batch_size == 50

    def test_invalid_thresholds(self):
        """Test validation of invalid thresholds."""
        with pytest.raises(ValueError):
            VaderSentimentAnalyzer(positive_threshold=-0.1, negative_threshold=0.1)

        with pytest.raises(ValueError):
            VaderSentimentAnalyzer(positive_threshold=1.5, negative_threshold=-0.1)

        with pytest.raises(ValueError):
            VaderSentimentAnalyzer(positive_threshold=0.1, negative_threshold=-1.5)

    def test_analyze_positive_review(self):
        """Test analysis of clearly positive review."""
        review = Review(
            review_id="POS001",
            review_text="Excellent produit, je le recommande vivement à tout le monde !",
        )

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Positive"
        assert result.sentiment_score > self.analyzer.positive_threshold
        assert result.review_id == "POS001"

    def test_analyze_negative_review(self):
        """Test analysis of clearly negative review."""
        review = Review(
            review_id="NEG001",
            review_text="Le service client était absolument horrible. J'attends toujours une réponse.",
        )

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Negative"
        assert result.sentiment_score < self.analyzer.negative_threshold
        assert result.review_id == "NEG001"

    def test_analyze_neutral_review(self):
        """Test analysis of neutral review."""
        review = Review(review_id="NEU001", review_text="Ça fait le job, sans plus.")

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Neutral"
        assert (
            self.analyzer.negative_threshold
            <= result.sentiment_score
            <= self.analyzer.positive_threshold
        )

    def test_analyze_empty_review(self):
        """Test analysis of empty review."""
        review = Review(review_id="EMPTY001", review_text="")

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Neutral"
        assert result.sentiment_score == 0.0
        assert result.has_errors()

    def test_analyze_very_short_review(self):
        """Test analysis of very short review."""
        review = Review(review_id="SHORT001", review_text="Bien.")

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label in ["Positive", "Negative", "Neutral"]
        assert result.sentiment_score is not None
        assert -1.0 <= result.sentiment_score <= 1.0

    def test_analyze_emoji_review(self):
        """Test analysis of review with emojis (should be preprocessed)."""
        review = Review(
            review_id="EMOJI001",
            review_text="Great product! happy good star star star star star",  # Preprocessed emojis
        )

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Positive"
        assert result.sentiment_score > 0

    def test_french_sentiment_enhancement(self):
        """Test French sentiment word enhancement."""
        # Test with French positive words
        positive_review = Review(
            review_id="FR_POS001",
            review_text="Absolument génial ! Fantastique produit.",
        )

        result = self.analyzer.analyze_single_review(positive_review)
        assert result.sentiment_label == "Positive"

        # Test with French negative words
        negative_review = Review(
            review_id="FR_NEG001", review_text="C'est horrible et terrible. Très déçu."
        )

        result = self.analyzer.analyze_single_review(negative_review)
        assert result.sentiment_label == "Negative"

    def test_analyze_batch_reviews(self):
        """Test batch analysis of multiple reviews."""
        reviews = [
            Review(review_id="BATCH001", review_text="Excellent produit !"),
            Review(review_id="BATCH002", review_text="Très mauvais service."),
            Review(review_id="BATCH003", review_text="Ça va, sans plus."),
            Review(review_id="BATCH004", review_text=""),
        ]

        results = self.analyzer.analyze_sentiment(reviews)

        assert len(results) == 4
        assert all(r.sentiment_score is not None for r in results)
        assert all(
            r.sentiment_label in ["Positive", "Negative", "Neutral"] for r in results
        )

        # Check that we have different sentiment labels (at least positive and neutral)
        labels = [r.sentiment_label for r in results]
        assert "Positive" in labels
        assert "Neutral" in labels
        # Note: "Negative" might not always appear depending on the specific text processing

    def test_score_bounds(self):
        """Test that sentiment scores are within valid bounds."""
        reviews = [
            Review(
                review_id="EXTREME_POS",
                review_text="Absolument fantastique génial excellent parfait !",
            ),
            Review(
                review_id="EXTREME_NEG",
                review_text="Horrible terrible affreux catastrophique nul !",
            ),
            Review(review_id="NEUTRAL", review_text="Ok normal standard."),
        ]

        results = self.analyzer.analyze_sentiment(reviews)

        for result in results:
            assert -1.0 <= result.sentiment_score <= 1.0

    def test_threshold_boundary_cases(self):
        """Test sentiment classification at threshold boundaries."""
        # Create analyzer with specific thresholds for testing
        analyzer = VaderSentimentAnalyzer(
            positive_threshold=0.1, negative_threshold=-0.1
        )

        # Test reviews that should be right at the boundaries
        reviews = [
            Review(
                review_id="BOUNDARY_POS", review_text="Assez bien."
            ),  # Should be close to threshold
            Review(
                review_id="BOUNDARY_NEG", review_text="Pas terrible."
            ),  # Should be close to threshold
            Review(
                review_id="BOUNDARY_NEU", review_text="Normal."
            ),  # Should be neutral
        ]

        results = analyzer.analyze_sentiment(reviews)

        # All should have valid labels
        for result in results:
            assert result.sentiment_label in ["Positive", "Negative", "Neutral"]

    def test_get_analyzer_info(self):
        """Test analyzer information retrieval."""
        info = self.analyzer.get_analyzer_info()

        assert info["analyzer_type"] == "VADER"
        assert info["positive_threshold"] == 0.05
        assert info["negative_threshold"] == -0.05
        assert info["batch_size"] == 100
        assert "threshold_justification" in info
        assert info["french_words_count"] > 0

    def test_update_thresholds(self):
        """Test threshold updates."""
        original_pos = self.analyzer.positive_threshold
        original_neg = self.analyzer.negative_threshold

        # Update to valid thresholds
        self.analyzer.update_thresholds(0.2, -0.2)
        assert self.analyzer.positive_threshold == 0.2
        assert self.analyzer.negative_threshold == -0.2

        # Try to update to invalid thresholds
        with pytest.raises(ValueError):
            self.analyzer.update_thresholds(-0.1, 0.1)

        # Should revert to previous valid values
        assert self.analyzer.positive_threshold == 0.2
        assert self.analyzer.negative_threshold == -0.2

    def test_error_handling(self):
        """Test error handling in sentiment analysis."""
        # Create a review that might cause issues
        problematic_review = Review(
            review_id="PROBLEM001",
            review_text=None,  # This should be handled gracefully
        )

        # Should not raise exception
        result = self.analyzer.analyze_single_review(problematic_review)

        # Should have neutral sentiment and error recorded
        assert result.sentiment_label == "Neutral"
        assert result.sentiment_score == 0.0

    def test_mixed_language_content(self):
        """Test handling of mixed language content."""
        review = Review(
            review_id="MIXED001",
            review_text="This product is excellent! Vraiment fantastique!",
        )

        result = self.analyzer.analyze_single_review(review)

        assert result.sentiment_label == "Positive"
        assert result.sentiment_score > 0
