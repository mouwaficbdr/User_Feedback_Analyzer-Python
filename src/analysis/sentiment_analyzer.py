"""Sentiment analysis functionality."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..models.review import Review
from ..utils.logger import handle_errors, log_execution_time
from ..utils.performance import BatchProcessor


class SentimentAnalyzerInterface(ABC):
    """Abstract interface for sentiment analyzers."""

    @abstractmethod
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """
        Analyze sentiment for a list of reviews.

        Args:
            reviews: List of Review objects to analyze

        Returns:
            List of Review objects with sentiment scores and labels
        """
        pass

    @abstractmethod
    def analyze_single_review(self, review: Review) -> Review:
        """
        Analyze sentiment for a single review.

        Args:
            review: Review object to analyze

        Returns:
            Review object with sentiment score and label
        """
        pass


class VaderSentimentAnalyzer(SentimentAnalyzerInterface):
    """
    VADER-based sentiment analyzer with configurable thresholds.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is particularly
    effective for social media text, handling emojis, punctuation, and
    capitalization for sentiment intensity.
    """

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        batch_size: int = 100,
    ):
        """
        Initialize VADER sentiment analyzer.

        Args:
            positive_threshold: Minimum compound score for positive classification
            negative_threshold: Maximum compound score for negative classification
            batch_size: Number of reviews to process in each batch
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # Initialize VADER analyzer
        try:
            self.analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VADER analyzer: {e}")
            raise RuntimeError(f"Could not initialize sentiment analyzer: {e}")

        # Validate thresholds
        self._validate_thresholds()
        
        # Initialize batch processor for large datasets
        self.batch_processor = BatchProcessor(batch_size=batch_size)

        # French sentiment words to enhance VADER for French text
        self.french_sentiment_words = {
            # Positive words
            "excellent": 3.0,
            "fantastique": 3.0,
            "génial": 3.0,
            "parfait": 3.0,
            "magnifique": 3.0,
            "superbe": 3.0,
            "formidable": 3.0,
            "merveilleux": 3.0,
            "extraordinaire": 3.0,
            "incroyable": 2.5,
            "remarquable": 2.5,
            "impressionnant": 2.5,
            "bon": 2.0,
            "bien": 2.0,
            "agréable": 2.0,
            "satisfait": 2.0,
            "content": 2.0,
            "heureux": 2.0,
            "ravi": 2.5,
            "adore": 3.0,
            "aime": 2.0,
            "recommande": 2.0,
            "facile": 1.5,
            "simple": 1.5,
            "rapide": 1.5,
            "efficace": 2.0,
            "utile": 1.5,
            "pratique": 1.5,
            "qualité": 2.0,
            "merci": 1.5,
            # Negative words
            "horrible": -3.0,
            "terrible": -3.0,
            "affreux": -3.0,
            "épouvantable": -3.0,
            "catastrophique": -3.0,
            "scandaleux": -3.0,
            "inadmissible": -3.0,
            "inacceptable": -3.0,
            "inutilisable": -3.0,
            "nul": -2.5,
            "mauvais": -2.0,
            "médiocre": -2.0,
            "décevant": -2.0,
            "frustrant": -2.0,
            "ennuyeux": -1.5,
            "difficile": -1.5,
            "compliqué": -1.5,
            "lent": -1.5,
            "cher": -1.0,
            "déçu": -2.0,
            "mécontent": -2.0,
            "fâché": -2.0,
            "énervé": -2.0,
            "triste": -2.0,
            "problème": -1.5,
            "erreur": -1.5,
            "bug": -2.0,
            "panne": -2.0,
            "cassé": -2.5,
            "défaillant": -2.0,
            "défectueux": -2.5,
            # Neutral/context words
            "ok": 0.0,
            "correct": 0.5,
            "normal": 0.0,
            "standard": 0.0,
            "moyen": 0.0,
            "ordinaire": 0.0,
            "quelconque": -0.5,
            "sans": 0.0,
            "rien": 0.0,
            "neutre": 0.0,
        }

    def _validate_thresholds(self) -> None:
        """Validate sentiment classification thresholds."""
        if self.positive_threshold <= self.negative_threshold:
            raise ValueError(
                f"Positive threshold ({self.positive_threshold}) must be greater than "
                f"negative threshold ({self.negative_threshold})"
            )

        if not -1.0 <= self.negative_threshold <= 1.0:
            raise ValueError(f"Negative threshold must be between -1.0 and 1.0")

        if not -1.0 <= self.positive_threshold <= 1.0:
            raise ValueError(f"Positive threshold must be between -1.0 and 1.0")

    @log_execution_time()
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """
        Analyze sentiment for a list of reviews with optimized batch processing.

        Args:
            reviews: List of Review objects to analyze

        Returns:
            List of Review objects with sentiment scores and labels
        """
        self.logger.info(f"Starting sentiment analysis for {len(reviews)} reviews")

        if len(reviews) <= self.batch_size:
            # Small dataset - process directly
            analyzed_reviews = [self.analyze_single_review(review) for review in reviews]
        else:
            # Large dataset - use optimized batch processing
            analyzed_reviews = self.batch_processor.process_in_batches(
                reviews, self._process_batch
            )

        # Log summary statistics
        self._log_analysis_summary(analyzed_reviews)

        return analyzed_reviews
    
    def _process_batch(self, batch: List[Review]) -> List[Review]:
        """
        Process a batch of reviews for sentiment analysis.
        
        Args:
            batch: List of Review objects to process
            
        Returns:
            List of analyzed Review objects
        """
        return [self.analyze_single_review(review) for review in batch]

    @handle_errors(reraise=False)
    def analyze_single_review(self, review: Review) -> Review:
        """
        Analyze sentiment for a single review.

        Args:
            review: Review object to analyze

        Returns:
            Review object with sentiment score and label
        """
        try:
            # Handle empty text
            if review.is_empty_text():
                return self._handle_empty_review(review)

            # Get VADER sentiment scores
            sentiment_scores = self.analyzer.polarity_scores(review.review_text)

            # Enhance with French sentiment words
            enhanced_score = self._enhance_with_french_sentiment(
                review.review_text, sentiment_scores["compound"]
            )

            # Classify sentiment based on thresholds
            sentiment_label = self._classify_sentiment(enhanced_score)

            # Create analyzed review
            analyzed_review = Review(
                review_id=review.review_id,
                review_text=review.review_text,
                sentiment_score=round(enhanced_score, 4),
                sentiment_label=sentiment_label,
                processing_errors=review.processing_errors.copy(),
            )

            self.logger.debug(
                f"Review {review.review_id}: score={enhanced_score:.4f}, "
                f"label={sentiment_label}"
            )

            return analyzed_review

        except Exception as e:
            self.logger.error(f"Error analyzing review {review.review_id}: {e}")
            review.add_error(f"Sentiment analysis failed: {e}")

            # Return review with neutral sentiment as fallback
            review.sentiment_score = 0.0
            review.sentiment_label = "Neutral"
            return review

    def _handle_empty_review(self, review: Review) -> Review:
        """
        Handle reviews with empty text.

        Args:
            review: Review with empty text

        Returns:
            Review with neutral sentiment
        """
        self.logger.debug(
            f"Assigning neutral sentiment to empty review {review.review_id}"
        )

        review.sentiment_score = 0.0
        review.sentiment_label = "Neutral"

        if not any("empty" in error.lower() for error in review.processing_errors):
            review.add_error("Empty text assigned neutral sentiment")

        return review

    def _enhance_with_french_sentiment(self, text: str, base_score: float) -> float:
        """
        Enhance VADER score with French sentiment words.

        Args:
            text: Review text to analyze
            base_score: Base VADER compound score

        Returns:
            Enhanced sentiment score
        """
        if not text:
            return base_score

        text_lower = text.lower()
        french_adjustments = []

        # Look for French sentiment words
        for word, sentiment_value in self.french_sentiment_words.items():
            if word in text_lower:
                # Weight by word frequency and length
                word_count = text_lower.count(word)
                adjustment = sentiment_value * word_count * 0.1  # Scale factor
                french_adjustments.append(adjustment)

                self.logger.debug(
                    f"Found French word '{word}' {word_count} times, "
                    f"adjustment: {adjustment:.3f}"
                )

        # Calculate total French adjustment
        total_french_adjustment = sum(french_adjustments)

        # Combine with base score (weighted average)
        if total_french_adjustment != 0:
            # Give more weight to French words for French text
            french_weight = min(0.3, len(french_adjustments) * 0.1)
            enhanced_score = (
                base_score * (1 - french_weight)
                + total_french_adjustment * french_weight
            )

            # Ensure score stays within [-1, 1] bounds
            enhanced_score = max(-1.0, min(1.0, enhanced_score))

            self.logger.debug(
                f"Enhanced score: {base_score:.4f} -> {enhanced_score:.4f} "
                f"(French adjustment: {total_french_adjustment:.4f})"
            )

            return enhanced_score

        return base_score

    def _classify_sentiment(self, score: float) -> str:
        """
        Classify sentiment based on score and thresholds.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            Sentiment label (Positive, Negative, or Neutral)
        """
        if score > self.positive_threshold:
            return "Positive"
        elif score < self.negative_threshold:
            return "Negative"
        else:
            return "Neutral"

    def _log_analysis_summary(self, reviews: List[Review]) -> None:
        """
        Log summary statistics of sentiment analysis.

        Args:
            reviews: List of analyzed reviews
        """
        if not reviews:
            return

        positive_count = sum(1 for r in reviews if r.sentiment_label == "Positive")
        negative_count = sum(1 for r in reviews if r.sentiment_label == "Negative")
        neutral_count = sum(1 for r in reviews if r.sentiment_label == "Neutral")

        total = len(reviews)

        self.logger.info(
            f"Sentiment analysis completed: {total} reviews processed\n"
            f"  Positive: {positive_count} ({positive_count/total*100:.1f}%)\n"
            f"  Negative: {negative_count} ({negative_count/total*100:.1f}%)\n"
            f"  Neutral: {neutral_count} ({neutral_count/total*100:.1f}%)"
        )

        # Log score distribution
        scores = [r.sentiment_score for r in reviews if r.sentiment_score is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            self.logger.info(
                f"Score distribution: avg={avg_score:.3f}, "
                f"min={min_score:.3f}, max={max_score:.3f}"
            )

    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        Get information about the analyzer configuration.

        Returns:
            Dictionary with analyzer information
        """
        return {
            "analyzer_type": "VADER",
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "batch_size": self.batch_size,
            "french_words_count": len(self.french_sentiment_words),
            "threshold_justification": (
                f"Positive threshold ({self.positive_threshold}) and negative threshold "
                f"({self.negative_threshold}) chosen to create balanced classification "
                f"with neutral zone for ambiguous sentiment. VADER compound scores "
                f"range from -1 (most negative) to +1 (most positive)."
            ),
        }

    def update_thresholds(
        self, positive_threshold: float, negative_threshold: float
    ) -> None:
        """
        Update sentiment classification thresholds.

        Args:
            positive_threshold: New positive threshold
            negative_threshold: New negative threshold
        """
        old_pos, old_neg = self.positive_threshold, self.negative_threshold

        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        try:
            self._validate_thresholds()
            self.logger.info(
                f"Updated thresholds: positive {old_pos} -> {positive_threshold}, "
                f"negative {old_neg} -> {negative_threshold}"
            )
        except ValueError as e:
            # Revert to old values
            self.positive_threshold = old_pos
            self.negative_threshold = old_neg
            raise e
