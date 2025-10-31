"""Core data models for sentiment analysis."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Review:
    """
    Represents a customer review with sentiment analysis results.

    Attributes:
        review_id: Unique identifier for the review
        review_text: The actual review content
        sentiment_score: Numerical sentiment score (-1 to 1)
        sentiment_label: Classification label (Positive, Negative, Neutral)
        processing_errors: List of errors encountered during processing
    """

    review_id: str
    review_text: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate review data after initialization."""
        if not self.review_id:
            raise ValueError("review_id cannot be empty")

        # Handle None review_text
        if self.review_text is None:
            self.review_text = ""

    def add_error(self, error_message: str) -> None:
        """Add a processing error to the review."""
        self.processing_errors.append(error_message)

    def has_errors(self) -> bool:
        """Check if the review has any processing errors."""
        return len(self.processing_errors) > 0

    def is_empty_text(self) -> bool:
        """Check if the review text is empty or whitespace only."""
        return not self.review_text or not self.review_text.strip()


@dataclass
class SentimentResult:
    """
    Aggregated results from sentiment analysis.

    Attributes:
        total_reviews: Total number of reviews processed
        positive_count: Number of positive reviews
        negative_count: Number of negative reviews
        neutral_count: Number of neutral reviews
        positive_percentage: Percentage of positive reviews
        negative_percentage: Percentage of negative reviews
        neutral_percentage: Percentage of neutral reviews
        processing_errors: List of processing errors encountered
    """

    total_reviews: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate sentiment result data."""
        if self.total_reviews < 0:
            raise ValueError("total_reviews cannot be negative")

        # Validate counts sum to total
        count_sum = self.positive_count + self.negative_count + self.neutral_count
        if count_sum != self.total_reviews:
            raise ValueError(
                f"Sentiment counts ({count_sum}) don't match total reviews ({self.total_reviews})"
            )

        # Validate percentages sum to approximately 100 (except for empty case)
        percentage_sum = (
            self.positive_percentage
            + self.negative_percentage
            + self.neutral_percentage
        )
        if (
            self.total_reviews > 0 and abs(percentage_sum - 100.0) > 0.1
        ):  # Allow small rounding errors
            raise ValueError(f"Percentages don't sum to 100: {percentage_sum}")

    @classmethod
    def from_reviews(cls, reviews: List[Review]) -> "SentimentResult":
        """
        Create SentimentResult from a list of analyzed reviews.

        Args:
            reviews: List of Review objects with sentiment labels

        Returns:
            SentimentResult object with calculated statistics
        """
        total = len(reviews)
        if total == 0:
            return cls(0, 0, 0, 0, 0.0, 0.0, 0.0)

        positive = sum(1 for r in reviews if r.sentiment_label == "Positive")
        negative = sum(1 for r in reviews if r.sentiment_label == "Negative")
        neutral = sum(1 for r in reviews if r.sentiment_label == "Neutral")

        # Calculate percentages with proper rounding
        pos_pct = round((positive / total) * 100, 1)
        neg_pct = round((negative / total) * 100, 1)
        neu_pct = round((neutral / total) * 100, 1)

        # Adjust for rounding errors to ensure sum is exactly 100
        total_pct = pos_pct + neg_pct + neu_pct
        if total_pct != 100.0:
            # Adjust the largest percentage
            max_pct = max(pos_pct, neg_pct, neu_pct)
            if max_pct == pos_pct:
                pos_pct += 100.0 - total_pct
            elif max_pct == neg_pct:
                neg_pct += 100.0 - total_pct
            else:
                neu_pct += 100.0 - total_pct

        # Collect all processing errors
        all_errors = []
        for review in reviews:
            all_errors.extend(review.processing_errors)

        return cls(
            total_reviews=total,
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            positive_percentage=pos_pct,
            negative_percentage=neg_pct,
            neutral_percentage=neu_pct,
            processing_errors=all_errors,
        )
