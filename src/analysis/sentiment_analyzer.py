"""Sentiment analysis functionality."""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from ..models.review import Review
from ..utils.logger import handle_errors, log_execution_time
from .french_lexicon import FRENCH_LEXICON, NEGATIONS, BOOSTERS, IDIOMS


class SentimentAnalyzerInterface(ABC):
    """Abstract interface for sentiment analyzers."""

    @abstractmethod
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """Analyze sentiment for a list of reviews."""
        pass


class VaderSentimentAnalyzer(SentimentAnalyzerInterface):
    """
    Enhanced Rule-Based Sentiment Analyzer for French.
    
    Adapted from VADER concepts but built from scratch for French linguistics.
    Features:
    - Comprehensive French sentiment lexicon
    - Negation handling (ne...pas, jamais, etc.)
    - Intensifier handling (très, trop, peu, etc.)
    - Idiom detection
    - "But" clause handling
    - Punctuation and capitalization analysis
    """

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        batch_size: int = 100,  # Kept for compatibility but unused in simple loop
    ):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Smart French Sentiment Analyzer initialized")

    @log_execution_time()
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """Analyze sentiment for a list of reviews."""
        self.logger.info(f"Starting analysis for {len(reviews)} reviews")
        
        analyzed_reviews = []
        for review in reviews:
            analyzed_reviews.append(self.analyze_single_review(review))
            
        self._log_analysis_summary(analyzed_reviews)
        return analyzed_reviews

    @handle_errors(reraise=False)
    def analyze_single_review(self, review: Review) -> Review:
        """Analyze sentiment for a single review."""
        try:
            if review.is_empty_text():
                review.sentiment_score = 0.0
                review.sentiment_label = "Neutral"
                return review

            # 1. Calculate raw sentiment score
            score = self._calculate_sentiment_score(review.review_text)
            
            # 2. Normalize score to [-1, 1]
            normalized_score = self._normalize_score(score)
            
            # 3. Classify
            label = self._classify_sentiment(normalized_score)

            # Update review object
            review.sentiment_score = round(normalized_score, 4)
            review.sentiment_label = label
            
            return review

        except Exception as e:
            self.logger.error(f"Error analyzing review {review.review_id}: {e}")
            review.add_error(f"Analysis failed: {e}")
            review.sentiment_score = 0.0
            review.sentiment_label = "Neutral"
            return review

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score based on lexical rules."""
        if not text:
            return 0.0

        # Pre-processing
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        
        current_score = 0.0
        
        # 1. Check for Idioms first (they override individual words)
        # We remove found idioms from text to avoid double counting
        temp_text = text_lower
        for idiom, val in IDIOMS.items():
            if idiom in temp_text:
                count = temp_text.count(idiom)
                current_score += val * count
                # Remove idiom to avoid processing its words individually
                temp_text = temp_text.replace(idiom, "")
        
        # Re-tokenize after idiom removal
        words = re.findall(r"\b\w+\b", temp_text)
        
        # 2. Process individual words with context (negation/boosters)
        current_weight = 1.0
        
        for i, word in enumerate(words):
            # Handle "mais" (but) logic: increase weight for subsequent words
            if word == "mais":
                current_weight = 1.5
                continue
                
            if word in FRENCH_LEXICON:
                valence = FRENCH_LEXICON[word]
                
                # Check for negation (look back 2 words)
                # "ce n'est pas bon" -> "pas" is at i-1, "n'" at i-2
                is_negated = False
                if i > 0 and words[i-1] in NEGATIONS:
                    is_negated = True
                elif i > 1 and words[i-2] in NEGATIONS:
                    is_negated = True
                
                # Check for boosters (look back 1 word)
                booster_factor = 1.0
                if i > 0 and words[i-1] in BOOSTERS:
                    booster_factor = BOOSTERS[words[i-1]]
                
                # Apply logic
                if is_negated:
                    # Invert and reduce intensity (negation isn't always perfect opposite)
                    # "pas terrible" (-3) -> +1.5 (not terrible is okay-ish)
                    # "pas bon" (+1.5) -> -1.0 (not good is bad)
                    valence = valence * -0.74
                
                valence = valence * booster_factor * current_weight
                
                # Caps lock check (in original text)
                # Find the word in original text roughly
                if word.upper() in text:
                    valence *= 1.25
                
                current_score += valence

        # 3. "BUT" handling (The "Mais" rule) - Already handled in loop

        # 4. Punctuation handling
        # Exclamation marks boost intensity
        exclamations = text.count("!")
        if exclamations > 0:
            boost = min(exclamations, 4) * 0.2
            if current_score > 0:
                current_score += boost
            elif current_score < 0:
                current_score -= boost

        return current_score

    def _normalize_score(self, score: float) -> float:
        """
        Normalize score to [-1, 1] using hyperbolic tangent-like function.
        VADER normalization formula: x / sqrt(x^2 + alpha)
        """
        alpha = 15  # Normalization constant
        norm_score = score / ((score * score + alpha) ** 0.5)
        return max(-1.0, min(1.0, norm_score))

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score and thresholds."""
        if score > self.positive_threshold:
            return "Positive"
        elif score < self.negative_threshold:
            return "Negative"
        else:
            return "Neutral"

    def _log_analysis_summary(self, reviews: List[Review]) -> None:
        """Log summary statistics."""
        if not reviews:
            return
            
        pos = sum(1 for r in reviews if r.sentiment_label == "Positive")
        neg = sum(1 for r in reviews if r.sentiment_label == "Negative")
        neu = sum(1 for r in reviews if r.sentiment_label == "Neutral")
        total = len(reviews)
        
        self.logger.info(
            f"Analysis complete: {total} reviews. "
            f"Pos: {pos} ({pos/total:.1%}), "
            f"Neg: {neg} ({neg/total:.1%}), "
            f"Neu: {neu} ({neu/total:.1%})"
        )

    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get info for report."""
        return {
            "analyzer_type": "Smart French Rule-Based",
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "lexicon_size": len(FRENCH_LEXICON),
            "features": ["Negation", "Intensifiers", "Idioms", "Punctuation"]
        }
    
    def update_thresholds(self, positive: float, negative: float) -> None:
        self.positive_threshold = positive
        self.negative_threshold = negative
