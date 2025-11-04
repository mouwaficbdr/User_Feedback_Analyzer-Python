"""Machine Learning-based sentiment analyzer for French text."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC

from .sentiment_analyzer import SentimentAnalyzerInterface
from ..models.review import Review
from ..utils.logger import handle_errors, log_execution_time
from ..utils.performance import BatchProcessor


class MLSentimentAnalyzer(SentimentAnalyzerInterface):
    """
    Machine Learning-based sentiment analyzer using pre-trained models.
    
    Supports multiple French language models:
    - nlptown/bert-base-multilingual-uncased-sentiment (5-star rating)
    - cardiffnlp/twitter-xlm-roberta-base-sentiment (3-class sentiment)
    - camembert-base (French BERT)
    """

    def __init__(
        self,
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        batch_size: int = 32,
        use_gpu: bool = False,
    ):
        """
        Initialize ML sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            positive_threshold: Minimum score for positive classification
            negative_threshold: Maximum score for negative classification
            batch_size: Number of reviews to process in each batch
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.model_name = model_name
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        self.logger.info(
            f"ML Sentiment Analyzer initialized with model: {model_name}"
        )

    def _initialize_model(self) -> None:
        """Initialize the ML model and tokenizer."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Determine device
            device = -1  # CPU by default
            if self.use_gpu and torch.cuda.is_available():
                device = 0
                self.logger.info("Using GPU acceleration")
            else:
                self.logger.info("Using CPU")
            
            # Load model based on type
            if "nlptown" in self.model_name:
                # 5-star rating model
                self.model = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device,
                    top_k=None,
                )
                self.model_type = "5-star"
            elif "cardiffnlp" in self.model_name or "twitter" in self.model_name:
                # 3-class sentiment model
                self.model = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device,
                )
                self.model_type = "3-class"
            else:
                # Generic sentiment analysis
                self.model = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=device,
                )
                self.model_type = "generic"
            
            self.logger.info(f"Model loaded successfully (type: {self.model_type})")
            
        except ImportError as e:
            self.logger.error(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            raise RuntimeError(
                "ML Sentiment Analyzer requires transformers library. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Could not initialize ML model: {e}") from e

    @log_execution_time()
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """
        Analyze sentiment for a list of reviews using ML model.
        
        Args:
            reviews: List of Review objects to analyze
            
        Returns:
            List of Review objects with sentiment scores and labels
        """
        self.logger.info(f"Starting ML sentiment analysis for {len(reviews)} reviews")
        
        if len(reviews) <= self.batch_size:
            # Small dataset - process directly
            analyzed_reviews = [self.analyze_single_review(review) for review in reviews]
        else:
            # Large dataset - use batch processing
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
        Analyze sentiment for a single review using ML model.
        
        Args:
            review: Review object to analyze
            
        Returns:
            Review object with sentiment score and label
        """
        try:
            # Handle empty text
            if review.is_empty_text():
                return self._handle_empty_review(review)
            
            # Truncate very long texts (models have token limits)
            text = review.review_text[:512]  # Most models support up to 512 tokens
            
            # Get ML model prediction
            result = self.model(text)
            
            # Convert model output to sentiment score
            sentiment_score = self._convert_model_output(result)
            
            # Classify sentiment based on thresholds
            sentiment_label = self._classify_sentiment(sentiment_score)
            
            # Create analyzed review
            analyzed_review = Review(
                review_id=review.review_id,
                review_text=review.review_text,
                sentiment_score=round(sentiment_score, 4),
                sentiment_label=sentiment_label,
                processing_errors=review.processing_errors.copy(),
            )
            
            self.logger.debug(
                f"Review {review.review_id}: score={sentiment_score:.4f}, "
                f"label={sentiment_label}"
            )
            
            return analyzed_review
            
        except Exception as e:
            self.logger.error(f"Error analyzing review {review.review_id}: {e}")
            review.add_error(f"ML sentiment analysis failed: {e}")
            
            # Return review with neutral sentiment as fallback
            review.sentiment_score = 0.0
            review.sentiment_label = "Neutral"
            return review

    def _convert_model_output(self, result: Any) -> float:
        """
        Convert model output to normalized sentiment score [-1, 1].
        
        Args:
            result: Model output (varies by model type)
            
        Returns:
            Normalized sentiment score
        """
        if self.model_type == "5-star":
            # nlptown model returns 5-star ratings
            # Result is a list of dictionaries with labels and scores
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Top-k results
                    predictions = result[0]
                    # Calculate weighted average
                    weighted_score = 0.0
                    for pred in predictions:
                        stars = int(pred['label'].split()[0])  # Extract number from "1 star", "2 stars", etc.
                        score = pred['score']
                        # Convert 1-5 stars to -1 to 1 scale
                        normalized = (stars - 3) / 2.0  # 1->-1, 2->-0.5, 3->0, 4->0.5, 5->1
                        weighted_score += normalized * score
                    return weighted_score
                else:
                    # Single prediction
                    label = result[0]['label']
                    stars = int(label.split()[0])
                    # Convert 1-5 stars to -1 to 1 scale
                    return (stars - 3) / 2.0
            
        elif self.model_type == "3-class":
            # cardiffnlp or similar 3-class models
            if isinstance(result, list) and len(result) > 0:
                label = result[0]['label'].lower()
                score = result[0]['score']
                
                if 'positive' in label:
                    return score  # 0 to 1
                elif 'negative' in label:
                    return -score  # -1 to 0
                else:  # neutral
                    return 0.0
        
        # Generic fallback
        if isinstance(result, list) and len(result) > 0:
            label = result[0].get('label', '').lower()
            score = result[0].get('score', 0.5)
            
            if any(pos in label for pos in ['positive', 'pos', '4', '5']):
                return score
            elif any(neg in label for neg in ['negative', 'neg', '1', '2']):
                return -score
            else:
                return 0.0
        
        return 0.0

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
            f"ML Sentiment analysis completed: {total} reviews processed\n"
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
            "analyzer_type": "ML",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "threshold_justification": (
                f"ML-based analyzer using {self.model_name}. "
                f"Positive threshold ({self.positive_threshold}) and negative threshold "
                f"({self.negative_threshold}) applied to normalized model scores. "
                f"Model trained on multilingual data including French."
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
        
        if positive_threshold <= negative_threshold:
            # Revert to old values
            self.positive_threshold = old_pos
            self.negative_threshold = old_neg
            raise ValueError(
                f"Positive threshold ({positive_threshold}) must be greater than "
                f"negative threshold ({negative_threshold})"
            )
        
        self.logger.info(
            f"Updated thresholds: positive {old_pos} -> {positive_threshold}, "
            f"negative {old_neg} -> {negative_threshold}"
        )


class HybridSentimentAnalyzer(SentimentAnalyzerInterface):
    """
    Hybrid analyzer combining VADER and ML models.
    
    Uses weighted combination of both approaches for improved accuracy.
    """

    def __init__(
        self,
        vader_weight: float = 0.3,
        ml_weight: float = 0.7,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        batch_size: int = 32,
    ):
        """
        Initialize hybrid analyzer.
        
        Args:
            vader_weight: Weight for VADER score (0-1)
            ml_weight: Weight for ML score (0-1)
            positive_threshold: Minimum score for positive classification
            negative_threshold: Maximum score for negative classification
            batch_size: Number of reviews to process in each batch
        """
        from .sentiment_analyzer import VaderSentimentAnalyzer
        
        self.vader_weight = vader_weight
        self.ml_weight = ml_weight
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize both analyzers
        self.vader_analyzer = VaderSentimentAnalyzer(
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            batch_size=batch_size,
        )
        
        self.ml_analyzer = MLSentimentAnalyzer(
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            batch_size=batch_size,
        )
        
        # Normalize weights
        total_weight = vader_weight + ml_weight
        self.vader_weight = vader_weight / total_weight
        self.ml_weight = ml_weight / total_weight
        
        self.logger.info(
            f"Hybrid Analyzer initialized (VADER: {self.vader_weight:.2f}, "
            f"ML: {self.ml_weight:.2f})"
        )

    @log_execution_time()
    def analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """
        Analyze sentiment using hybrid approach.
        
        Args:
            reviews: List of Review objects to analyze
            
        Returns:
            List of Review objects with combined sentiment scores and labels
        """
        self.logger.info(
            f"Starting hybrid sentiment analysis for {len(reviews)} reviews"
        )
        
        # Get VADER predictions
        vader_reviews = self.vader_analyzer.analyze_sentiment(reviews)
        
        # Get ML predictions
        ml_reviews = self.ml_analyzer.analyze_sentiment(reviews)
        
        # Combine predictions
        combined_reviews = []
        for vader_rev, ml_rev in zip(vader_reviews, ml_reviews):
            # Weighted average of scores
            combined_score = (
                self.vader_weight * vader_rev.sentiment_score
                + self.ml_weight * ml_rev.sentiment_score
            )
            
            # Classify based on combined score
            sentiment_label = self._classify_sentiment(combined_score)
            
            # Create combined review
            combined_review = Review(
                review_id=vader_rev.review_id,
                review_text=vader_rev.review_text,
                sentiment_score=round(combined_score, 4),
                sentiment_label=sentiment_label,
                processing_errors=vader_rev.processing_errors.copy(),
            )
            
            combined_reviews.append(combined_review)
        
        self.logger.info("Hybrid sentiment analysis completed")
        return combined_reviews

    def analyze_single_review(self, review: Review) -> Review:
        """
        Analyze sentiment for a single review using hybrid approach.
        
        Args:
            review: Review object to analyze
            
        Returns:
            Review object with combined sentiment score and label
        """
        # Get VADER prediction
        vader_review = self.vader_analyzer.analyze_single_review(review)
        
        # Get ML prediction
        ml_review = self.ml_analyzer.analyze_single_review(review)
        
        # Combine scores
        combined_score = (
            self.vader_weight * vader_review.sentiment_score
            + self.ml_weight * ml_review.sentiment_score
        )
        
        # Classify
        sentiment_label = self._classify_sentiment(combined_score)
        
        # Create combined review
        combined_review = Review(
            review_id=review.review_id,
            review_text=review.review_text,
            sentiment_score=round(combined_score, 4),
            sentiment_label=sentiment_label,
            processing_errors=review.processing_errors.copy(),
        )
        
        return combined_review

    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score and thresholds."""
        if score > self.positive_threshold:
            return "Positive"
        elif score < self.negative_threshold:
            return "Negative"
        else:
            return "Neutral"

    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the hybrid analyzer configuration."""
        return {
            "analyzer_type": "Hybrid",
            "vader_weight": self.vader_weight,
            "ml_weight": self.ml_weight,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "vader_info": self.vader_analyzer.get_analyzer_info(),
            "ml_info": self.ml_analyzer.get_analyzer_info(),
        }

    def update_thresholds(
        self, positive_threshold: float, negative_threshold: float
    ) -> None:
        """Update thresholds for both analyzers."""
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.vader_analyzer.update_thresholds(positive_threshold, negative_threshold)
        self.ml_analyzer.update_thresholds(positive_threshold, negative_threshold)
