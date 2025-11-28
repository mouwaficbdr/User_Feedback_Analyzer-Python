"""Text preprocessing for sentiment analysis."""

import re
import logging
import unicodedata
from typing import List, Dict, Optional
from ..models.review import Review
from ..utils.logger import handle_errors


class ReviewPreprocessor:
    """
    Preprocesses review text for sentiment analysis.

    Handles various text normalization tasks including:
    - Encoding standardization
    - Emoji and special character processing
    - Text cleaning and normalization
    - Empty text handling
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(__name__)

        # Emoji mappings for sentiment-relevant emojis
        self.emoji_mappings = {
            # Positive emojis
            "😀": " happy ",
            "😃": " happy ",
            "😄": " happy ",
            "😁": " happy ",
            "😊": " happy ",
            "😍": " love ",
            "🥰": " love ",
            "😘": " love ",
            "🤗": " happy ",
            "😎": " cool ",
            "👍": " good ",
            "👌": " good ",
            "💯": " perfect ",
            "⭐": " star ",
            "🌟": " star ",
            "✨": " good ",
            "❤️": " love ",
            "💖": " love ",
            "💕": " love ",
            "🎉": " celebrate ",
            "🔥": " amazing ",
            # Negative emojis
            "😞": " sad ",
            "😢": " sad ",
            "😭": " crying ",
            "😠": " angry ",
            "😡": " angry ",
            "🤬": " angry ",
            "😤": " frustrated ",
            "🙄": " annoyed ",
            "😒": " annoyed ",
            "😑": " disappointed ",
            "😐": " neutral ",
            "👎": " bad ",
            "💔": " heartbroken ",
            "😰": " worried ",
            "😨": " scared ",
            "🤮": " disgusted ",
            "🤢": " sick ",
            # Neutral emojis
            "🤔": " thinking ",
            "😐": " neutral ",
            "😶": " neutral ",
            "🤷": " unsure ",
            "¯\\_(ツ)_/¯": " unsure ",
        }

        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.phone_pattern = re.compile(
            r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        )
        self.number_pattern = re.compile(r"\b\d+([.,]\d+)?\b")
        self.whitespace_pattern = re.compile(r"\s+")
        self.punctuation_pattern = re.compile(r"[.!?]+")

        # French-specific patterns
        self.french_contractions = {
            "j'": "je ",
            "l'": "le ",
            "d'": "de ",
            "n'": "ne ",
            "m'": "me ",
            "t'": "te ",
            "s'": "se ",
            "c'": "ce ",
            "qu'": "que ",
        }

    def preprocess_reviews(self, reviews: List[Review]) -> List[Review]:
        """
        Preprocess a list of reviews.

        Args:
            reviews: List of Review objects to preprocess

        Returns:
            List of preprocessed Review objects
        """
        self.logger.info(f"Preprocessing {len(reviews)} reviews")

        preprocessed_reviews = []
        for review in reviews:
            try:
                preprocessed_review = self.preprocess_review(review)
                preprocessed_reviews.append(preprocessed_review)
            except Exception as e:
                self.logger.error(f"Error preprocessing review {review.review_id}: {e}")
                review.add_error(f"Preprocessing failed: {e}")
                preprocessed_reviews.append(review)

        self.logger.info("Preprocessing completed")
        return preprocessed_reviews

    def preprocess_review(self, review: Review) -> Review:
        """
        Preprocess a single review.

        Args:
            review: Review object to preprocess

        Returns:
            Preprocessed Review object
        """
        # Handle empty text
        if review.is_empty_text():
            review = self.handle_empty_text(review)
            return review

        # Preprocess the text
        processed_text = self.preprocess_text(review.review_text)

        # Create new review with processed text
        processed_review = Review(
            review_id=review.review_id,
            review_text=processed_text,
            sentiment_score=review.sentiment_score,
            sentiment_label=review.sentiment_label,
            processing_errors=review.processing_errors.copy(),
        )

        return processed_review

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Normalize encoding
        processed_text = self.normalize_encoding(text)

        # Step 2: Handle emojis
        processed_text = self.process_emojis(processed_text)

        # Step 3: Handle French contractions
        processed_text = self.expand_french_contractions(processed_text)

        # Step 4: Clean URLs, emails, phone numbers
        processed_text = self.clean_structured_data(processed_text)

        # Step 5: Normalize punctuation
        processed_text = self.normalize_punctuation(processed_text)

        # Step 6: Normalize whitespace
        processed_text = self.normalize_whitespace(processed_text)

        # Step 7: Final cleanup
        processed_text = processed_text.strip()

        return processed_text

    def normalize_encoding(self, text: str) -> str:
        """
        Normalize text encoding and handle special characters.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        try:
            # Normalize Unicode characters
            text = unicodedata.normalize("NFKC", text)

            # Handle common encoding issues
            replacements = {
                "â€™": "'",  # Smart apostrophe
                "â€œ": '"',  # Smart quote left
                "â€": '"',  # Smart quote right
                'â€"': "-",  # Em dash
                "â€¦": "...",  # Ellipsis
                "Ã©": "é",  # Common French character encoding issue
                "Ã¨": "è",
                "Ã ": "à",
                "Ã§": "ç",
                "Ã´": "ô",
                "Ã»": "û",
                "Ã®": "î",
                "Ã¢": "â",
            }

            for old, new in replacements.items():
                text = text.replace(old, new)

            return text

        except Exception as e:
            self.logger.warning(f"Encoding normalization failed: {e}")
            return text

    def process_emojis(self, text: str) -> str:
        """
        Process emojis by converting them to sentiment-relevant text.

        Args:
            text: Text containing emojis

        Returns:
            Text with emojis converted to words
        """
        processed_text = text

        # Replace known emojis with sentiment words
        for emoji, replacement in self.emoji_mappings.items():
            processed_text = processed_text.replace(emoji, replacement)

        # Remove remaining emojis that we don't have mappings for
        # This regex matches most emoji characters
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        processed_text = emoji_pattern.sub(" ", processed_text)

        return processed_text

    def expand_french_contractions(self, text: str) -> str:
        """
        Expand French contractions for better sentiment analysis.

        Args:
            text: Text with French contractions

        Returns:
            Text with expanded contractions
        """
        processed_text = text.lower()

        for contraction, expansion in self.french_contractions.items():
            processed_text = processed_text.replace(contraction, expansion)

        return processed_text

    def clean_structured_data(self, text: str) -> str:
        """
        Clean URLs, emails, phone numbers, and other structured data.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = self.url_pattern.sub(" ", text)

        # Remove email addresses
        text = self.email_pattern.sub(" ", text)

        # Remove phone numbers
        text = self.phone_pattern.sub(" ", text)

        # Handle product codes and reference numbers
        # Pattern for codes like "XYZ-789", "Code: ABC123"
        code_pattern = re.compile(
            r"\b[A-Z]{2,}-\d+\b|\bcode\s*[:\s]\s*[A-Z0-9-]+\b", re.IGNORECASE
        )
        text = code_pattern.sub(" ", text)

        # Handle prices (keep the sentiment context)
        price_pattern = re.compile(
            r"\d+[.,]\d+\s*[€$£]|\b\d+\s*euros?\b", re.IGNORECASE
        )
        text = price_pattern.sub(" prix ", text)

        return text

    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation for consistent processing.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized punctuation
        """
        # Normalize multiple punctuation marks
        # text = self.punctuation_pattern.sub(lambda m: m.group()[-1], text)  # Disabled to preserve intensity


        # Add spaces around punctuation for better tokenization
        text = re.sub(r"([.!?])", r" \1 ", text)
        text = re.sub(r"([,;:])", r" \1 ", text)

        # Handle parentheses and brackets
        text = re.sub(r"([()\\[\\]])", r" \1 ", text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(" ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def handle_empty_text(self, review: Review) -> Review:
        """
        Handle reviews with empty text.

        Args:
            review: Review with empty text

        Returns:
            Review with appropriate handling
        """
        self.logger.debug(f"Handling empty text for review {review.review_id}")

        # Set empty text to a neutral placeholder
        review.review_text = ""
        review.add_error("Review text is empty")

        return review

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get basic statistics about processed text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {"character_count": 0, "word_count": 0, "sentence_count": 0}

        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
        }
