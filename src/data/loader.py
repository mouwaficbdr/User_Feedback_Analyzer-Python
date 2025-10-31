"""Data loading functionality for the sentiment analysis engine."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import chardet

from ..models.review import Review
from ..utils.logger import handle_errors, ErrorAggregator


class DataLoaderInterface(ABC):
    """Abstract interface for data loaders."""

    @abstractmethod
    def load_reviews(self, file_path: str) -> List[Review]:
        """
        Load reviews from a data source.

        Args:
            file_path: Path to the data source

        Returns:
            List of Review objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data format is invalid
        """
        pass


class JSONDataLoader(DataLoaderInterface):
    """
    Loads review data from JSON files with robust error handling.

    Handles various edge cases including:
    - Encoding issues (UTF-8, Latin-1, CP1252)
    - Malformed JSON entries
    - Missing required fields
    - Empty or null values
    """

    def __init__(self, encoding_fallbacks: Optional[List[str]] = None):
        """
        Initialize JSON data loader.

        Args:
            encoding_fallbacks: List of encodings to try in order
        """
        self.encoding_fallbacks = encoding_fallbacks or ["utf-8", "latin-1", "cp1252"]
        self.logger = logging.getLogger(__name__)
        self.error_aggregator = ErrorAggregator(self.logger)

    def load_reviews(self, file_path: str) -> List[Review]:
        """
        Load reviews from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of Review objects

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be processed
        """
        self.logger.info(f"Loading reviews from {file_path}")

        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Review file not found: {file_path}")

        # Load and parse JSON data
        raw_data = self._load_json_file(file_path)

        # Convert to Review objects
        reviews = self._parse_reviews(raw_data)

        self.logger.info(f"Successfully loaded {len(reviews)} reviews")
        if self.error_aggregator.has_errors():
            self.error_aggregator.log_summary()

        return reviews

    def _load_json_file(self, file_path: str) -> Any:
        """
        Load JSON data from file with encoding detection and fallbacks.

        Args:
            file_path: Path to the JSON file

        Returns:
            Parsed JSON data

        Raises:
            ValueError: If the file cannot be parsed as JSON
        """
        # First, try to detect encoding
        detected_encoding = self._detect_encoding(file_path)
        if detected_encoding:
            encodings_to_try = [detected_encoding] + self.encoding_fallbacks
        else:
            encodings_to_try = self.encoding_fallbacks

        # Remove duplicates while preserving order
        encodings_to_try = list(dict.fromkeys(encodings_to_try))

        last_exception = None

        for encoding in encodings_to_try:
            try:
                self.logger.debug(
                    f"Trying to load {file_path} with encoding: {encoding}"
                )

                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                # Handle potential BOM
                if content.startswith("\ufeff"):
                    content = content[1:]

                data = json.loads(content)
                self.logger.info(f"Successfully loaded JSON with encoding: {encoding}")
                return data

            except UnicodeDecodeError as e:
                last_exception = e
                self.logger.debug(f"Encoding {encoding} failed: {e}")
                continue

            except json.JSONDecodeError as e:
                # JSON parsing failed - this is more serious
                self.error_aggregator.add_error(
                    f"JSON parsing failed with encoding {encoding}: {e}",
                    f"File: {file_path}",
                )
                last_exception = e
                continue

        # If we get here, all encodings failed
        raise ValueError(
            f"Could not load JSON file {file_path}. "
            f"Tried encodings: {encodings_to_try}. "
            f"Last error: {last_exception}"
        )

    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """
        Detect file encoding using chardet.

        Args:
            file_path: Path to the file

        Returns:
            Detected encoding or None if detection fails
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB for detection

            result = chardet.detect(raw_data)
            if result and result["confidence"] > 0.7:
                encoding = result["encoding"]
                self.logger.debug(
                    f"Detected encoding: {encoding} (confidence: {result['confidence']:.2f})"
                )
                return encoding

        except Exception as e:
            self.logger.debug(f"Encoding detection failed: {e}")

        return None

    def _parse_reviews(self, raw_data: Any) -> List[Review]:
        """
        Parse raw JSON data into Review objects.

        Args:
            raw_data: Raw JSON data (should be a list of dictionaries)

        Returns:
            List of Review objects
        """
        reviews = []

        # Handle different JSON structures
        if isinstance(raw_data, dict):
            # Check if it's a wrapper object with reviews array
            if "reviews" in raw_data:
                review_data = raw_data["reviews"]
            elif len(raw_data) == 1 and isinstance(list(raw_data.values())[0], list):
                # Single key with array value
                review_data = list(raw_data.values())[0]
            else:
                # Treat the dict as a single review
                review_data = [raw_data]
        elif isinstance(raw_data, list):
            review_data = raw_data
        else:
            raise ValueError(
                f"Unexpected JSON structure. Expected list or dict, got {type(raw_data)}"
            )

        # Process each review
        for i, item in enumerate(review_data):
            try:
                review = self._parse_single_review(item, i)
                if review:
                    reviews.append(review)
            except Exception as e:
                self.error_aggregator.add_error(
                    f"Failed to parse review at index {i}: {e}", "Review parsing"
                )
                continue

        return reviews

    def _parse_single_review(self, item: Any, index: int) -> Optional[Review]:
        """
        Parse a single review item into a Review object.

        Args:
            item: Raw review data
            index: Index of the review in the source data

        Returns:
            Review object or None if parsing fails
        """
        if not isinstance(item, dict):
            self.error_aggregator.add_error(
                f"Review at index {index} is not a dictionary: {type(item)}",
                "Review parsing",
            )
            return None

        # Extract required fields
        review_id = self._extract_review_id(item, index)
        if not review_id:
            return None

        review_text = self._extract_review_text(item)

        # Create Review object
        try:
            review = Review(review_id=review_id, review_text=review_text)

            # Log if review text is empty
            if review.is_empty_text():
                self.error_aggregator.add_error(
                    f"Review {review_id} has empty text", "Data quality"
                )

            return review

        except Exception as e:
            self.error_aggregator.add_error(
                f"Failed to create Review object for {review_id}: {e}",
                "Review creation",
            )
            return None

    def _extract_review_id(self, item: Dict[str, Any], index: int) -> Optional[str]:
        """
        Extract review ID from review item.

        Args:
            item: Review dictionary
            index: Index of the review

        Returns:
            Review ID or None if not found
        """
        # Try common field names for review ID
        id_fields = ["review_id", "id", "reviewId", "review_ID"]

        for field in id_fields:
            if field in item:
                review_id = item[field]
                if review_id is not None:
                    return str(review_id).strip()

        # If no ID field found, generate one
        generated_id = f"REVIEW_{index + 1:03d}"
        self.error_aggregator.add_error(
            f"No review ID found at index {index}, generated: {generated_id}",
            "Data quality",
        )
        return generated_id

    def _extract_review_text(self, item: Dict[str, Any]) -> str:
        """
        Extract review text from review item.

        Args:
            item: Review dictionary

        Returns:
            Review text (empty string if not found)
        """
        # Try common field names for review text
        text_fields = [
            "review_text",
            "text",
            "content",
            "review",
            "comment",
            "reviewText",
        ]

        for field in text_fields:
            if field in item:
                text = item[field]
                if text is not None:
                    return str(text).strip()

        # Return empty string if no text found
        return ""

    def get_loading_errors(self) -> List[str]:
        """
        Get list of errors encountered during loading.

        Returns:
            List of error messages
        """
        return self.error_aggregator.get_errors()

    def has_loading_errors(self) -> bool:
        """
        Check if any errors occurred during loading.

        Returns:
            True if errors occurred, False otherwise
        """
        return self.error_aggregator.has_errors()
