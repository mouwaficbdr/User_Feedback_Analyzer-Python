"""Report generation functionality for sentiment analysis results."""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models.review import Review, SentimentResult
from ..utils.logger import handle_errors, check_disk_space, validate_file_path


class ReportGenerator:
    """
    Generates summary and detailed reports from sentiment analysis results.

    Supports multiple output formats and handles file I/O with proper error handling.
    """

    def __init__(
        self,
        summary_format: str = "json",
        results_format: str = "csv",
        summary_filename: str = "summary",
        results_filename: str = "results",
    ):
        """
        Initialize report generator.

        Args:
            summary_format: Format for summary report ("json" or "txt")
            results_format: Format for detailed results ("csv")
            summary_filename: Base filename for summary report (without extension)
            results_filename: Base filename for results report (without extension)
        """
        self.summary_format = summary_format.lower()
        self.results_format = results_format.lower()
        self.summary_filename = summary_filename
        self.results_filename = results_filename
        self.logger = logging.getLogger(__name__)

        # Validate formats
        self._validate_formats()

    def _validate_formats(self) -> None:
        """Validate output formats."""
        valid_summary_formats = ["json", "txt"]
        valid_results_formats = ["csv"]

        if self.summary_format not in valid_summary_formats:
            raise ValueError(f"Summary format must be one of {valid_summary_formats}")

        if self.results_format not in valid_results_formats:
            raise ValueError(f"Results format must be one of {valid_results_formats}")

    def generate_reports(
        self,
        reviews: List[Review],
        output_dir: str = ".",
        analyzer_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate both summary and detailed reports.

        Args:
            reviews: List of analyzed Review objects
            output_dir: Directory to save reports
            analyzer_info: Information about the analyzer used

        Returns:
            Dictionary with paths to generated files
        """
        self.logger.info(f"Generating reports for {len(reviews)} reviews")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Check disk space
        check_disk_space(str(output_path), required_mb=10.0)

        # Calculate sentiment statistics
        sentiment_result = SentimentResult.from_reviews(reviews)

        # Generate reports
        generated_files = {}

        # Generate summary report
        summary_path = self.generate_summary_report(
            sentiment_result, output_dir, analyzer_info
        )
        generated_files["summary"] = summary_path

        # Generate detailed results
        results_path = self.generate_detailed_report(reviews, output_dir)
        generated_files["results"] = results_path

        self.logger.info(
            f"Reports generated successfully: {list(generated_files.values())}"
        )
        return generated_files

    @handle_errors(reraise=True)
    def generate_summary_report(
        self,
        sentiment_result: SentimentResult,
        output_dir: str = ".",
        analyzer_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate summary report with sentiment statistics.

        Args:
            sentiment_result: Aggregated sentiment results
            output_dir: Directory to save the report
            analyzer_info: Information about the analyzer used

        Returns:
            Path to the generated summary file
        """
        if self.summary_format == "json":
            return self._generate_json_summary(
                sentiment_result, output_dir, analyzer_info
            )
        else:  # txt format
            return self._generate_txt_summary(
                sentiment_result, output_dir, analyzer_info
            )

    def _generate_json_summary(
        self,
        sentiment_result: SentimentResult,
        output_dir: str,
        analyzer_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate JSON format summary report."""
        output_path = Path(output_dir) / f"{self.summary_filename}.json"

        # Prepare summary data
        summary_data = {
            "analysis_summary": {
                "total_reviews": sentiment_result.total_reviews,
                "sentiment_distribution": {
                    "positive": {
                        "count": sentiment_result.positive_count,
                        "percentage": sentiment_result.positive_percentage,
                    },
                    "negative": {
                        "count": sentiment_result.negative_count,
                        "percentage": sentiment_result.negative_percentage,
                    },
                    "neutral": {
                        "count": sentiment_result.neutral_count,
                        "percentage": sentiment_result.neutral_percentage,
                    },
                },
                "processing_info": {
                    "timestamp": datetime.now().isoformat(),
                    "errors_count": len(sentiment_result.processing_errors),
                    "errors": (
                        sentiment_result.processing_errors
                        if sentiment_result.processing_errors
                        else []
                    ),
                },
            }
        }

        # Add analyzer information if provided
        if analyzer_info:
            summary_data["analysis_summary"]["configuration"] = {
                "analyzer_type": analyzer_info.get("analyzer_type", "Unknown"),
                "positive_threshold": analyzer_info.get("positive_threshold"),
                "negative_threshold": analyzer_info.get("negative_threshold"),
                "threshold_justification": analyzer_info.get(
                    "threshold_justification", ""
                ),
            }

        # Write JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON summary report generated: {output_path}")
        return str(output_path)

    def _generate_txt_summary(
        self,
        sentiment_result: SentimentResult,
        output_dir: str,
        analyzer_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text format summary report."""
        output_path = Path(output_dir) / f"{self.summary_filename}.txt"

        # Prepare summary text
        lines = [
            "SENTIMENT ANALYSIS SUMMARY REPORT",
            "=" * 40,
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Reviews Analyzed: {sentiment_result.total_reviews}",
            "",
            "SENTIMENT DISTRIBUTION:",
            "-" * 25,
            f"Positive: {sentiment_result.positive_count} reviews ({sentiment_result.positive_percentage}%)",
            f"Negative: {sentiment_result.negative_count} reviews ({sentiment_result.negative_percentage}%)",
            f"Neutral:  {sentiment_result.neutral_count} reviews ({sentiment_result.neutral_percentage}%)",
            "",
        ]

        # Add analyzer configuration if provided
        if analyzer_info:
            lines.extend(
                [
                    "ANALYZER CONFIGURATION:",
                    "-" * 25,
                    f"Analyzer Type: {analyzer_info.get('analyzer_type', 'Unknown')}",
                    f"Positive Threshold: {analyzer_info.get('positive_threshold', 'N/A')}",
                    f"Negative Threshold: {analyzer_info.get('negative_threshold', 'N/A')}",
                    "",
                ]
            )

            if analyzer_info.get("threshold_justification"):
                lines.extend(
                    [
                        "THRESHOLD JUSTIFICATION:",
                        "-" * 25,
                        analyzer_info["threshold_justification"],
                        "",
                    ]
                )

        # Add processing errors if any
        if sentiment_result.processing_errors:
            lines.extend(
                [
                    f"PROCESSING ERRORS ({len(sentiment_result.processing_errors)}):",
                    "-" * 25,
                ]
            )
            for i, error in enumerate(sentiment_result.processing_errors, 1):
                lines.append(f"{i}. {error}")
            lines.append("")
        else:
            lines.extend(
                [
                    "PROCESSING STATUS:",
                    "-" * 25,
                    "No errors encountered during processing.",
                    "",
                ]
            )

        # Write text file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info(f"Text summary report generated: {output_path}")
        return str(output_path)

    @handle_errors(reraise=True)
    def generate_detailed_report(
        self, reviews: List[Review], output_dir: str = "."
    ) -> str:
        """
        Generate detailed CSV report with all review data and sentiment results.

        Args:
            reviews: List of analyzed Review objects
            output_dir: Directory to save the report

        Returns:
            Path to the generated CSV file
        """
        output_path = Path(output_dir) / f"{self.results_filename}.csv"

        self.logger.info(f"Generating detailed CSV report: {output_path}")

        # Define CSV headers
        headers = [
            "review_id",
            "review_text",
            "sentiment_final",
            "sentiment_score",
            "processing_errors",
        ]

        # Write CSV file
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

            # Write header
            writer.writerow(headers)

            # Write review data
            for review in reviews:
                # Prepare error string
                error_str = (
                    "; ".join(review.processing_errors)
                    if review.processing_errors
                    else ""
                )

                # Clean review text for CSV (remove newlines, limit length)
                clean_text = self._clean_text_for_csv(review.review_text)

                row = [
                    review.review_id,
                    clean_text,
                    review.sentiment_label or "Unknown",
                    (
                        review.sentiment_score
                        if review.sentiment_score is not None
                        else ""
                    ),
                    error_str,
                ]

                writer.writerow(row)

        self.logger.info(f"Detailed CSV report generated with {len(reviews)} reviews")
        return str(output_path)

    def _clean_text_for_csv(self, text: str, max_length: int = 500) -> str:
        """
        Clean text for CSV output.

        Args:
            text: Text to clean
            max_length: Maximum length of text to include

        Returns:
            Cleaned text suitable for CSV
        """
        if not text:
            return ""

        # Remove newlines and excessive whitespace
        cleaned = " ".join(text.split())

        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[: max_length - 3] + "..."

        return cleaned

    def calculate_statistics(self, reviews: List[Review]) -> SentimentResult:
        """
        Calculate sentiment statistics from reviews.

        Args:
            reviews: List of analyzed Review objects

        Returns:
            SentimentResult with calculated statistics
        """
        return SentimentResult.from_reviews(reviews)

    def validate_reviews(self, reviews: List[Review]) -> List[str]:
        """
        Validate reviews before report generation.

        Args:
            reviews: List of Review objects to validate

        Returns:
            List of validation errors
        """
        errors = []

        if not reviews:
            errors.append("No reviews provided for report generation")
            return errors

        # Check for required fields
        for i, review in enumerate(reviews):
            if not review.review_id:
                errors.append(f"Review at index {i} missing review_id")

            if review.sentiment_label not in ["Positive", "Negative", "Neutral", None]:
                errors.append(
                    f"Review {review.review_id} has invalid sentiment_label: {review.sentiment_label}"
                )

            if review.sentiment_score is not None:
                if not -1.0 <= review.sentiment_score <= 1.0:
                    errors.append(
                        f"Review {review.review_id} has invalid sentiment_score: {review.sentiment_score}"
                    )

        return errors

    def get_report_info(self) -> Dict[str, Any]:
        """
        Get information about report generator configuration.

        Returns:
            Dictionary with configuration information
        """
        return {
            "summary_format": self.summary_format,
            "results_format": self.results_format,
            "summary_filename": self.summary_filename,
            "results_filename": self.results_filename,
            "supported_summary_formats": ["json", "txt"],
            "supported_results_formats": ["csv"],
        }
