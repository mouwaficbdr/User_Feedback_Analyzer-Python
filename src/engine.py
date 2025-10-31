"""Main sentiment analysis engine orchestrator."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .models.review import Review
from .data.loader import JSONDataLoader
from .preprocessing.preprocessor import ReviewPreprocessor
from .analysis.sentiment_analyzer import VaderSentimentAnalyzer
from .reporting.report_generator import ReportGenerator
from .config.config_manager import ConfigManager
from .utils.logger import setup_logger, log_execution_time, ErrorAggregator
from .utils.performance import PerformanceMonitor, ProgressTracker


class SentimentAnalysisEngine:
    """
    Main orchestrator for the sentiment analysis pipeline.

    Coordinates data loading, preprocessing, analysis, and reporting
    with comprehensive error handling and logging.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the sentiment analysis engine.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = ConfigManager(config_path)

        # Set up logging
        self.logger = setup_logger(
            name="SentimentAnalysisEngine",
            log_file=self.config.get_log_file(),
            log_level=self.config.get_log_level(),
            log_format=self.config.get_log_format(),
        )

        # Initialize error aggregator and performance monitor
        self.error_aggregator = ErrorAggregator(self.logger)
        self.performance_monitor = PerformanceMonitor()

        # Initialize pipeline components
        self._initialize_components()

        self.logger.info("Sentiment Analysis Engine initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Data loader
            self.data_loader = JSONDataLoader(
                encoding_fallbacks=self.config.get_encoding_fallbacks()
            )

            # Preprocessor
            self.preprocessor = ReviewPreprocessor()

            # Sentiment analyzer
            self.sentiment_analyzer = VaderSentimentAnalyzer(
                positive_threshold=self.config.get_positive_threshold(),
                negative_threshold=self.config.get_negative_threshold(),
                batch_size=self.config.get_batch_size(),
            )

            # Report generator
            self.report_generator = ReportGenerator(
                summary_format=self.config.get_summary_format(),
                results_format=self.config.get_results_format(),
                summary_filename=self.config.get_summary_filename(),
                results_filename=self.config.get_results_filename(),
            )

            self.logger.info("All pipeline components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise RuntimeError(f"Engine initialization failed: {e}")

    @log_execution_time()
    def analyze_reviews(
        self,
        input_file: str,
        output_dir: str = ".",
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete sentiment analysis pipeline.

        Args:
            input_file: Path to input JSON file with reviews
            output_dir: Directory to save output files
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary with analysis results and file paths
        """
        self.logger.info(f"Starting sentiment analysis pipeline")
        self.logger.info(f"Input file: {input_file}")
        self.logger.info(f"Output directory: {output_dir}")

        try:
            # Step 1: Load reviews
            if progress_callback:
                progress_callback("Loading reviews...", 0)

            with self.performance_monitor.monitor_operation("load_reviews") as metrics:
                reviews = self._load_reviews(input_file)
                metrics.items_processed = len(reviews)
            self.logger.info(f"Loaded {len(reviews)} reviews")

            # Step 2: Preprocess reviews
            if progress_callback:
                progress_callback("Preprocessing text...", 25)

            with self.performance_monitor.monitor_operation("preprocess_reviews", len(reviews)) as metrics:
                preprocessed_reviews = self._preprocess_reviews(reviews)
            self.logger.info("Text preprocessing completed")

            # Step 3: Analyze sentiment
            if progress_callback:
                progress_callback("Analyzing sentiment...", 50)

            with self.performance_monitor.monitor_operation("analyze_sentiment", len(preprocessed_reviews)) as metrics:
                analyzed_reviews = self._analyze_sentiment(preprocessed_reviews)
            self.logger.info("Sentiment analysis completed")

            # Step 4: Generate reports
            if progress_callback:
                progress_callback("Generating reports...", 75)

            with self.performance_monitor.monitor_operation("generate_reports", len(analyzed_reviews)) as metrics:
                report_files = self._generate_reports(analyzed_reviews, output_dir)
            self.logger.info("Report generation completed")

            # Step 5: Compile results
            if progress_callback:
                progress_callback("Finalizing results...", 100)

            results = self._compile_results(analyzed_reviews, report_files)

            # Add performance metrics to results
            results["performance_metrics"] = self.performance_monitor.get_overall_stats()

            self.logger.info("Sentiment analysis pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.error_aggregator.add_exception(e, "Pipeline execution")
            raise

    def _load_reviews(self, input_file: str) -> List[Review]:
        """Load reviews from input file."""
        try:
            reviews = self.data_loader.load_reviews(input_file)

            # Collect loading errors
            if self.data_loader.has_loading_errors():
                for error in self.data_loader.get_loading_errors():
                    self.error_aggregator.add_error(error, "Data loading")

            return reviews

        except Exception as e:
            self.logger.error(f"Failed to load reviews from {input_file}: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def _preprocess_reviews(self, reviews: List[Review]) -> List[Review]:
        """Preprocess review texts."""
        try:
            preprocessed_reviews = self.preprocessor.preprocess_reviews(reviews)

            # Collect preprocessing errors
            for review in preprocessed_reviews:
                if review.has_errors():
                    for error in review.processing_errors:
                        if "preprocessing" in error.lower():
                            self.error_aggregator.add_error(
                                f"Review {review.review_id}: {error}", "Preprocessing"
                            )

            return preprocessed_reviews

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise RuntimeError(f"Text preprocessing failed: {e}")

    def _analyze_sentiment(self, reviews: List[Review]) -> List[Review]:
        """Analyze sentiment for all reviews."""
        try:
            analyzed_reviews = self.sentiment_analyzer.analyze_sentiment(reviews)

            # Collect analysis errors
            for review in analyzed_reviews:
                if review.has_errors():
                    for error in review.processing_errors:
                        if "sentiment" in error.lower() or "analysis" in error.lower():
                            self.error_aggregator.add_error(
                                f"Review {review.review_id}: {error}",
                                "Sentiment analysis",
                            )

            return analyzed_reviews

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            raise RuntimeError(f"Sentiment analysis failed: {e}")

    def _generate_reports(
        self, reviews: List[Review], output_dir: str
    ) -> Dict[str, str]:
        """Generate summary and detailed reports."""
        try:
            # Get analyzer information for the report
            analyzer_info = self.sentiment_analyzer.get_analyzer_info()

            # Generate reports
            report_files = self.report_generator.generate_reports(
                reviews, output_dir, analyzer_info
            )

            return report_files

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise RuntimeError(f"Report generation failed: {e}")

    def _compile_results(
        self, reviews: List[Review], report_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Compile final results summary."""
        # Calculate statistics
        sentiment_stats = self.report_generator.calculate_statistics(reviews)

        # Compile all errors
        all_errors = self.error_aggregator.get_errors()

        results = {
            "status": "completed",
            "statistics": {
                "total_reviews": sentiment_stats.total_reviews,
                "positive_count": sentiment_stats.positive_count,
                "negative_count": sentiment_stats.negative_count,
                "neutral_count": sentiment_stats.neutral_count,
                "positive_percentage": sentiment_stats.positive_percentage,
                "negative_percentage": sentiment_stats.negative_percentage,
                "neutral_percentage": sentiment_stats.neutral_percentage,
            },
            "files_generated": report_files,
            "processing_info": {
                "total_errors": len(all_errors),
                "errors": all_errors,
                "configuration": {
                    "positive_threshold": self.config.get_positive_threshold(),
                    "negative_threshold": self.config.get_negative_threshold(),
                    "analyzer_type": "VADER",
                },
            },
        }

        return results

    def validate_input_file(self, file_path: str) -> List[str]:
        """
        Validate input file before processing.

        Args:
            file_path: Path to input file

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            file_path_obj = Path(file_path)

            # Check if file exists
            if not file_path_obj.exists():
                errors.append(f"Input file does not exist: {file_path}")
                return errors

            # Check if it's a file
            if not file_path_obj.is_file():
                errors.append(f"Input path is not a file: {file_path}")
                return errors

            # Check file extension
            if file_path_obj.suffix.lower() != ".json":
                errors.append(
                    f"Input file must be a JSON file, got: {file_path_obj.suffix}"
                )

            # Check file size (warn if very large)
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB
                errors.append(
                    f"Warning: Large file size ({file_size_mb:.1f}MB) may require significant processing time"
                )

            # Try to validate JSON structure (basic check)
            try:
                self.data_loader.load_reviews(file_path)
            except Exception as e:
                errors.append(f"File validation failed: {e}")

        except Exception as e:
            errors.append(f"Error validating input file: {e}")

        return errors

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the engine configuration.

        Returns:
            Dictionary with engine information
        """
        return {
            "version": "1.0.0",
            "components": {
                "data_loader": "JSONDataLoader",
                "preprocessor": "ReviewPreprocessor",
                "sentiment_analyzer": "VaderSentimentAnalyzer",
                "report_generator": "ReportGenerator",
            },
            "configuration": self.config.get_config(),
            "analyzer_info": self.sentiment_analyzer.get_analyzer_info(),
            "report_info": self.report_generator.get_report_info(),
        }

    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """
        Update engine configuration.

        Args:
            config_updates: Dictionary with configuration updates
        """
        try:
            # Update configuration
            self.config.update_config(config_updates)

            # Reinitialize components that depend on configuration
            if any(
                key.startswith("sentiment_thresholds") for key in config_updates.keys()
            ):
                self.sentiment_analyzer.update_thresholds(
                    self.config.get_positive_threshold(),
                    self.config.get_negative_threshold(),
                )

            self.logger.info("Configuration updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise RuntimeError(f"Configuration update failed: {e}")
