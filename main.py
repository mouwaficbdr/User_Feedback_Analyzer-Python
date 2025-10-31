#!/usr/bin/env python3
"""
Sentiment Analysis Engine - Main Entry Point

A robust Python application for analyzing customer review sentiment.
Processes JSON files containing reviews and generates comprehensive reports.

Usage:
    python main.py [input_file] [--output-dir OUTPUT_DIR] [--config CONFIG_FILE]

Examples:
    python main.py reviews.json
    python main.py reviews.json --output-dir ./results
    python main.py reviews.json --config custom_config.json --output-dir ./output
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.engine import SentimentAnalysisEngine
from src.utils.logger import setup_logger


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Engine - Analyze customer review sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s reviews.json
  %(prog)s reviews.json --output-dir ./results
  %(prog)s reviews.json --config custom_config.json --output-dir ./output
  
The program will generate:
  - summary.json (or .txt): Summary statistics and analysis overview
  - results.csv: Detailed results with sentiment classifications
        """,
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default="reviews.json",
        help="Path to input JSON file containing reviews (default: reviews.json)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to save output files (default: current directory)",
    )

    parser.add_argument(
        "--config", "-c", help="Path to configuration file (default: config.json)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input file without processing",
    )

    parser.add_argument(
        "--version", action="version", version="Sentiment Analysis Engine 1.0.0"
    )

    return parser


def setup_console_logger(verbose: bool = False, quiet: bool = False) -> None:
    """Set up console logging based on verbosity settings."""
    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    setup_logger(
        name="main", log_level=log_level, log_format="%(levelname)s: %(message)s"
    )


def progress_callback(message: str, percentage: int) -> None:
    """Progress callback for user feedback."""
    print(f"[{percentage:3d}%] {message}")


def validate_arguments(args: argparse.Namespace) -> list:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of validation errors
    """
    errors = []

    # Validate input file
    if not os.path.exists(args.input_file):
        errors.append(f"Input file does not exist: {args.input_file}")
    elif not os.path.isfile(args.input_file):
        errors.append(f"Input path is not a file: {args.input_file}")

    # Validate output directory
    output_path = Path(args.output_dir)
    if output_path.exists() and not output_path.is_dir():
        errors.append(f"Output path exists but is not a directory: {args.output_dir}")

    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        errors.append(f"Configuration file does not exist: {args.config}")

    # Check for conflicting options
    if args.verbose and args.quiet:
        errors.append("Cannot use both --verbose and --quiet options")

    return errors


def main() -> int:
    """
    Main entry point for the sentiment analysis engine.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up console logging
    setup_console_logger(args.verbose, args.quiet)
    logger = setup_logger("main")

    try:
        # Validate arguments
        validation_errors = validate_arguments(args)
        if validation_errors:
            for error in validation_errors:
                logger.error(error)
            return 1

        # Initialize engine
        if not args.quiet:
            print("Initializing Sentiment Analysis Engine...")

        engine = SentimentAnalysisEngine(config_path=args.config)

        # Validate input file
        if not args.quiet:
            print(f"Validating input file: {args.input_file}")

        input_validation_errors = engine.validate_input_file(args.input_file)
        if input_validation_errors:
            for error in input_validation_errors:
                if error.startswith("Warning:"):
                    logger.warning(error)
                else:
                    logger.error(error)
                    return 1

        # If validate-only mode, exit here
        if args.validate_only:
            if not args.quiet:
                print("Input file validation completed successfully.")
            return 0

        # Create output directory if it doesn't exist
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run sentiment analysis
        if not args.quiet:
            print("Starting sentiment analysis...")
            print("This may take a few moments depending on the number of reviews.")
            print()

        # Use progress callback only if not in quiet mode
        callback = progress_callback if not args.quiet else None

        results = engine.analyze_reviews(
            input_file=args.input_file,
            output_dir=args.output_dir,
            progress_callback=callback,
        )

        # Display results
        if not args.quiet:
            print()
            print("=" * 50)
            print("SENTIMENT ANALYSIS COMPLETED")
            print("=" * 50)
            print()

            stats = results["statistics"]
            print(f"Total reviews processed: {stats['total_reviews']}")
            print(
                f"Positive reviews: {stats['positive_count']} ({stats['positive_percentage']}%)"
            )
            print(
                f"Negative reviews: {stats['negative_count']} ({stats['negative_percentage']}%)"
            )
            print(
                f"Neutral reviews: {stats['neutral_count']} ({stats['neutral_percentage']}%)"
            )
            print()

            print("Generated files:")
            for file_type, file_path in results["files_generated"].items():
                print(f"  {file_type.capitalize()}: {file_path}")
            print()

            # Show processing info
            processing_info = results["processing_info"]
            if processing_info["total_errors"] > 0:
                print(
                    f"Processing completed with {processing_info['total_errors']} warnings/errors."
                )
                print("Check the log file for details.")
            else:
                print("Processing completed successfully with no errors.")

            print()
            print("Analysis complete! Check the generated files for detailed results.")

        return 0

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        if not args.quiet:
            print("\nAnalysis interrupted by user.")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            logger.exception("Full error details:")

        if not args.quiet:
            print(f"\nError: {e}")
            print("Use --verbose flag for detailed error information.")

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
