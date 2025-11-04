#!/usr/bin/env python3
"""
Installation validation script for the sentiment analysis engine.

This script verifies that all components are properly installed
and working as expected.
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path


def print_header(title):
    """Display a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_step(step_name, status="RUNNING"):
    """Display the status of a step."""
    status_symbols = {
        "RUNNING": "⏳",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️"
    }
    symbol = status_symbols.get(status, "❓")
    print(f"{symbol} {step_name}")


def check_python_version():
    """Check Python version."""
    print_step("Checking Python version", "RUNNING")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_step(f"Python {version.major}.{version.minor}.{version.micro} - OK", "SUCCESS")
        return True
    else:
        print_step(f"Python {version.major}.{version.minor}.{version.micro} - Insufficient version (required: 3.8+)", "ERROR")
        return False


def check_dependencies():
    """Check Python dependencies."""
    print_step("Checking dependencies", "RUNNING")
    
    required_packages = [
        "vaderSentiment",
        "pandas",
        "numpy",
        "chardet",
        "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_step(f"  {package} - OK", "SUCCESS")
        except ImportError:
            print_step(f"  {package} - MISSING", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_step(f"Missing dependencies: {', '.join(missing_packages)}", "ERROR")
        return False
    else:
        print_step("All dependencies are installed", "SUCCESS")
        return True


def check_project_structure():
    """Check project structure."""
    print_step("Checking project structure", "RUNNING")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "config.example.json",
        "README.md",
        "src/__init__.py",
        "src/models/review.py",
        "src/data/loader.py",
        "src/preprocessing/preprocessor.py",
        "src/analysis/sentiment_analyzer.py",
        "src/reporting/report_generator.py",
        "src/config/config_manager.py",
        "src/engine.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_step(f"  {file_path} - OK", "SUCCESS")
        else:
            print_step(f"  {file_path} - MISSING", "ERROR")
            missing_files.append(file_path)
    
    if missing_files:
        print_step(f"Missing files: {', '.join(missing_files)}", "ERROR")
        return False
    else:
        print_step("Project structure is correct", "SUCCESS")
        return True


def test_basic_functionality():
    """Test basic functionality."""
    print_step("Testing basic functionality", "RUNNING")
    
    try:
        # Import main modules
        from src.models.review import Review, SentimentResult
        from src.data.loader import JSONDataLoader
        from src.analysis.sentiment_analyzer import VaderSentimentAnalyzer
        from src.engine import SentimentAnalysisEngine
        
        print_step("  Module imports - OK", "SUCCESS")
        
        # Test object creation
        review = Review(review_id="TEST001", review_text="Test review")
        loader = JSONDataLoader()
        analyzer = VaderSentimentAnalyzer()
        engine = SentimentAnalysisEngine()
        
        print_step("  Object creation - OK", "SUCCESS")
        
        # Test simple analysis
        test_review = Review(review_id="TEST002", review_text="Excellent produit !")
        analyzed = analyzer.analyze_single_review(test_review)
        
        if analyzed.sentiment_label in ["Positive", "Negative", "Neutral"]:
            print_step("  Sentiment analysis - OK", "SUCCESS")
        else:
            print_step("  Sentiment analysis - ERROR", "ERROR")
            return False
        
        return True
        
    except Exception as e:
        print_step(f"Error during test: {e}", "ERROR")
        return False


def test_end_to_end():
    """Test complete pipeline."""
    print_step("Testing complete pipeline", "RUNNING")
    
    try:
        # Create test data
        test_data = [
            {"review_id": "VAL001", "review_text": "Excellent produit, je le recommande !"},
            {"review_id": "VAL002", "review_text": "Service client horrible."},
            {"review_id": "VAL003", "review_text": "Produit correct, sans plus."},
            {"review_id": "VAL004", "review_text": ""},  # Test with empty text
            {"review_id": "VAL005", "review_text": "Très satisfait ! 😀"}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = os.path.join(temp_dir, "test_reviews.json")
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            print_step("  Test file created - OK", "SUCCESS")
            
            # Run analysis
            from src.engine import SentimentAnalysisEngine
            engine = SentimentAnalysisEngine()
            
            results = engine.analyze_reviews(input_file, temp_dir)
            
            # Verify results
            if results["status"] == "completed":
                print_step("  Pipeline executed - OK", "SUCCESS")
            else:
                print_step("  Pipeline executed - ERROR", "ERROR")
                return False
            
            if results["statistics"]["total_reviews"] == 5:
                print_step("  Review count - OK", "SUCCESS")
            else:
                print_step("  Review count - ERROR", "ERROR")
                return False
            
            # Verify output files
            summary_file = os.path.join(temp_dir, "summary.json")
            results_file = os.path.join(temp_dir, "results.csv")
            
            if os.path.exists(summary_file) and os.path.exists(results_file):
                print_step("  Output files generated - OK", "SUCCESS")
            else:
                print_step("  Output files generated - ERROR", "ERROR")
                return False
            
            return True
            
    except Exception as e:
        print_step(f"Error during E2E test: {e}", "ERROR")
        return False


def test_command_line():
    """Test command-line interface."""
    print_step("Testing command-line interface", "RUNNING")
    
    try:
        # Test help command
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and "Sentiment Analysis Engine" in result.stdout:
            print_step("  --help command - OK", "SUCCESS")
        else:
            print_step("  --help command - ERROR", "ERROR")
            return False
        
        # Test validation
        if os.path.exists("reviews.json"):
            result = subprocess.run([sys.executable, "main.py", "--validate-only"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print_step("  File validation - OK", "SUCCESS")
            else:
                print_step("  File validation - ERROR", "ERROR")
                return False
        else:
            print_step("  reviews.json file not found - IGNORED", "WARNING")
        
        return True
        
    except Exception as e:
        print_step(f"Error during CLI test: {e}", "ERROR")
        return False


def run_unit_tests():
    """Run unit tests."""
    print_step("Running unit tests", "RUNNING")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count passed tests
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "passed" in line:
                    print_step(f"  Unit tests - {line.strip()}", "SUCCESS")
                    break
            return True
        else:
            print_step(f"  Unit tests - FAILED", "ERROR")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print_step(f"Error during unit tests: {e}", "ERROR")
        return False


def main():
    """Main validation function."""
    print_header("INSTALLATION VALIDATION")
    print("This script verifies that the sentiment analysis engine")
    print("is properly installed and functional.")
    
    # List of checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Basic Functionality", test_basic_functionality),
        ("Complete Pipeline", test_end_to_end),
        ("Command-Line Interface", test_command_line),
        ("Unit Tests", run_unit_tests)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        print_header(f"CHECK: {check_name}")
        results[check_name] = check_function()
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for check_name, success in results.items():
        status = "SUCCESS" if success else "ERROR"
        print_step(f"{check_name}", status)
    
    print(f"\nResult: {success_count}/{total_count} checks passed")
    
    if success_count == total_count:
        print_step("INSTALLATION VALIDATED SUCCESSFULLY", "SUCCESS")
        print("\n🎉 The sentiment analysis engine is ready to use!")
        print("\nBasic commands:")
        print("  python main.py reviews.json")
        print("  python main.py --help")
        return 0
    else:
        print_step("PROBLEMS DETECTED IN INSTALLATION", "ERROR")
        print("\n❌ Please fix the errors before using the system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())