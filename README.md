[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-164%2B-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](tests/)

A high-precision sentiment analysis engine for French customer reviews, achieving **91.07% accuracy**. Built with modern NLP techniques and optimized for production use.

## 🏆 Key Metrics

- **Accuracy**: 91.07%
- **Precision**: 92.39%
- **Recall**: 91.07%
- **F1-Score**: 0.9057

## 🎯 Overview

This sentiment analysis engine processes French customer reviews, automatically classifies them (Positive, Negative, Neutral), and generates comprehensive reports for decision-making.

## ✨ Features

- **High Accuracy**: 91.07% accuracy with optimized Hybrid analyzer
- **Multiple Analyzers**: VADER, ML (BERT multilingual), and Hybrid (VADER + ML)
- **Scientific Validation**: Complete metrics (Accuracy, Precision, Recall, F1-Score)
- **Threshold Optimization**: Automatic search for optimal classification thresholds
- **Robust Handling**: Emojis, special characters, empty texts, negations, mixed sentiments
- **Comprehensive Reports**: Detailed statistics, confusion matrices, CSV export
- **CLI Interface**: Simple and intuitive command-line interface
- **Modular Architecture**: Maintainable and extensible codebase
- **Well Tested**: 164+ tests with 90%+ coverage
- **Production Ready**: Advanced error handling ensures system never crashes

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/mouwaficbdr/UserFeedbackAnalyzer-Python.git
cd UserFeedbackAnalyzer-Python

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python validate_installation.py
```

## 📖 Usage

### Basic Usage

```bash
# Analyze default reviews.json file
python main.py

# Analyze a specific file
python main.py my_reviews.json

# Specify output directory
python main.py reviews.json --output-dir ./results
```

### Advanced Options

```bash
# Use custom configuration
python main.py reviews.json --config my_config.json

# Verbose mode for detailed output
python main.py reviews.json --verbose

# Quiet mode (errors only)
python main.py reviews.json --quiet

# Validate input file only
python main.py reviews.json --validate-only

# Show help
python main.py --help
```

### Analyzer Selection

```bash
# Use ML analyzer (BERT multilingual)
python main.py reviews.json --analyzer ml

# Use Hybrid analyzer (VADER + ML) - RECOMMENDED for 91% accuracy
python main.py reviews.json --analyzer hybrid

# Use VADER analyzer (fastest, rule-based)
python main.py reviews.json --analyzer vader
```

### Quality Validation

```bash
# Validate analyzer quality
python main.py --validate-quality validation_dataset.json

# Validate with specific analyzer
python main.py --validate-quality validation_dataset.json --analyzer hybrid

# Optimize classification thresholds
python main.py --optimize-thresholds validation_dataset.json --metric f1_score

# Optimize with specific analyzer and metric
python main.py --optimize-thresholds validation_dataset.json --analyzer hybrid --metric accuracy
```

## 📁 Input Data Format

The input file must be in JSON format with the following structure:

```json
[
  {
    "review_id": "REV001",
    "review_text": "Excellent produit, je le recommande vivement !"
  },
  {
    "review_id": "REV002",
    "review_text": "Service client décevant."
  }
]
```

### Supported Formats

- **Simple structure**: List of objects with `review_id` and `review_text`
- **Wrapped structure**: `{"reviews": [...]}`
- **Alternative fields**: `id`, `text`, `content` are automatically detected
- **Robust handling**: Empty texts, special characters, emojis

## 📊 Output Files

### Summary Report (`summary.json`)

```json
{
  "analysis_summary": {
    "total_reviews": 50,
    "sentiment_distribution": {
      "positive": { "count": 12, "percentage": 24.0 },
      "negative": { "count": 9, "percentage": 18.0 },
      "neutral": { "count": 29, "percentage": 58.0 }
    },
    "processing_info": {
      "timestamp": "2025-10-31T10:00:00Z",
      "errors_count": 2,
      "configuration": {
        "positive_threshold": 0.05,
        "negative_threshold": -0.05
      }
    }
  }
}
```

### Detailed Results (`results.csv`)

```csv
review_id,review_text,sentiment_final,sentiment_score,processing_errors
REV001,"Excellent produit !",Positive,0.8516,
REV002,"Service décevant",Negative,-0.7269,
REV003,"",Neutral,0.0,Review text is empty
```

## ⚙️ Configuration

### Configuration File (`config.json` or `config.example.json`)

```json
{
  "sentiment_thresholds": {
    "positive": 0.05,
    "negative": -0.05
  },
  "output": {
    "summary_format": "json",
    "results_format": "csv"
  },
  "logging": {
    "level": "INFO",
    "file": "sentiment_analysis.log"
  }
}
```

### Threshold Justification

- **Positive threshold (0.28)**: Score > 0.28 for positive classification
- **Negative threshold (-0.28)**: Score < -0.28 for negative classification
- **Neutral zone**: Between -0.28 and 0.28 for ambiguous sentiments

These optimized thresholds achieve 91.07% accuracy on French customer reviews. Adjust based on your use case:

- **E-commerce**: ±0.05 (balanced)
- **Social media**: ±0.1 (stricter)
- **Customer support**: ±0.03 (more sensitive)

## 🏗️ Architecture

### Project Structure

```
UserFeedbackAnalyzer-Python/
├── src/                    # Main source code
│   ├── models/            # Data models
│   ├── data/              # Data loading
│   ├── preprocessing/     # Text preprocessing
│   ├── analysis/          # Sentiment analysis
│   ├── validation/        # Quality validation
│   ├── optimization/      # Threshold optimization
│   ├── reporting/         # Report generation
│   ├── config/            # Configuration management
│   └── utils/             # Utilities
├── tests/                 # Unit and integration tests
├── main.py               # Entry point
├── config.example.json   # Example configuration
└── requirements.txt      # Dependencies
```

### Main Components

1. **DataLoader**: Robust JSON file loading with encoding detection
2. **ReviewPreprocessor**: Text cleaning and normalization
3. **SentimentAnalyzers**: VADER, ML (BERT), and Hybrid analyzers
4. **ReportGenerator**: Summary and detailed report generation
5. **SentimentAnalysisEngine**: Main pipeline orchestrator
6. **SentimentValidator**: Quality metrics and validation
7. **ThresholdOptimizer**: Automatic threshold optimization

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run integration tests only
python -m pytest tests/test_integration.py -v

# Run specific test file
python -m pytest tests/test_engine_comprehensive.py -v
```

### Test Coverage

The project maintains 90%+ test coverage with comprehensive tests for:

- All main components
- Error cases and edge cases
- Complete integration pipeline
- Problematic data handling
- 164+ tests total

## 🔧 Development

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Check style with Flake8
flake8 src/ tests/

# Run all quality checks
black src/ tests/ && flake8 src/ tests/ && pytest tests/
```

### Adding New Features

1. **New analyzers**: Implement `SentimentAnalyzerInterface`
2. **New formats**: Extend `DataLoaderInterface`
3. **New reports**: Modify `ReportGenerator`

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 🚨 Error Handling

The system is designed to **never crash**:

- **Corrupted files**: Automatic detection and recovery
- **Encoding issues**: Fallback to multiple encodings (UTF-8, Latin-1, CP1252)
- **Missing data**: Default values and detailed logging
- **Limited resources**: Memory management and batch processing

## 📈 Performance

### Optimizations

- **Batch processing**: Configurable based on available memory
- **Memory management**: Automatic monitoring and optimization
- **Smart caching**: Reuse of expensive computations
- **Efficient logging**: Automatic log file rotation

### Benchmarks

- **50 reviews**: < 5 seconds
- **500 reviews**: < 30 seconds
- **5000 reviews**: < 5 minutes
- **Memory usage**: ~340MB for ML models

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Code style guidelines (PEP 8, Black, Flake8)
- Testing requirements (80%+ coverage)
- Pull request process
- Development setup

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`black`, `flake8`, `pytest`)
5. Submit a pull request

## 📝 Documentation

- **[README.md](README.md)**: This file - Quick start and overview
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contribution guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Encoding errors**:

```bash
# Check file encoding
file -i reviews.json
# System automatically handles UTF-8, Latin-1, CP1252
```

**Out of memory**:

```json
// Reduce batch size in config.json
{
  "processing": {
    "batch_size": 50
  }
}
```

**Unexpected results**:

```bash
# Use verbose mode for diagnostics
python main.py reviews.json --verbose
```

## 🙏 Acknowledgments

- Built with [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- ML models from [HuggingFace Transformers](https://huggingface.co/transformers/)
- BERT multilingual model: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## 📧 Support

For questions or issues:

- Open an issue on GitHub
- Check the documentation in this README
- Use `--verbose` mode for detailed diagnostics

---

**Made with ❤️ by mouwaficbdr**
