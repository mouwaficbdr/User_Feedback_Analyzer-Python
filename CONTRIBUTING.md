# Contributing to User Feedback Analyzer

Thank you for your interest in contributing to User Feedback Analyzer! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (venv)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/mouwaficbdr/UserFeedbackAnalyzer-Python.git
cd UserFeedbackAnalyzer-Python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
pytest tests/ -v
```

## 🤝 How to Contribute

### 1. Fork the Repository

Click the "Fork" button at the top right of the repository page.

### 2. Create a Feature Branch

```bash
git checkout -b feature/amazing-feature
```

Branch naming conventions:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### 3. Make Your Changes

- Write clean, readable code
- Follow the project's code style (PEP 8)
- Add tests for new features
- Update documentation as needed

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Check code style
black src/ tests/ --check
flake8 src/ tests/
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add amazing feature"
```

Commit message guidelines:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- First line should be 50 characters or less
- Reference issues and pull requests when relevant

### 6. Push to Your Fork

```bash
git push origin feature/amazing-feature
```

### 7. Open a Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Provide a clear description of your changes
- Link any related issues

## 📝 Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters (Black default)

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Update README.md if adding new features
- Add inline comments for complex logic

Example docstring:

```python
def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a given text.

    Args:
        text: The text to analyze

    Returns:
        Dictionary containing sentiment classification and score

    Raises:
        ValueError: If text is empty or invalid
    """
    pass
```

### Testing

- Write unit tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Test edge cases and error conditions

Example test:

```python
def test_analyze_sentiment_positive():
    """Test sentiment analysis with positive text."""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze("This is excellent!")
    assert result["sentiment"] == "Positive"
    assert result["score"] > 0.5
```

## 🐛 Reporting Bugs

### Before Submitting a Bug Report

- Check existing issues to avoid duplicates
- Verify the bug exists in the latest version
- Collect relevant information (error messages, logs, etc.)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Run command '...'
2. With input file '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**

- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10]
- Package version: [e.g., 1.0.0]

**Additional context**
Any other relevant information.
```

## 💡 Suggesting Enhancements

### Enhancement Suggestion Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 🔍 Code Review Process

1. All submissions require review before merging
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation
   - Performance implications
3. Address review comments promptly
4. Be open to feedback and suggestions

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] Tests pass locally (`pytest tests/`)
- [ ] New tests added for new features
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Code is formatted with Black
- [ ] Flake8 shows no errors

## 🎯 Development Priorities

Current focus areas:

1. Improving sentiment analysis accuracy
2. Adding support for more languages
3. Performance optimizations
4. Better error handling and logging
5. Enhanced documentation

## 📚 Resources

- [Project Documentation](README.md)
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Project documentation

## 📧 Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Check existing documentation
- Review closed issues for similar questions

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

Thank you for contributing to User Feedback Analyzer! 🎉
