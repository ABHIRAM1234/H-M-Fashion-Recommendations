# Contributing to H&M Fashion Recommendations

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the H&M Fashion Recommendations solution.

## Project Overview

This project implements a two-stage recommendation system that achieved 45th place out of 3,006 teams in the H&M Personalized Fashion Recommendations Kaggle competition. The solution demonstrates advanced techniques in recommendation systems, machine learning, and data engineering.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ABHIRAM1234/H-M-Fashion-Recommendations.git
   cd H-M-Fashion-Recommendations
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write comprehensive docstrings
- Keep functions focused and modular

## Testing

Run tests to ensure your changes don't break existing functionality:

```bash
pytest tests/
```

## Documentation

- Update docstrings for any new functions or classes
- Update README.md if adding new features
- Ensure all code examples work correctly

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Ensure all tests pass
6. Update documentation
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Code Review

All contributions require review before merging. Please ensure:

- Code follows project style guidelines
- Tests pass
- Documentation is updated
- Changes are well-documented

## Questions or Issues?

If you have questions or encounter issues, please:

1. Check existing issues first
2. Create a new issue with a clear description
3. Provide minimal reproduction steps if reporting a bug

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License. 