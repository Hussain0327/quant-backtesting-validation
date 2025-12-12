# Contributing to Trading Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/algorithmic-trading-research.git
   cd algorithmic-trading-research
   ```
3. **Set up the development environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they follow the project structure

3. **Run tests** to make sure nothing is broken:
   ```bash
   pytest
   ```

4. **Commit your changes** with a clear message:
   ```bash
   git commit -m "Add: description of your changes"
   ```

5. **Push to your fork** and create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

## Adding a New Strategy

1. Create a new file in `strategies/` (e.g., `strategies/my_strategy.py`)
2. Inherit from `Strategy` base class in `strategies/base.py`
3. Implement required methods:
   - `generate_signals(data)` - returns DataFrame with 'signal' column
   - `get_params()` - returns dict of strategy parameters
4. Add strategy to `strategies/__init__.py`
5. Add strategy template to `STRATEGY_TEMPLATES` in `app.py`
6. Write tests in `tests/test_strategies.py`

Example:
```python
from .base import Strategy

class MyStrategy(Strategy):
    name = "My Strategy"

    def __init__(self, param1=10):
        self.param1 = param1

    def generate_signals(self, data):
        df = data.copy()
        # Your signal logic here
        df['signal'] = 0  # -1, 0, or 1
        return df

    def get_params(self):
        return {'param1': self.param1}
```

## Adding New Metrics

1. Add function to `analytics/metrics.py`
2. Update `calculate_metrics()` to include new metric
3. Add tests in `tests/test_metrics.py`

## Pull Request Guidelines

- **Title**: Clear, concise description of changes
- **Description**: Explain what and why, not just how
- **Tests**: Include tests for new functionality
- **Documentation**: Update README if needed

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Feature Requests

Feature requests are welcome! Please:

- Check existing issues first
- Describe the use case
- Explain why it would be useful

## Questions?

Feel free to open an issue for questions or discussions.

---

Thank you for contributing!
