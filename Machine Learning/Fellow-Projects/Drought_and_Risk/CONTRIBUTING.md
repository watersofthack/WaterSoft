# Contributing to WaterSoft Hydrological ML

We welcome contributions to the WaterSoft Hydrological ML project! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/LSTM-SSI.git
   cd LSTM-SSI
   ```
3. **Set up the development environment**:
   ```bash
   conda env create -f environment.yml
   conda activate watersoft-ml
   ```

## Development Workflow

1. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black src/ tests/
  ```
- Use [isort](https://pycqa.github.io/isort/) for import sorting:
  ```bash
  isort src/ tests/
  ```
- Use [flake8](https://flake8.pycqa.org/) for linting:
  ```bash
  flake8 src/ tests/
  ```

### Documentation

- Write clear, concise docstrings for all functions and classes
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Update README.md if your changes affect usage
- Add inline comments for complex logic

### Testing

- Write unit tests for new functions and classes
- Ensure all tests pass before submitting a PR
- Aim for good test coverage of new code
- Use descriptive test names and docstrings

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, package versions)
- Relevant error messages or logs

### Feature Requests

For new features, please:
- Describe the problem your feature would solve
- Explain your proposed solution
- Consider alternative approaches
- Discuss potential impacts on existing functionality

### Code Contributions

We welcome contributions in these areas:
- **Model improvements**: New architectures, hyperparameter optimization
- **Data processing**: Enhanced preprocessing, new data sources
- **Reservoir operations**: Advanced decision algorithms, optimization
- **Visualization**: Better plots, interactive dashboards
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests, benchmarks

## Project Structure

```
LSTM-SSI/
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing modules
│   ├── models/            # Model implementations
│   ├── postprocessing/    # Analysis and reservoir operations
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── configs/               # Configuration files
├── data/                  # Data directory (not in git)
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- **Add**: New features or files
- **Fix**: Bug fixes
- **Update**: Changes to existing functionality
- **Remove**: Deleted features or files
- **Docs**: Documentation changes
- **Test**: Test-related changes
- **Refactor**: Code restructuring without functionality changes

Examples:
```
Add: LSTM model with attention mechanism
Fix: Handle missing values in streamflow data
Update: Improve SSI calculation performance
Docs: Add usage examples for reservoir operations
```

## Code Review Process

1. All submissions require review before merging
2. Reviewers will check for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Compatibility with existing code
3. Address reviewer feedback promptly
4. Maintain a respectful, collaborative tone

## Development Environment

### Required Tools

- Python 3.8+
- Git
- Conda or pip for package management

### Optional Tools

- Jupyter Lab/Notebook for interactive development
- VS Code or PyCharm for IDE
- Docker for containerized development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/

# Run specific test file
python -m pytest tests/test_preprocessing.py
```

### Building Documentation

```bash
cd docs/
make html
```

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact the maintainers at contact@watersoft-ml.org

## Recognition

Contributors will be acknowledged in:
- The project README
- Release notes for significant contributions
- Academic publications (where appropriate)

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to WaterSoft Hydrological ML! 🌊
