# Contributing to torchfits

Thank you for your interest in contributing to torchfits! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

Before setting up the development environment, make sure you have the required system libraries:

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install libcfitsio-dev libwcs-dev
```

**macOS (Homebrew):**

```bash
brew install cfitsio wcslib
```

### Setting Up the Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/sfabbro/torchfits.git
   cd torchfits
   ```

2. Create and activate a development environment using pixi:

   ```bash
   pixi run dev-setup
   ```
   
   Or use pip directly:

   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

### Code Style

We use:

- [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](http://mypy-lang.org/) for static type checking

You can format your code by running:

```bash
pixi run format
# or
black src tests examples && isort src tests examples
```

### Running Tests

Run the test suite with:

```bash
pixi run test
# or
pytest
```

To run specific tests:

```bash
pytest tests/test_specific_file.py
pytest tests/test_specific_file.py::test_specific_function
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. These are automatically installed with `pixi run dev-setup`, but you can run them manually with:

```bash
pre-commit run --all-files
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Make your changes, including appropriate test cases.
3. Ensure all tests pass and the code is properly formatted.
4. Submit a pull request to the `main` branch.

## Modern Package Structure

This package uses modern Python packaging standards:

- **`pyproject.toml`**: All package configuration
- **`build.py`**: C++ extension building (called automatically)
- **No `setup.py`**: We use the modern build system only

### Building the Package

```bash
pip install build
python -m build
```

## Release Process

For maintainers only:

1. Update the version in `src/torchfits/version.py` and `pyproject.toml`.
2. Update the CHANGELOG.md file.
3. Commit the changes: `git commit -m "Release v{version}"`.
4. Tag the release: `git tag v{version}`.
5. Push to the repository: `git push origin main --tags`.
6. Build and publish the package to PyPI:

   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## License

By contributing to torchfits, you agree that your contributions will be licensed under the project's GPL-2.0 License.
