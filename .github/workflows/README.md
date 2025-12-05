# GitHub Workflows

This directory contains the GitHub Actions CI/CD workflows for torchfits.

## Active Workflows

### `ci.yml` - Comprehensive CI Pipeline
**Triggers:** Push to main/develop branches, Pull Requests, Releases  
**Purpose:** Complete build, test, quality assurance, and deployment pipeline

**Jobs:**
1. **Code Quality** - Code formatting and linting (Python 3.12)
   - black (formatting)
   - ruff (linting)
   
2. **Build & Test** - Multi-version builds and tests
   - **OS:** Ubuntu
   - **Python:** 3.11, 3.12
   - **System deps:** cfitsio, wcslib
   - **Tests:** Full pytest suite
   
3. **Memory Safety** - Memory leak detection (placeholder for valgrind)

4. **Performance Benchmarks** - Performance regression testing (placeholder)

5. **Documentation** - Documentation build verification (placeholder)

6. **Publish** - Automated PyPI publishing on GitHub releases

**Key Features:**
- C++ extension compilation with cfitsio/wcslib
- Cross-Python version testing
- Code quality enforcement
- Automated PyPI releases

## CI Configuration

- **Python Versions:** 3.11, 3.12
- **Code Quality Tools:** black, ruff
- **Build Dependencies:** cfitsio, wcslib (installed via apt)
- **PyPI Publishing:** Requires `PYPI_API_TOKEN` repository secret

## Setup for PyPI Publishing

Add the following secret to your GitHub repository settings:
- `PYPI_API_TOKEN`: Your PyPI API token

The workflow automatically publishes to PyPI when you create a GitHub release.
