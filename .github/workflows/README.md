# GitHub Workflows

This directory contains the GitHub Actions CI/CD workflows for TorchFits.

## Active Workflows

### `ci.yml` - Main CI Pipeline
**Triggers:** Push to main/develop, Pull Requests
**Purpose:** Complete build, test, and quality pipeline

**Jobs:**
1. **Code Quality** - Black, isort, mypy, pre-commit checks (Python 3.12)
2. **Build & Test** - Multi-platform builds and tests
   - **OS:** Ubuntu, macOS, Windows  
   - **Python:** 3.10, 3.11, 3.12, 3.13
   - **System deps:** cfitsio, wcslib via package managers
   - **Tests:** Import verification, basic functionality, full test suite, examples
3. **Memory Safety** - Valgrind memory leak detection with stress testing
4. **Performance** - Benchmark suite execution and reporting
5. **Documentation** - Documentation build verification  
6. **Publish** - Automated PyPI publishing on GitHub releases

**Key Features:**
- Cross-platform C++ extension compilation
- Memory leak detection with valgrind
- Performance regression testing
- Automated PyPI releases
- Code quality enforcement

## Backup Workflows

### `memory_tests.yml.backup` - Original Memory Tests
**Status:** Archived (replaced by comprehensive CI)
**Purpose:** Basic memory leak testing with valgrind

The original simple memory testing workflow, kept for reference. The memory testing functionality has been integrated into the main CI pipeline with enhanced stress testing.

## CI Configuration Notes

- **Default Python Version:** 3.12 (for tooling and single-version jobs)
- **Supported Python:** 3.10, 3.11, 3.12, 3.13
- **Build Dependencies:** Automatic installation of cfitsio, wcslib system libraries
- **PyPI Publishing:** Requires `PYPI_API_TOKEN` secret to be configured
- **Memory Testing:** Comprehensive stress testing with 100+ file operations
- **Performance:** Automated benchmark execution with regression detection

## Setup Requirements

To enable PyPI publishing, add the following secret to your GitHub repository:
- `PYPI_API_TOKEN`: Your PyPI API token for automated package publishing

The workflow will automatically publish to PyPI when you create a GitHub release.
