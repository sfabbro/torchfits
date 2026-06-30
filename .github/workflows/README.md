# GitHub Workflows

This directory contains the GitHub Actions CI/CD workflows for torchfits.

## Active Workflows

### `ci.yml` — CI pipeline

**Triggers:** Push to main/develop, pull requests, releases.

**Jobs:** code quality (ruff), multi-OS build & test (Python 3.10–3.13 on Ubuntu and macOS), C++ backend performance check, and PyPI publish on release.

**CFITSIO:** vendored via `extern/vendor.sh` and `extern/VERSIONS.txt` (not a system apt dependency).

### `build_wheels.yml` — Wheel builds

Builds platform wheels for releases.

## Setup for PyPI publishing

Add `PYPI_API_TOKEN` to repository secrets. The publish job runs on GitHub release.
