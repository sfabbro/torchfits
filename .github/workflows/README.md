# GitHub Workflows

This directory contains the GitHub Actions CI/CD workflows for torchfits.

## Active Workflows

### `ci.yml` — CI pipeline

**Triggers:** Push to main/develop, pull requests, releases.

**Jobs:** code quality (ruff), multi-version build & test (Python 3.11/3.12 on Ubuntu), C++ backend performance check, and PyPI publish on release.

**System deps:** `cfitsio` (via apt on Linux CI).

### `build_wheels.yml` — Wheel builds

Builds platform wheels for releases.

## Setup for PyPI publishing

Add `PYPI_API_TOKEN` to repository secrets. The publish job runs on GitHub release.
