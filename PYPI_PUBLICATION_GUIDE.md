# PyPI Publication Guide - torchfits v0.1.1

## Overview

This guide walks you through publishing torchfits v0.1.1 to PyPI. This will be the **first public release** on PyPI.

## Prerequisites

✅ **Completed:**
- Version 0.1.1 in all files
- Git tag v0.1.1 created and pushed
- Source distribution built: `dist/torchfits-0.1.1.tar.gz`
- Release notes created: `RELEASE_NOTES_v0.1.1.md`
- `twine` installed

## Step-by-Step Publication

### Step 1: Verify Package Contents

```bash
# Check what's in the distribution
tar -tzf dist/torchfits-0.1.1.tar.gz | head -20

# Verify package metadata
pixi run python -m build --sdist --no-isolation  # Already done ✓
```

### Step 2: Check Package with Twine

```bash
pixi run twine check dist/*
```

This validates:
- README.md renders correctly on PyPI
- Package metadata is valid
- No critical issues

### Step 3: (Optional but Recommended) Test on TestPyPI

TestPyPI is a separate instance for testing uploads.

```bash
# Upload to TestPyPI
pixi run twine upload --repository testpypi dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your TestPyPI API token>
```

**Get TestPyPI token:**
1. Go to https://test.pypi.org/manage/account/token/
2. Create a new API token
3. Copy the token (starts with `pypi-`)

**Test installation from TestPyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ --no-deps torchfits
```

### Step 4: Upload to PyPI (Production)

**IMPORTANT:** This is irreversible. Once uploaded, you cannot delete or re-upload the same version.

```bash
pixi run twine upload dist/*

# You'll be prompted for:
# Username: __token__
# Password: <your PyPI API token>
```

**Get PyPI token:**
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Scope: "Entire account" (for first upload) or "Project: torchfits" (for updates)
4. Copy the token (starts with `pypi-`)

**Alternative: Use .pypirc file**

Create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

Then upload without prompts:
```bash
pixi run twine upload dist/*
```

### Step 5: Verify on PyPI

After upload, check:
- Package page: https://pypi.org/project/torchfits/
- README renders correctly
- Metadata is accurate
- Download link works

### Step 6: Test Installation

```bash
# In a fresh environment
pip install torchfits

# Verify
python -c "import torchfits; print(torchfits.__version__)"
# Should print: 0.1.1
```

### Step 7: Create GitHub Release

1. Go to https://github.com/sfabbro/torchfits/releases/new
2. Choose tag: `v0.1.1`
3. Release title: `v0.1.1 - First PyPI Release`
4. Copy content from `RELEASE_NOTES_v0.1.1.md`
5. Attach `dist/torchfits-0.1.1.tar.gz`
6. Click "Publish release"

## Troubleshooting

### "File already exists"
- You cannot re-upload the same version
- Bump version to 0.1.2 and try again

### "Invalid distribution"
- Run `pixi run twine check dist/*` for details
- Common issues: README formatting, missing metadata

### "Authentication failed"
- Verify token is correct
- Ensure username is `__token__` (not your PyPI username)
- Check token hasn't expired

### Build fails
- Ensure all dependencies available: `pixi install`
- Try building with: `pixi run python -m build --sdist --no-isolation`

## Post-Publication Checklist

- [ ] Package visible on PyPI
- [ ] `pip install torchfits` works
- [ ] GitHub release created
- [ ] Update documentation links if needed
- [ ] Announce on relevant channels (optional)

## Next Steps

For future releases:
1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create git tag
4. Build: `pixi run python -m build --sdist --no-isolation`
5. Upload: `pixi run twine upload dist/*`

## Notes

- **Source-only distribution**: We're uploading source distribution only (no pre-built wheels)
- Users will compile the C++ extension during `pip install`
- This requires users to have cfitsio and wcslib installed
- For better user experience, consider building wheels for common platforms in the future

## Security

- **Never commit** `.pypirc` with tokens to git
- Store tokens securely (password manager, GitHub secrets)
- Use scoped tokens (project-specific) when possible
- Rotate tokens periodically
