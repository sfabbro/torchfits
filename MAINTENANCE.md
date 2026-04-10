# torchfits maintenance policy

**Active development has moved to [torchsky](https://github.com/sfabbro/torchsky)** — a torch-native astronomy stack that supersedes torchfits for new features and performance work.

## Release lines

- **v0.3.2** and the **0.3.x** branch are **frozen** except for:
  - **Security** fixes (backports as needed)
  - **Trivial** maintenance (packaging, metadata, obvious one-line fixes) when required for PyPI or CI
- A possible **0.3.3** (or similar) may appear only for the above; do not expect API or feature expansion here.

## Where to contribute

- New FITS I/O, WCS, transforms, sphere, and mapmaking work: **torchsky**
- Historical context: see torchsky’s `THREAD_HISTORY_FROM_TORCHFITS.md`
