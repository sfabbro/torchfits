---
name: docs-authoring-review
description: Comprehensive guidelines for writing, auditing, and reviewing documentation in torchfits. Enforces human-first writing, zero AI/developer jargon, technical precision against src/, clean Mermaid rendering, and multi-tier verification.
disable-model-invocation: true
---

# Documentation Authoring & Review Guidelines (torchfits)

Use when authoring new documentation pages, updating existing guides, or conducting documentation audits.

---

## 1. Core Principles

- **Human-First & Direct:** Write for astronomers, astrophysicists, and machine learning researchers. Lead with direct, runnable Python and CLI examples rather than speculative boilerplate.
- **Zero AI / Developer Jargon:** Strip out internal development terms from user-facing pages:
  - ❌ *"smoke tests"*, *"tax"*, *"lanes"*, *"inventory"*, *"coding agents"*
  - ✅ *"validation"*, *"startup overhead"*, *"environments/targets"*, *"inspection"*, *"developers"*
- **Faithful to Implementation:** Every function signature, parameter name, CLI flag, and default value must strictly match the code in `src/torchfits/`.
- **Astronomical Context:** Use realistic astronomical examples (e.g. multi-band survey cutouts, SDSS/DESI spectra, Chandra event tables, CFHT mosaics) rather than toy random arrays where helpful.

---

## 2. Formatting & Syntax Rules

- **Clickable Links:** Use relative Markdown links with explicit anchor IDs where needed.
- **Mermaid Diagrams:**
  - Quote all node labels containing special characters, brackets, or equations: `node_id["Label Text (Extra Info)"]`.
  - Prefer clean, structured flowcharts (`flowchart LR` or `flowchart TD`) and sequence diagrams.
- **Code Blocks:** Use fenced code blocks with appropriate language tags (`python`, `bash`, `text`, `yaml`, `mermaid`). All Python snippets in docstrings and Markdown are formatted and verified via Ruff.

---

## 3. Verification & Gating

Before submitting any documentation changes, execute all verification tiers:

```bash
# 1. Verify docs build, examples sync, and docs contract tests
pixi run docs-contract

# 2. Check for broken internal links and empty pages
pixi run docs-links

# 3. Code formatting, typing, and preflight checks
pixi run preflight-push
```
