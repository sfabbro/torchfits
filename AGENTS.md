# Agent guidance (torchfits)

Pixi-first: use `pixi run …`, never bare `python` for project work.

## Verify tiers

| When | Command |
|---|---|
| During edits | `pixi run preflight-push` |
| Before push / PR | `pixi run ci-local` |
| Before tag | `pixi run release-gate` |

Durable notes live under `.cursor/harness/` (playbook, trajectories) — not long
chat scrollback. Deferred product work: [`.cursor/post-1.0-backlog.md`](.cursor/post-1.0-backlog.md).

## Git workflow

`origin` is the **fork** `sfabbro/torchfits`; `upstream` is `astroai/torchfits`.

- Work is committed **directly on the fork's `main`** — do not open feature
  branches (`wip/*`). A branch is only ever a temporary landing vehicle, and a
  merged one is deleted.
- Keep the fork in lockstep with `astroai/torchfits`:

  ```bash
  git fetch upstream && git merge --ff-only upstream/main
  git push origin main
  ```

- The fork carries exactly one branch (`main`) and between rounds is
  `identical` to `upstream/main`:

  ```bash
  gh api repos/astroai/torchfits/compare/main...sfabbro:main --jq '.status'
  ```

## Humans / coding agents

- Docs must match the public façade (`docs/api*.md`); env vars must exist in `src/`.
- **Documentation Standards:**
  - **Human-First & Direct:** Prioritize clear, copy-pasteable Python and CLI recipes over boilerplate or speculative notes.
  - **Zero AI / Developer Jargon:** Avoid internal terminology ("lanes", "tax", "smoke", "inventory", "coding agents") in user-facing documentation.
  - **Faithful to Code & Faustian Rigor:** All documented signatures, flags, exit codes, and examples must match `src/torchfits/` implementations.
  - **Mermaid Best Practices:** Always quote labels with special characters (`id["Label (info)"]`) to prevent syntax rendering glitches.
  - **Verification:** Run `pixi run docs-contract`, `pixi run docs-links`, and `pixi run preflight-push` on any docs change.
- Prefer smallest correct diffs; no new dependencies without a clear need.
- Pre-tag public-API audit: [`.cursor/skills/release-api-freeze-review/SKILL.md`](.cursor/skills/release-api-freeze-review/SKILL.md).

## Jules / autonomous PR agents

Prefer **correctness** and **measurable performance** over style, a11y drive-bys,
renames, comment churn, or docs-only nits.

Before opening a PR:

1. Read [`.cursor/jules-ledger.md`](.cursor/jules-ledger.md) and search recently
   merged Jules PRs — **do not repeat a theme already landed**.
2. Cite evidence: a failing test / assert that fails before and passes after, a
   CFITSIO or public API contract, or before/after timing from an existing
   `pixi run bench-*` case (same host, same `case_id`).
3. One logical change per PR; title names the bug or the bench case.
4. No new dependencies; no SemVer bumps; no force-push.
5. Run `pixi run preflight-push` (or the smallest relevant pytest) before opening
   the PR.

Out of scope unless a human explicitly labels the issue: HTML / `repr_html`
cosmetics, markdown wording, “clean up” without a repro.

If research finds nothing serious: **open no PR**.

Weekly Jules prompt: [`.cursor/jules.md`](.cursor/jules.md).

## Env hygiene

- Prefer `pixi run` / `pixi run python` over bare `python3` when Pixi exists.
- Never `pip install --user` or install into `~/.local` / `$HOME/.local` (esp. CANFAR `/arc/home`).
- Headless/batch: `export PYTHONNOUSERSITE=1` and `unset PYTHONPATH`.
- On CANFAR: read skill `canfar-lab-workflow` (mounts, quotas, resources, headless, ports).
