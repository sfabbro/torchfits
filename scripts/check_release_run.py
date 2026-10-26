#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys


def has_successful_run(runs: list[dict[str, object]], tag: str, sha: str) -> bool:
    return any(
        run.get("event") == "push"
        and run.get("conclusion") == "success"
        and run.get("headBranch") == tag
        and run.get("headSha") == sha
        for run in runs
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    runs = json.load(sys.stdin)
    if not isinstance(runs, list) or not has_successful_run(runs, args.tag, args.sha):
        print(
            f"no successful push workflow run for {args.tag} at {args.sha}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
