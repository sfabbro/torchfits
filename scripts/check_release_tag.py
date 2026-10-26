#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re

STABLE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
RC_TAG = re.compile(r"^v\d+\.\d+\.\d+rc\d+$")


def release_kind(tag: str, event: str) -> str:
    if event == "workflow_dispatch":
        if STABLE_TAG.fullmatch(tag):
            return "stable"
    elif event == "push":
        if RC_TAG.fullmatch(tag):
            return "rc"
        if STABLE_TAG.fullmatch(tag):
            return "stable"
    raise ValueError(f"{tag!r} is not allowed for {event}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--event", required=True)
    args = parser.parse_args()
    try:
        print(release_kind(args.tag, args.event))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
