"""Shared CLI helpers: paths, HDU selection, structured output, exit codes."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO, TypeVar

import torch

from torchfits._io_engine.paths import cfitsio_base_path

EXIT_OK = 0
EXIT_DIFF = 1
EXIT_USAGE = 2
EXIT_IO = 3
EXIT_VERIFY_FAIL = 4

_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "vos://", "vos:", "vault:")
_EMIT_FORMATS = ("text", "json", "jsonl")
_SPLIT_MODES = ("file", "hdu")
_JOBS_HELP = (
    "PyTorch intra-op threads (default: 0 = CPU count). "
    "Use -J/--file-jobs for multi-file fan-out."
)
_FILE_JOBS_HELP = (
    "parallel file workers via a Python thread pool (default: 0 = CPU count "
    "when ≥2 files, else 1). Caps PyTorch intra-op threads to 1 per worker."
)

T = TypeVar("T")
R = TypeVar("R")


class CliError(Exception):
    """Base CLI error with a stable exit code."""

    exit_code = EXIT_USAGE

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UsageError(CliError):
    exit_code = EXIT_USAGE


class IoError(CliError):
    exit_code = EXIT_IO


def is_remote_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(_REMOTE_PREFIXES)


def resolve_paths(paths: list[str] | None, *, use_stdin: bool) -> list[str]:
    resolved = [p for p in (paths or []) if p]
    if use_stdin or (not resolved and not sys.stdin.isatty()):
        for line in sys.stdin:
            item = line.strip()
            if item:
                resolved.append(item)
    if not resolved:
        raise UsageError("no input paths (argv or stdin)")
    return resolved


def parse_hdu_list(hdu: str | None) -> list[int] | None:
    if hdu is None:
        return None
    out: list[int] = []
    for part in hdu.split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError as exc:
            raise UsageError(f"invalid HDU index: {piece!r}") from exc
    if not out:
        raise UsageError("--hdu requires at least one HDU index")
    return out


def selected_hdu_indices(num_hdus: int, hdu: str | None) -> list[int]:
    if num_hdus <= 0:
        return []
    wanted = parse_hdu_list(hdu)
    if wanted is None:
        return list(range(num_hdus))
    for idx in wanted:
        if idx < 0 or idx >= num_hdus:
            raise UsageError(f"HDU {idx} out of range (0..{num_hdus - 1})")
    return wanted


def hdu_type_name(header: Any, hdu_obj: Any) -> str:
    from .._hdu.table_hdu_ref import TableHDURef

    if isinstance(hdu_obj, TableHDURef):
        return "TABLE"
    xtension = str(header.get("XTENSION", "")).strip().upper()
    if xtension in {"BINTABLE", "TABLE", "A3DTABLE"}:
        return "TABLE"
    if xtension == "IMAGE":
        return "IMAGE"
    naxis = header.get("NAXIS")
    try:
        if naxis is not None and int(naxis) > 0:
            return "IMAGE"
    except (TypeError, ValueError):
        pass
    return "UNKNOWN"


def header_extname(header: Any, index: int) -> str:
    name = header.get("EXTNAME")
    if name:
        return str(name)
    return "PRIMARY" if index == 0 else f"HDU{index}"


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def add_hdu_arg(
    parser: Any,
    *,
    default: Any = None,
    type: Any = None,
    help: str = "comma-separated HDU indices (default: all)",
) -> None:
    """Add ``-e`` / ``--hdu`` (``-e`` avoids clashing with ``-h`` help)."""
    kwargs: dict[str, Any] = {"help": help}
    if default is not None:
        kwargs["default"] = default
    if type is not None:
        kwargs["type"] = type
    parser.add_argument("-e", "--hdu", **kwargs)


def add_out_arg(parser: Any, *, help: str = "output path") -> None:
    """Add ``-o`` / ``--out`` plus optional positional ``output`` alias."""
    parser.add_argument("-o", "--out", default=None, help=help)
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=f"{help} (positional alias of -o/--out)",
    )


def resolve_out_path(args: Any) -> str:
    """Resolve ``-o`` / ``--out`` vs positional ``output`` (same path required)."""
    flag = getattr(args, "out", None)
    positional = getattr(args, "output", None)
    if flag and positional and flag != positional:
        raise UsageError("conflicting output paths: use either -o/--out or positional")
    out = flag or positional
    if not out:
        raise UsageError("output path required (-o/--out or positional)")
    return str(out)


def add_keyword_arg(parser: Any, **kwargs: Any) -> None:
    """Add ``-k`` / ``--keyword`` (append by default)."""
    kwargs.setdefault("action", "append")
    kwargs.setdefault("dest", "keywords")
    kwargs.setdefault(
        "help",
        "filter to keyword(s); repeat for multiple",
    )
    parser.add_argument("-k", "--keyword", **kwargs)


def add_jobs_arg(parser: Any) -> None:
    """Add ``-j`` / ``--jobs`` controlling PyTorch intra-op threads."""
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        metavar="N",
        help=_JOBS_HELP,
    )


def configure_torch_jobs(jobs: int) -> int:
    """Resolve ``0`` → CPU count and apply ``torch.set_num_threads``."""
    if jobs < 0:
        raise UsageError("--jobs must be >= 0 (0 = CPU count)")
    resolved = max(1, os.cpu_count() or 1) if jobs == 0 else jobs
    torch.set_num_threads(resolved)
    return resolved


def add_file_jobs_arg(parser: Any) -> None:
    """Add ``-J`` / ``--file-jobs`` for multi-file ThreadPool fan-out."""
    parser.add_argument(
        "-J",
        "--file-jobs",
        type=int,
        default=0,
        metavar="N",
        help=_FILE_JOBS_HELP,
    )


def resolve_file_jobs(jobs: int, n_paths: int) -> int:
    """Resolve ``-J``: serial for one path; ``0`` → CPU count when multi-file."""
    if jobs < 0:
        raise UsageError("--file-jobs must be >= 0 (0 = CPU count when ≥2 files)")
    if n_paths <= 1:
        return 1
    if jobs == 0:
        return max(1, os.cpu_count() or 1)
    return min(jobs, n_paths)


def run_file_jobs(
    items: list[T],
    fn: Callable[[T], R],
    jobs: int,
) -> list[R]:
    """Run ``fn`` over ``items`` serially or via a thread pool.

    When ``jobs > 1``, each worker caps ``torch.set_num_threads(1)`` so ATen
    does not oversubscribe beside CFITSIO I/O. Results keep input order.
    On the first worker failure, remaining futures are cancelled (best-effort).
    """
    if not items:
        return []
    workers = max(1, min(jobs, len(items)))
    if workers == 1:
        return [fn(item) for item in items]

    def _worker(item: T) -> R:
        torch.set_num_threads(1)
        return fn(item)

    ordered: dict[int, R] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, item): idx for idx, item in enumerate(items)}
        try:
            for fut in as_completed(futures):
                ordered[futures[fut]] = fut.result()
        except BaseException:
            for pending in futures:
                pending.cancel()
            raise
    return [ordered[i] for i in range(len(items))]


def ensure_unique_basenames(paths: list[str], *, label: str = "inputs") -> None:
    """Reject multi-file batches whose basenames would collide under ``--out-dir``."""
    seen: dict[str, str] = {}
    for path in paths:
        name = Path(cfitsio_base_path(path)).name
        prior = seen.get(name)
        if prior is not None:
            raise UsageError(
                f"duplicate basename under --out-dir for {label}: {name!r} "
                f"({prior} and {path})"
            )
        seen[name] = path


def ensure_unique_split_stems(paths: list[str], *, label: str = "inputs") -> None:
    """Reject ``--split hdu`` batches whose ``{stem}_hduNN`` outputs would collide."""
    seen: dict[str, str] = {}
    for path in paths:
        stem = Path(cfitsio_base_path(path)).stem
        prior = seen.get(stem)
        if prior is not None:
            raise UsageError(
                f"duplicate stem under --split hdu for {label}: {stem!r} "
                f"({prior} and {path})"
            )
        seen[stem] = path


def expand_at_list_paths(paths: list[str]) -> list[str]:
    """Expand classic ``@file`` path lists (one path per line, ``#`` comments)."""
    expanded: list[str] = []
    for item in paths:
        if not item.startswith("@") or len(item) == 1:
            expanded.append(item)
            continue
        list_path = item[1:]
        try:
            text = Path(list_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise IoError(f"cannot read path list {list_path}: {exc}") from exc
        for line in text.splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                expanded.append(entry)
    if not expanded:
        raise UsageError("no input paths after expanding @list files")
    return expanded


def resolve_batch_io_pairs(
    paths: list[str],
    *,
    out: str | None,
    out_dir: str | None,
    positional_output: str | None = None,
) -> list[tuple[str, str]]:
    """Resolve ``INPUT [OUTPUT]``, ``-o``, or multi-input ``--out-dir`` pairs."""
    if out and positional_output and out != positional_output:
        raise UsageError("conflicting output paths: use either -o/--out or positional")
    out_flag = out or positional_output

    if out_dir and out_flag:
        raise UsageError("use either --out-dir or -o/--out, not both")

    if out_dir:
        ensure_unique_basenames(paths)
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return [
            (path, str(directory / Path(cfitsio_base_path(path)).name))
            for path in paths
        ]

    if out_flag:
        if len(paths) != 1:
            raise UsageError("-o/--out requires exactly one input path")
        return [(paths[0], str(out_flag))]

    if len(paths) == 2:
        return [(paths[0], paths[1])]

    if len(paths) == 1:
        raise UsageError("output path required (-o/--out or positional OUTPUT)")

    raise UsageError("multiple inputs require --out-dir")


def add_split_arg(parser: Any) -> None:
    """Add ``--split {file,hdu}`` for per-file vs per-HDU outputs."""
    parser.add_argument(
        "--split",
        choices=_SPLIT_MODES,
        default="file",
        help="output granularity: one file per input (file) or per image HDU (hdu)",
    )


def add_emit_format_args(parser: Any) -> None:
    """Add ``-f`` / ``--format`` / ``--json`` / ``--jsonl`` to an inventory command."""
    parser.add_argument(
        "-f",
        "--format",
        choices=_EMIT_FORMATS,
        default=None,
        help="output format: text (default), json, or jsonl",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON array (alias)")
    parser.add_argument(
        "--jsonl", action="store_true", help="emit JSONL records (alias)"
    )


def resolve_emit_format(args: Any) -> str:
    """Resolve structured-output format from ``--format`` / ``--json`` / ``--jsonl``."""
    fmt = getattr(args, "format", None)
    json_flag = bool(getattr(args, "json", False))
    jsonl_flag = bool(getattr(args, "jsonl", False))
    if json_flag and jsonl_flag:
        raise UsageError("use only one of --json or --jsonl")
    if jsonl_flag:
        if fmt is not None and fmt != "jsonl":
            raise UsageError("conflicting --format and --jsonl")
        return "jsonl"
    if json_flag:
        if fmt is not None and fmt != "json":
            raise UsageError("conflicting --format and --json")
        return "json"
    return fmt or "text"


def emit_records(
    records: Iterable[dict[str, Any]],
    *,
    format: str = "text",
    stream: TextIO | None = None,
) -> None:
    """Emit inventory records as text, JSON, or JSONL."""
    out = stream or sys.stdout
    items = list(records)
    if format == "jsonl":
        for record in items:
            print(json.dumps(record, default=json_default), file=out)
        return
    if format == "json":
        print(json.dumps(items, default=json_default, indent=2), file=out)
        return
    for record in items:
        parts = [f"{key}={record[key]!r}" for key in sorted(record)]
        print(" ".join(parts), file=out)


def iter_file_hdu_pairs(
    paths: Iterable[str], hdu: str | None
) -> Iterator[tuple[str, int, Any]]:
    import torchfits

    for path in paths:
        if is_remote_path(path):
            raise IoError(f"remote paths are not supported: {path}")
        try:
            with torchfits.open(path) as hdul:
                for index in selected_hdu_indices(len(hdul), hdu):
                    yield path, index, hdul[index]
        except CliError:
            raise
        except Exception as exc:
            raise IoError(f"{path}: {exc}") from exc
