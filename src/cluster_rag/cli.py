"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Run from the repository root so relative paths in the config resolve.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from cluster_rag import __version__
from cluster_rag.config import load_config
from cluster_rag.data import LoadReport, load_jsonl, summarize_records

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per CLI command."""
    parser = argparse.ArgumentParser(
        prog="cluster-rag",
        description="Retrieval-augmented multi-label topic classification with feedback.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    stats = subparsers.add_parser(
        "stats",
        help="Preview basic dataset statistics (full EDA is a Phase 1 deliverable).",
        description="Load the raw dataset and print basic counts after validation.",
    )
    stats.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file, e.g. configs/base.yaml.",
    )
    return parser


def _print_stats(report: LoadReport) -> None:
    """Print the dataset summary and validation outcome to stdout."""
    summary = summarize_records(report.records)
    print(f"Dataset: {report.path}")
    print(f"Records: {summary['n_records']} valid, {report.n_invalid} malformed line(s) dropped")
    print(f"Unique labels: {summary['n_unique_labels']}")
    languages = (
        summary["language_counts"] if summary["n_language_values"] <= 10 else "more than 10 values"
    )
    print(f"Languages ({summary['n_language_values']}): {languages}")
    lpr = summary["labels_per_record"]
    print(f"Labels per record: mean {lpr['mean']:.2f} (min {lpr['min']}, max {lpr['max']})")
    lengths = summary["text_length_chars"]
    mean_len = lengths["mean"]
    print(f"Text length (chars): mean {mean_len:.1f} (min {lengths['min']}, max {lengths['max']})")
    print(f"Top {len(summary['top_labels'])} labels by frequency:")
    for label, count in summary["top_labels"]:
        print(f"  {count:6d}  {label}")
    if report.errors:
        first = list(report.errors.items())[:5]
        logger.warning(
            "Dropped %d malformed line(s); first reasons: %s%s",
            report.n_invalid,
            first,
            " ..." if report.n_invalid > len(first) else "",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "stats":
            config = load_config(args.config)
            report = load_jsonl(config.paths.data_raw)
            _print_stats(report)
            return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc.filename or exc)
        return 1
    except ValueError as exc:  # includes pydantic ValidationError and YAML errors
        logger.error("Invalid config or data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
