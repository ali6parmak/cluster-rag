"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Run from the repository root so relative paths in the config resolve. Commands
write their artifacts to ``results/<run_id>/`` (frozen resolved config plus a
JSON report), so ``run_id`` must be set in the config for any command.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from cluster_rag import __version__
from cluster_rag.config import AppConfig, load_config
from cluster_rag.data import SPLIT_NAMES, LoadReport, build_splits, load_jsonl
from cluster_rag.dataset_stats import build_dataset_stats, file_sha256

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
        help="Descriptive dataset statistics: buckets, co-occurrence, prefixes (Phase 1 EDA).",
        description="Load the raw dataset, validate it and write the full statistics report.",
    )
    stats.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file, e.g. configs/base.yaml.",
    )

    split = subparsers.add_parser(
        "split",
        help="Create the fixed train/val/test split and the label-holdout set (Phase 1).",
        description=(
            "Multi-label iterative stratification, deterministic coverage rebalance "
            "and label-holdout selection. Writes id lists and a manifest to "
            "data_splits; the result is reproducible for a fixed config and seed."
        ),
    )
    split.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file, e.g. configs/base.yaml.",
    )
    split.add_argument(
        "--force",
        action="store_true",
        help="Overwrite already existing split id files in data_splits.",
    )
    return parser


def _require_run_id(config: AppConfig) -> str:
    """Return the run_id from the config or raise, since artifacts need it."""
    if not config.run_id:
        raise ValueError(
            "run_id must be set in the config: commands write to results/<run_id>/ "
            "and the experiment log references the same run_id."
        )
    return config.run_id


def _write_resolved_config(results_dir: Path, config: AppConfig) -> None:
    """Freeze the resolved config into results/<run_id>/config.yaml."""
    payload = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    (results_dir / "config.yaml").write_text(payload, encoding="utf-8")


def _print_stats(report: LoadReport, dataset_stats: dict[str, Any]) -> None:
    """Print a human-readable summary of the dataset statistics to stdout."""
    overview = dataset_stats["overview"]
    print(f"Dataset: {report.path}")
    print(f"Records: {overview['n_records']} valid, {report.n_invalid} malformed line(s) dropped")
    print(f"Unique labels: {overview['n_unique_labels']}")
    languages = (
        overview["language_counts"]
        if overview["n_language_values"] <= 10
        else "more than 10 values"
    )
    print(f"Languages ({overview['n_language_values']}): {languages}")
    per_record = overview["labels_per_record"]
    mean_labels = per_record["mean"]
    print(
        f"Labels per record: mean {mean_labels:.2f} "
        f"(min {per_record['min']}, max {per_record['max']})"
    )
    lengths = overview["text_length_chars"]
    mean_len = lengths["mean"]
    print(f"Text length (chars): mean {mean_len:.1f} (min {lengths['min']}, max {lengths['max']})")

    print(f"Top {len(overview['top_labels'])} labels by frequency:")
    for label, count in overview["top_labels"]:
        print(f"  {count:6d}  {label}")

    buckets = dataset_stats["frequency_buckets"]
    thresholds = buckets["thresholds"]
    head_min, torso_min = thresholds["head_min"], thresholds["torso_min"]
    print(f"Frequency buckets (head_min={head_min}, torso_min={torso_min}):")
    for bucket in ("head", "torso", "tail"):
        print(f"  {bucket:5s}: {buckets[bucket]['n_labels']} labels")
    for bucket in ("head", "tail"):
        sample = buckets[bucket]["label_counts"]
        if sample:
            names = ", ".join(f"{label} ({count})" for label, count in sample[:3])
            print(f"    {bucket} examples: {names}")

    print("Top co-occurring label pairs:")
    for first, second, count in dataset_stats["cooccurrence_top"]:
        print(f"  {count:6d}  {first} + {second}")

    print("Most common leading prefixes:")
    for n_words, pairs in dataset_stats["prefixes"].items():
        shown = ", ".join(f"{prefix} ({count})" for prefix, count in pairs[:5])
        print(f"  {n_words}-word: {shown}")

    if report.errors:
        first = list(report.errors.items())[:5]
        logger.warning(
            "Dropped %d malformed line(s); first reasons: %s%s",
            report.n_invalid,
            first,
            " ..." if report.n_invalid > len(first) else "",
        )


def _cmd_stats(config: AppConfig) -> int:
    """Run the stats command: compute, print and persist the dataset report."""
    run_id = _require_run_id(config)
    report = load_jsonl(config.paths.data_raw)
    stats_config = config.stats
    dataset_stats = build_dataset_stats(
        report.records,
        head_min=stats_config.frequency_head_min,
        torso_min=stats_config.frequency_torso_min,
        prefix_max_words=stats_config.prefix_max_words,
        prefix_lowercase=stats_config.prefix_lowercase,
        top_n_labels=stats_config.top_n_labels,
        top_n_pairs=stats_config.top_n_pairs,
        top_n_prefixes=stats_config.top_n_prefixes,
    )
    payload: dict[str, Any] = {
        "run_id": run_id,
        "source": str(config.paths.data_raw),
        "source_sha256": file_sha256(config.paths.data_raw),
        "n_valid_records": report.n_valid,
        "n_malformed_lines": report.n_invalid,
        "malformed_lines": [[line, reason] for line, reason in report.errors.items()],
        "dataset_stats": dataset_stats,
    }
    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_stats(report, dataset_stats)
    print(f"Report written to {results_dir / 'stats.json'}")
    return 0


def _write_split_files(
    report: dict[str, Any], splits_dir: Path, run_id: str, source_path: Path
) -> None:
    """Write the split id lists, the holdout file and the manifest to data_splits."""
    provenance = {
        "run_id": run_id,
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
    }
    for name in SPLIT_NAMES:
        content = {"split": name, **provenance, "ids": report["ids"][name]}
        (splits_dir / f"{name}_ids.json").write_text(
            json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    holdout = {
        "labels": report["label_holdout"],
        "details": report["holdout_details"],
        **provenance,
    }
    (splits_dir / "label_holdout.json").write_text(
        json.dumps(holdout, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {**provenance, **report}
    (splits_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _cmd_split(config: AppConfig, force: bool) -> int:
    """Run the split command: build the fixed splits and persist all artifacts."""
    run_id = _require_run_id(config)
    splits_dir = config.paths.data_splits
    existing = [
        f"{name}_ids.json" for name in SPLIT_NAMES if (splits_dir / f"{name}_ids.json").exists()
    ]
    if existing and not force:
        raise ValueError(
            f"split files already exist ({', '.join(existing)}); "
            "rerun with --force to overwrite them deliberately"
        )

    report = load_jsonl(config.paths.data_raw)
    if report.errors:
        logger.warning("%d malformed line(s) dropped before splitting", report.n_invalid)
    manifest = build_splits(report.records, config.splits, config.seed)

    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_split_files(manifest, splits_dir, run_id, config.paths.data_raw)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "split_report.json").write_text(
        json.dumps(
            {**manifest, "run_id": run_id, "source_sha256": file_sha256(config.paths.data_raw)},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    sizes = manifest["sizes"]
    holdout = manifest["label_holdout"]
    print(f"Split written to {splits_dir} (run_id={run_id})")
    print(f"  sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}")
    print(f"  label holdout ({len(holdout)} labels): {', '.join(holdout)}")
    accidental = [
        label
        for label in manifest["labels_missing_from_train"]
        if label not in manifest["label_holdout"]
    ]
    print(f"  coverage moves between splits: {len(manifest['coverage_moves'])}")
    print(
        f"  labels absent from train: {len(manifest['labels_missing_from_train'])} "
        f"(holdout {len(manifest['label_holdout'])}, accidental {len(accidental)})"
    )
    print(f"Report written to {results_dir / 'split_report.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "stats":
            return _cmd_stats(config)
        if args.command == "split":
            return _cmd_split(config, force=args.force)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc.filename or exc)
        return 1
    except ValueError as exc:  # includes pydantic ValidationError and YAML errors
        logger.error("Invalid config or data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
