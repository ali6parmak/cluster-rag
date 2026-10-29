"""The ``stats``, ``stats --omct`` and ``split`` commands.

``stats`` writes the descriptive dataset statistics (Phase 1 EDA) and
``stats --omct`` the auxiliary pairwise data's; ``split`` writes the fixed
train/val/test split ids and the label-holdout set (Phase 1).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cluster_rag.commands._common import (
    _omct_run_id,
    _require_run_id,
    _warn_upr_prefix,
    _write_resolved_config,
)
from cluster_rag.config import AppConfig
from cluster_rag.data import SPLIT_NAMES, LoadReport, build_splits, load_jsonl
from cluster_rag.dataset_stats import build_dataset_stats, file_sha256
from cluster_rag.pairs import PairLoadReport, load_pairs, summarize_pairs

logger = logging.getLogger(__name__)


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


def _cmd_stats(config: AppConfig, omct: bool = False) -> int:
    """Run the stats command: compute, print and persist the dataset report."""
    if omct:
        return _cmd_stats_omct(config)
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
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


def _print_stats_omct(
    report: PairLoadReport, pair_stats: dict[str, Any], results_dir: Path
) -> None:
    """Print a human-readable summary of the pairwise dataset to stdout."""
    print(f"Pairwise dataset: {report.path}")
    print(
        f"Pairs: {pair_stats['n_pairs']} valid, {report.n_invalid} malformed entry/entries skipped"
    )
    positive = pair_stats["label_counts"]["1"]
    negative = pair_stats["label_counts"]["0"]
    share = pair_stats["positive_share"]
    print(f"Labels: {positive} positive / {negative} negative ({share:.1%} positive)")
    print(
        f"Distinct questions: {pair_stats['n_distinct_question_numbers']} numbers, "
        f"{pair_stats['n_distinct_question_texts']} texts"
    )
    print(f"Documents ({pair_stats['n_documents']}): {pair_stats['document_counts']}")
    print(f"Positive pairs by document: {pair_stats['positive_by_document']}")
    print(f"Duplicate pair ids: {pair_stats['n_duplicate_pair_ids']}")
    para = pair_stats["paragraph_text_chars"]
    quest = pair_stats["question_text_chars"]
    mean_para = para["mean"]
    mean_quest = quest["mean"]
    print(f"Paragraph length (chars): mean {mean_para:.1f} (min {para['min']}, max {para['max']})")
    print(
        f"Question length (chars): mean {mean_quest:.1f} (min {quest['min']}, max {quest['max']})"
    )
    print(f"Top-level metadata: {report.metadata}")
    if report.errors:
        first = list(report.errors.items())[:5]
        logger.warning(
            "Dropped %d malformed pair(s); first reasons: %s%s",
            report.n_invalid,
            first,
            " ..." if report.n_invalid > len(first) else "",
        )


def _cmd_stats_omct(config: AppConfig) -> int:
    """Run `stats --omct`: summarize the OMCT pairwise dataset (omct-prefixed run)."""
    run_id = _require_run_id(config)
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to summarize pairs")
    report = load_pairs(config.paths.data_external)
    pair_stats = summarize_pairs(report.records)
    prefixed_run_id = _omct_run_id(run_id)
    payload = {
        "run_id": prefixed_run_id,
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": report.n_valid,
        "n_malformed_pairs": report.n_invalid,
        "malformed_pairs": [[index, reason] for index, reason in report.errors.items()],
        "dataset_metadata": report.metadata,
        "pair_stats": pair_stats,
    }
    results_dir = config.paths.results / prefixed_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_stats_omct(report, pair_stats, results_dir)
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
    _warn_upr_prefix(run_id)
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
