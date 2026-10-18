"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Commands: stats (--pairs for the auxiliary pairwise data), split, embed
(Phase 2: dataset embedding through the disk cache), probe-aux (Phase 2: the
auxiliary pairwise embedding probe). Run from the repository root so relative
paths in the config resolve. Commands write their artifacts to
``results/<run_id>/`` (frozen resolved config plus a JSON report), so
``run_id`` must be set in the config for any command.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cluster_rag import __version__
from cluster_rag.aux_probe import ProbeCacheOptions, run_probe
from cluster_rag.config import AppConfig, load_config
from cluster_rag.data import SPLIT_NAMES, LoadReport, build_splits, load_jsonl
from cluster_rag.dataset_stats import build_dataset_stats, file_sha256
from cluster_rag.embed import KINDS, build_embedder
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.pairs import PairLoadReport, load_pairs, summarize_pairs
from cluster_rag.preprocess import build_preprocess_spec

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
    stats.add_argument(
        "--pairs",
        action="store_true",
        help="Summarize the auxiliary pairwise dataset (PLAN section 2), not the UPR dataset.",
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

    embed = subparsers.add_parser(
        "embed",
        help="Embed texts with the configured backend, cached on disk (Phase 2).",
        description=(
            "Load the dataset, apply the configured preprocessing variant and "
            "embed every text with the configured backend (hf / ollama), through "
            "the disk cache keyed by (model_id, prefix, preprocessing_version, "
            "sha1(text)). Writes results/<run_id>/embeddings.npz plus meta.json."
        ),
    )
    embed.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file with an embed: section, "
        "e.g. configs/experiments/embed_e5_base_v2.yaml.",
    )
    embed.add_argument(
        "--kinds",
        type=str,
        default="document",
        help="Comma-separated encode kinds to embed for (default: document). "
        "One matrix per kind is written; the cache separates them by prefix.",
    )

    probe_aux = subparsers.add_parser(
        "probe-aux",
        help="Auxiliary pairwise-data probe: AUC + question retrieval per embedder (Phase 2).",
        description=(
            "Embed paragraphs and GTI questions from the auxiliary pairwise dataset "
            "with the configured backend and compute cosine-similarity AUC (judged "
            "0 vs 1) plus hit@k/recall@k question retrieval in two scopes (all "
            "questions / same document), for each configured question transform "
            "('keep' vs 'strip_block'), with truncation diagnostics against the "
            "model context. Auxiliary evidence only: results are written to "
            "results/aux-<run_id>/ and prefixed aux- in the experiment log."
        ),
    )
    probe_aux.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with embed: and paths.data_external set.",
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


def _cmd_stats(config: AppConfig, pairs: bool = False) -> int:
    """Run the stats command: compute, print and persist the dataset report."""
    if pairs:
        return _cmd_stats_pairs(config)
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


def _print_stats_pairs(
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


def _cmd_stats_pairs(config: AppConfig) -> int:
    """Run `stats --pairs`: summarize the auxiliary pairwise dataset (aux-prefixed run)."""
    run_id = _require_run_id(config)
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to summarize pairs")
    report = load_pairs(config.paths.data_external)
    pair_stats = summarize_pairs(report.records)
    aux_run_id = f"aux-{run_id}"
    payload = {
        "run_id": aux_run_id,
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": report.n_valid,
        "n_malformed_pairs": report.n_invalid,
        "malformed_pairs": [[index, reason] for index, reason in report.errors.items()],
        "dataset_metadata": report.metadata,
        "pair_stats": pair_stats,
    }
    results_dir = config.paths.results / aux_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "pairs_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _print_stats_pairs(report, pair_stats, results_dir)
    return 0


def _parse_kinds(raw: str) -> list[str]:
    """Parse and validate the --kinds flag ("document" or "document,query")."""
    kinds = [part.strip() for part in raw.split(",") if part.strip()]
    if not kinds:
        raise ValueError("--kinds must name at least one encode kind")
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"Unknown encode kind {kind!r}; expected one of {KINDS}")
    return kinds


def _cmd_embed(config: AppConfig, kinds: Sequence[str]) -> int:
    """Run the embed command: preprocess, embed through the disk cache, persist.

    Writes ``results/<run_id>/embeddings.npz`` (one ``vectors_<kind>`` matrix
    plus ``ids``) and ``meta.json`` (model/digest/revision, prefixes,
    preprocessing diagnostics, cache hit/miss statistics, timings). A rerun
    with the same config must report ``cache_hits == n_records``.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for the `embed` command"
        )
    report = load_jsonl(config.paths.data_raw)
    if report.errors:
        logger.warning("%d malformed line(s) dropped before embedding", report.n_invalid)
    ids = [record.id for record in report.records]
    raw_texts = [record.text for record in report.records]

    spec = build_preprocess_spec(config.preprocess)
    processed = spec.apply(raw_texts)
    preprocess_meta = spec.describe(raw_texts)

    embedder = build_embedder(config.embed)
    info = embedder.info
    if config.cache.enabled:
        embedder = CachedEmbedder(embedder, config.paths.cache, preprocessing_version=spec.version)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)

    arrays: dict[str, Any] = {"ids": np.asarray(ids)}
    kind_stats: dict[str, dict[str, Any]] = {}

    def _cache_counters() -> tuple[int, int, int]:
        return (
            getattr(embedder, "cache_hits", 0),
            getattr(embedder, "cache_misses", 0),
            getattr(embedder, "n_stored", 0),
        )

    for kind in kinds:
        started_kind = time.perf_counter()
        before = _cache_counters()
        vectors = embedder.encode(processed, kind)
        after = _cache_counters()
        kind_stats[kind] = {
            "n_vectors": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "l2_rows": float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) else 1.0,
            "cache_hits": after[0] - before[0],
            "cache_misses": after[1] - before[1],
            "n_stored": after[2] - before[2],
            "seconds": round(time.perf_counter() - started_kind, 3),
        }
        arrays[f"vectors_{kind}"] = vectors
    np.savez_compressed(results_dir / "embeddings.npz", **arrays)

    totals = _cache_counters()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": "embed",
        "kinds": list(kinds),
        "n_records": len(ids),
        "n_malformed_lines": report.n_invalid,
        "source_sha256": file_sha256(config.paths.data_raw),
        "embedder": {
            "backend": info.backend,
            "model_id": info.model_id,
            "dimension": info.dimension,
            "revision_or_digest": info.revision_or_digest,
            "context_tokens": info.context_tokens,
            "prefixes": dict(info.prefixes),
            "extra": info.extra,
        },
        "cache": {
            "enabled": config.cache.enabled,
            "path": str(config.paths.cache) if config.cache.enabled else None,
            "cache_hits": totals[0],
            "cache_misses": totals[1],
            "n_stored": totals[2],
        },
        "preprocess": preprocess_meta,
        "kind_stats": kind_stats,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Embedded {len(ids)} record(s) with {info.backend}:{info.model_id} -> {results_dir}")
    print(
        f"  kinds: {list(kinds)}, dimension: {info.dimension}, "
        f"preprocess: {spec.variant} ({preprocess_meta['n_changed']} text(s) changed)"
    )
    print(
        f"  cache: {totals[0]} hit(s), {totals[1]} miss(es), {totals[2]} stored, "
        f"{payload['total_seconds']}s total"
    )
    return 0


def _cmd_probe_aux(config: AppConfig) -> int:
    """Run probe-aux: the Phase 2 auxiliary evidence run (results/aux-<run_id>/)."""
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for probe-aux"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run the aux probe")

    pair_report = load_pairs(config.paths.data_external)
    spec = build_preprocess_spec(config.preprocess)
    embedder = build_embedder(config.embed)

    probe_report = run_probe(
        pair_report.records,
        embedder,
        preprocess_spec=spec,
        probe_config=config.probe,
        cache=ProbeCacheOptions(root=str(config.paths.cache)) if config.cache.enabled else None,
    )
    embedder_info = embedder.info
    aux_run_id = f"aux-{run_id}"

    payload = {
        "run_id": aux_run_id,
        "command": "probe-aux",
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": pair_report.n_valid,
        "n_malformed_pairs": pair_report.n_invalid,
        "embedder": probe_report["embedder"],
        "cache": (
            {"enabled": True, "path": str(config.paths.cache), **(probe_report["cache"] or {})}
            if config.cache.enabled
            else {"enabled": False}
        ),
        "preprocess": {
            "variant": spec.variant,
            "version": spec.version,
            "openers": list(spec.openers),
        },
        "probe": probe_report,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    results_dir = config.paths.results / aux_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "probe_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Auxiliary probe: {aux_run_id} with {embedder_info.backend}:{embedder_info.model_id}")
    print(
        f"  corpus: {pair_report.n_valid} judged pairs "
        f"({payload['probe']['corpus']['n_positive_pairs']} positive), "
        f"{payload['probe']['corpus']['n_distinct_paragraph_texts']} distinct paragraphs, "
        f"{payload['probe']['corpus']['n_distinct_question_texts']} distinct questions"
    )
    for variant in payload["probe"]["variants"]:
        same_doc_hit1 = variant["retrieval"]["same_document"]["hit@1"]
        trunc_share = variant["truncation"].get("share_paragraphs_over_context")
        trunc_text = "n/a" if trunc_share is None else f"{trunc_share:.1%}"
        print(
            f"  [{variant['transform']:>11s}] pair-AUC {variant['pair_auc']:.4f}; "
            f"hit@1 {variant['retrieval']['all']['hit@1']:.3f} (all) / "
            f"{same_doc_hit1:.3f} (same-doc); hit@10 "
            f"{variant['retrieval']['same_document']['hit@10']:.3f} (same-doc); "
            f"paragraphs over context: {trunc_text}"
        )
    print(f"Report written to {results_dir / 'probe_report.json'}")
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
            return _cmd_stats(config, pairs=args.pairs)
        if args.command == "split":
            return _cmd_split(config, force=args.force)
        if args.command == "embed":
            return _cmd_embed(config, kinds=_parse_kinds(args.kinds))
        if args.command == "probe-aux":
            return _cmd_probe_aux(config)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc.filename or exc)
        return 1
    except ValueError as exc:  # includes pydantic ValidationError and YAML errors
        logger.error("Invalid config or data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
