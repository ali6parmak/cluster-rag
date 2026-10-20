"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Commands: stats (--pairs for the auxiliary pairwise data), split, embed
(Phase 2: dataset embedding through the disk cache), probe-aux (Phase 2: the
auxiliary pairwise embedding probe), evaluate (Phase 3: any predictor on a
fixed val/test split), sweep (Phases 2.5+: run a Python sweep file's resolved
cells). Run from the repository root so relative
paths in the config resolve. Commands write their artifacts to
``results/<run_id>/`` (frozen resolved config plus a JSON report), so
``run_id`` must be set in the config for any command. When the command's
dataset's comparison table is set (``paths.experiment_table_upr`` /
``paths.experiment_table_omct``), ``embed``, ``evaluate`` and ``probe-aux``
also append one row per run (per transform for probe-aux) to it, upserted by
run key — the section heading comes from the run's ``table.sweep_label`` when
the run came from a sweep, else ``## <UPR|OMCT> — <command>`` (Phase 3.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cluster_rag import __version__
from cluster_rag.aux_probe import ProbeCacheOptions, run_probe
from cluster_rag.config import AppConfig, load_config
from cluster_rag.data import (
    SPLIT_NAMES,
    LoadReport,
    build_label_matrix,
    build_splits,
    load_jsonl,
    load_split_ids,
    select_records_by_ids,
)
from cluster_rag.dataset_stats import build_dataset_stats, file_sha256
from cluster_rag.embed import KINDS, build_embedder
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.eval.buckets import assign_label_buckets, labels_by_bucket
from cluster_rag.eval.multilabel import apply_decision_rule
from cluster_rag.eval.report import build_label_report
from cluster_rag.experiment_table import (
    EMBED_COLUMNS,
    EMBED_KEY_COLUMNS,
    EVALUATE_COLUMNS,
    EVALUATE_KEY_COLUMNS,
    PROBE_AUX_COLUMNS,
    PROBE_AUX_GROUPS_COLUMNS,
    PROBE_AUX_GROUPS_KEY_COLUMNS,
    PROBE_AUX_KEY_COLUMNS,
    append_table_rows,
    intro_for,
    section_heading,
)
from cluster_rag.pairs import PairLoadReport, load_pairs, summarize_pairs
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
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
        help=(
            "Summarize the auxiliary pairwise dataset (PLAN section 2), not the UPR "
            "dataset. The run_id gets the 'omct-' dataset prefix (results/omct-<run_id>/)."
        ),
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
            "questions / same document), under the configured question transforms "
            "(default 'strip_block'; add 'keep' for an explicit comparison), with "
            "truncation diagnostics against the model context. Auxiliary evidence "
            "only: results are written to results/omct-<run_id>/ — the omct- dataset "
            "prefix is added by the command itself (AGENTS.md run-id grammar)."
        ),
    )
    probe_aux.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with embed: and paths.data_external set.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate any predictor on val or test (Phase 3 metrics; no embeddings needed).",
        description=(
            "Load the dataset and the fixed split ids (never re-split), build the "
            "configured predictor (eval.predictor; trivial baselines now, retrieval "
            "predictors in Phase 4) from train context only, score the eval split "
            "and report P@k / R@k / nDCG@k, micro/macro/sample-F1 at the configured "
            "decision rule, label coverage and shortlist recall@m — each per "
            "frequency bucket (head/torso/tail/unseen) and overall. Writes "
            "results/<run_id>/metrics.json. Exit criterion: one command evaluates "
            "any predictor on val or test."
        ),
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with an eval: section, e.g. configs/experiments/eval_*.yaml.",
    )
    evaluate.add_argument(
        "--split",
        choices=("val", "test"),
        default=None,
        help="Which fixed split to evaluate (default: eval.split in the config, itself 'val').",
    )

    sweep = subparsers.add_parser(
        "sweep",
        help="Run a Python sweep file: typed cells, each executed as its own run (Phases 2.5+).",
        description=(
            "Load a sweep file (a Python module exposing SWEEP: list[SweepRun], built "
            "with typed cells — EvalCell/EmbedCell/ProbeCell/SweepCell — and "
            "define_sweep(cells, dataset=..., label=...)) and execute its cells in "
            "order through the existing command functions (embed / evaluate / "
            "probe-aux / stats / stats-pairs / split): one run folder per cell with "
            "the frozen resolved config, exactly as single runs. Embed cells run "
            "with the default kinds ('document'). The sweep file only produces "
            "configs; the first failing cell aborts the sweep with its exit code."
        ),
    )
    sweep.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the sweep file, e.g. experiments/baseline_eval_sweep.py.",
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


UPR_PREFIX = "upr-"  # UPR dataset runs (AGENTS.md "Runs and where things live")
OMCT_PREFIX = "omct-"  # auxiliary OMCT pairwise-data runs


def _warn_upr_prefix(run_id: str) -> None:
    """Warn (never fail) when a UPR command's run_id lacks the `upr-` prefix.

    UPR commands accept any run_id so existing configs keep working; the
    warning keeps the `<dataset>-<command>-<detail>[-<split>]` grammar visible.
    """
    if not run_id.startswith(UPR_PREFIX):
        logger.warning(
            "run_id %r does not start with the UPR dataset prefix %r "
            "(expected grammar: <dataset>-<command>-<detail>[-<split>] per AGENTS.md); "
            "the run proceeds without it.",
            run_id,
            UPR_PREFIX,
        )


def _omct_run_id(run_id: str) -> str:
    """Return the run_id with the `omct-` dataset prefix added by the command.

    OMCT commands prefix the run id themselves (the config carries the bare
    detail), so a config that already carries `omct-`/`upr-` would double up —
    warned about, never fatal.
    """
    if run_id.startswith((OMCT_PREFIX, UPR_PREFIX)):
        logger.warning(
            "run_id %r already carries a dataset prefix; the command adds %r "
            "itself, so artifacts land under results/%s%s/ (off-grammar but not fatal).",
            run_id,
            OMCT_PREFIX,
            OMCT_PREFIX,
            run_id,
        )
    return f"{OMCT_PREFIX}{run_id}"


def _fmt(value: float | None, decimals: int) -> str | None:
    """Fixed-precision rendering for table cells (None passes through)."""
    if value is None:
        return None
    return f"{value:.{decimals}f}"


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
    """Run `stats --pairs`: summarize the auxiliary pairwise dataset (omct-prefixed run)."""
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
    _warn_upr_prefix(run_id)
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

    if config.paths.experiment_table_upr is not None:
        append_table_rows(
            config.paths.experiment_table_upr,
            heading=section_heading("upr", command="embed", sweep_label=config.table.sweep_label),
            columns=EMBED_COLUMNS,
            key_columns=EMBED_KEY_COLUMNS,
            intro=intro_for("upr"),
            rows=[
                {
                    "run_id": run_id,
                    "date": date.today().isoformat(),
                    "backend": info.backend,
                    "model": info.model_id,
                    "model_version": (info.revision_or_digest or "")[:12],
                    "dim": info.dimension,
                    "ctx": info.context_tokens,
                    "kinds": ",".join(kinds),
                    "preprocess": spec.variant,
                    "n_texts": len(ids),
                    "seconds": payload["total_seconds"],
                    "cache_hits": totals[0],
                    "cache_misses": totals[1],
                }
            ],
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
    """Run probe-aux: the Phase 2 auxiliary evidence run (results/omct-<run_id>/)."""
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
    omct_run_id = _omct_run_id(run_id)

    payload = {
        "run_id": omct_run_id,
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
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "probe_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if config.paths.experiment_table_omct is not None:
        variant_rows: list[dict[str, Any]] = []
        group_rows: list[dict[str, Any]] = []
        for variant in payload["probe"]["variants"]:
            same_doc = variant["retrieval"]["same_document"]
            share_over = variant["truncation"].get("share_paragraphs_over_context")
            variant_rows.append(
                {
                    "run_id": omct_run_id,
                    "date": date.today().isoformat(),
                    "backend": embedder_info.backend,
                    "model": embedder_info.model_id,
                    "model_version": (embedder_info.revision_or_digest or "")[:12],
                    "dim": embedder_info.dimension,
                    "ctx": embedder_info.context_tokens,
                    "transform": variant["transform"],
                    "pair_auc": _fmt(variant["pair_auc"], 4),
                    "same-doc hit@1": _fmt(same_doc.get("hit@1"), 3),
                    "same-doc hit@10": _fmt(same_doc.get("hit@10"), 3),
                    "same-doc recall@10": _fmt(same_doc.get("recall@10"), 3),
                    "paras>ctx": None if share_over is None else f"{share_over:.1%}",
                    "n_paras": variant["retrieval"]["all"]["n_rows"],
                    "n_pairs(pos)": (
                        f"{payload['probe']['corpus']['n_pairs']}"
                        f" ({payload['probe']['corpus']['n_positive_pairs']})"
                    ),
                    "seconds": variant["seconds"],
                }
            )
            for group in variant["group_breakdown"]["groups"]:
                scope = group["same_document"]
                group_rows.append(
                    {
                        "run_id": omct_run_id,
                        "transform": variant["transform"],
                        "document": group["document"],
                        "source_csv": group["source_csv"],
                        "n_pairs": group["n_pairs"],
                        "pos": group["n_positive"],
                        "neg": group["n_negative"],
                        "pair_auc": _fmt(group["pair_auc"], 4),
                        "same-doc hit@1": None if scope is None else _fmt(scope.get("hit@1"), 3),
                        "same-doc hit@10": None if scope is None else _fmt(scope.get("hit@10"), 3),
                        "n_rows_with_gold": group["n_rows_with_gold"],
                    }
                )
        heading_label = config.table.sweep_label
        append_table_rows(
            config.paths.experiment_table_omct,
            heading=section_heading("omct", command="probe-aux", sweep_label=heading_label),
            columns=PROBE_AUX_COLUMNS,
            key_columns=PROBE_AUX_KEY_COLUMNS,
            intro=intro_for("omct"),
            rows=variant_rows,
        )
        append_table_rows(
            config.paths.experiment_table_omct,
            heading=section_heading(
                "omct", command="probe-aux", sweep_label=heading_label, groups=True
            ),
            columns=PROBE_AUX_GROUPS_COLUMNS,
            key_columns=PROBE_AUX_GROUPS_KEY_COLUMNS,
            intro=intro_for("omct"),
            rows=group_rows,
        )

    print(f"Auxiliary probe: {omct_run_id} with {embedder_info.backend}:{embedder_info.model_id}")
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
        for group in variant["group_breakdown"]["groups"]:
            auc_text = "n/a" if group["pair_auc"] is None else f"{group['pair_auc']:.4f}"
            if group["same_document"] is None:
                same_doc_text = "no gold rows"
            else:
                scope = group["same_document"]
                same_doc_text = (
                    f"hit@1 {scope['hit@1']:.3f} / hit@10 {scope['hit@10']:.3f} "
                    f"over {group['n_rows_with_gold']} gold row(s)"
                )
            print(
                f"      {group['document']} (csv {group['source_csv']}): "
                f"pairs {group['n_pairs']} ({group['n_positive']} pos, "
                f"{group['n_negative']} neg), rows {group['n_rows']} | "
                f"pair-AUC {auc_text} | same-doc {same_doc_text}"
            )
    print(f"Report written to {results_dir / 'probe_report.json'}")
    return 0


def _cmd_evaluate(config: AppConfig, split: str | None = None) -> int:
    """Run the evaluate command: any predictor on a fixed split (Phase 3 exit criterion).

    Loads the dataset and the fixed split ids (with a provenance sha check, never
    re-splitting), constructs the configured predictor from **train** context only,
    scores the evaluation split and assembles the per-bucket report. Writes
    ``results/<run_id>/{config.yaml, metrics.json}`` and the auto table row.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.eval is None:
        raise ValueError(
            "the eval: section (predictor, decision) is required for the `evaluate` command"
        )
    eval_config = config.eval
    split_name = split or eval_config.split

    full_report = load_jsonl(config.paths.data_raw)
    if full_report.errors:
        logger.warning("%d malformed line(s) dropped before evaluating", full_report.n_invalid)
    source_sha = file_sha256(config.paths.data_raw)

    ids_by_split: dict[str, list[str]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    for name in (split_name, "train"):
        ids, provenance = load_split_ids(config.paths.data_splits, name)
        recorded = provenance.get("source_sha256")
        if recorded and recorded != source_sha:
            raise ValueError(
                f"{name}_ids.json was built from a different data source (sha256 mismatch): "
                "the fixed splits and the dataset disagree; do not re-split silently."
            )
        ids_by_split[name] = ids
        provenances[name] = provenance

    eval_records = select_records_by_ids(full_report.records, ids_by_split[split_name])
    train_records = select_records_by_ids(full_report.records, ids_by_split["train"])

    labels = sorted({label for record in full_report.records for label in record.labels})
    dataset_counts: Counter[str] = Counter()
    for record in full_report.records:
        dataset_counts.update(record.labels)
    train_counts: Counter[str] = Counter()
    for record in train_records:
        train_counts.update(record.labels)
    buckets = assign_label_buckets(
        dataset_counts,
        train_counts,
        head_min=config.stats.frequency_head_min,
        torso_min=config.stats.frequency_torso_min,
    )

    predictor = build_predictor(
        eval_config.predictor.name,
        eval_config.predictor.params,
        PredictorContext(labels=tuple(labels), train_records=tuple(train_records)),
        seed=config.seed,
    )
    texts = [record.text for record in eval_records]
    scores = predictor.score(texts)
    expected_shape = (len(eval_records), len(labels))
    if scores.shape != expected_shape:
        raise ValueError(
            f"predictor {eval_config.predictor.name!r} returned scores of shape {scores.shape}, "
            f"expected {expected_shape} (n_eval_records, n_labels)"
        )

    gold = build_label_matrix(eval_records, labels)
    decision = eval_config.decision
    predicted = apply_decision_rule(
        scores,
        rule=decision.rule,
        k=decision.k,
        threshold=decision.threshold,
        min_score=decision.min_score,
    )
    report = build_label_report(
        scores,
        gold,
        labels,
        buckets,
        predicted,
        ks=eval_config.metrics.ks,
        shortlist_ms=eval_config.metrics.shortlist_ms,
    )
    overall = report["overall"]

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": "evaluate",
        "split": split_name,
        "source": str(config.paths.data_raw),
        "source_sha256": source_sha,
        "split_provenance": provenances,
        "n_records": len(eval_records),
        "n_records_train_context": len(train_records),
        "n_labels": len(labels),
        "seed": config.seed,
        "predictor": predictor.describe(),
        "decision": decision.model_dump(),
        "metrics_config": eval_config.metrics.model_dump(),
        "buckets": {
            "thresholds": {
                "head_min": config.stats.frequency_head_min,
                "torso_min": config.stats.frequency_torso_min,
            },
            "n_labels_per_bucket": {
                name: len(subset) for name, subset in labels_by_bucket(buckets).items()
            },
            "labels_by_bucket": labels_by_bucket(buckets),
        },
        "report": report,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if config.paths.experiment_table_upr is not None:
        decision_cell = {
            "top_k": f"top_k k={decision.k}",
            "score_threshold": f"thr={decision.threshold}",
            "top_k_min_score": f"top_k k={decision.k} min={decision.min_score}",
        }[decision.rule]
        append_table_rows(
            config.paths.experiment_table_upr,
            heading=section_heading(
                "upr", command="evaluate", sweep_label=config.table.sweep_label
            ),
            columns=EVALUATE_COLUMNS,
            key_columns=EVALUATE_KEY_COLUMNS,
            intro=intro_for("upr"),
            rows=[
                {
                    "run_id": run_id,
                    "date": date.today().isoformat(),
                    "split": split_name,
                    "predictor": eval_config.predictor.name,
                    "f1 decision": decision_cell,
                    "micro-F1": _fmt(overall["f1"]["micro_f1"], 4),
                    "macro-F1": _fmt(overall["f1"]["macro_f1"], 4),
                    "sample-F1": _fmt(overall["f1"]["sample_f1"], 4),
                    "coverage": _fmt(overall["coverage"]["coverage"], 3),
                    "P@5": _fmt(overall["ranking"].get("precision@5"), 4),
                    "R@10": _fmt(overall["ranking"].get("recall@10"), 4),
                    "nDCG@5": _fmt(overall["ranking"].get("ndcg@5"), 4),
                    "shortlist R@50": _fmt(overall["shortlist"].get("recall@50"), 4),
                    "n_texts": len(eval_records),
                    "seconds": payload["total_seconds"],
                }
            ],
        )

    ranking = overall["ranking"]
    print(
        f"Evaluated {len(eval_records)} record(s) [{split_name}] with predictor "
        f"'{eval_config.predictor.name}' ({len(labels)} label columns)"
    )
    print(
        f"  decision: {decision.rule} (k={decision.k}, threshold={decision.threshold}, "
        f"min_score={decision.min_score})"
    )
    p5 = ranking.get("precision@5", 0.0)
    r10 = ranking.get("recall@10", 0.0)
    ndcg5 = ranking.get("ndcg@5", 0.0)
    shortlist50 = overall["shortlist"].get("recall@50", 0.0)
    print(
        f"  overall: P@5 {p5:.4f}, R@10 {r10:.4f}, nDCG@5 {ndcg5:.4f}, "
        f"micro-F1 {overall['f1']['micro_f1']:.4f}, macro-F1 {overall['f1']['macro_f1']:.4f}, "
        f"sample-F1 {overall['f1']['sample_f1']:.4f}, "
        f"coverage {overall['coverage']['coverage']:.3f}, shortlist R@50 {shortlist50:.4f}"
    )
    for bucket, section in report["per_bucket"].items():
        print(
            f"  {bucket:>6s} (labels {section['n_labels']}, with_gold "
            f"{section['n_labels_with_gold']}, rows_with_gold {section['n_rows_with_gold']}): "
            f"micro-F1 {section['f1']['micro_f1']:.4f}, macro-F1 {section['f1']['macro_f1']:.4f}, "
            f"coverage {section['coverage']['coverage']:.3f}"
        )
    print(f"Report written to {results_dir / 'metrics.json'}")
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


def _load_sweep_runs(path: Path) -> list[Any]:
    """Import a sweep file and return its resolved SWEEP list (validated).

    The file is a Python module executed once at load; its ``SWEEP`` must be a
    non-empty list of :class:`~cluster_rag.sweep.SweepRun` — i.e. the product
    of ``define_sweep(cells, dataset=..., label=...)`` over ``configs/base.yaml``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("cluster_rag_sweep_file", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load sweep file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import error aborts with its cause
        raise ValueError(f"loading sweep file {path} failed: {exc}") from exc
    sweep = getattr(module, "SWEEP", None)
    if not isinstance(sweep, list) or not sweep:
        raise ValueError(
            f"sweep file {path} must expose a non-empty SWEEP list "
            "(define_sweep(cells, dataset=..., label=...))"
        )
    from cluster_rag.sweep import SweepRun

    if not all(isinstance(run, SweepRun) for run in sweep):
        raise ValueError(
            f"sweep file {path}: SWEEP must contain SweepRun entries "
            "(from define_sweep), got "
            f"{[type(entry).__name__ for entry in sweep[:3]]}"
        )
    return sweep


def _cmd_sweep(path: Path) -> int:
    """Run the sweep command: execute each resolved cell as its own single run."""
    from cluster_rag.sweep import SweepRun

    runs = _load_sweep_runs(path)
    assert all(isinstance(run, SweepRun) for run in runs)  # typing only; _load_sweep_runs checks
    total = len(runs)
    started = time.perf_counter()
    print(f"Sweep {path}: {total} cell(s)")
    for index, run in enumerate(runs, start=1):
        print(f"  [{index}/{total}] {run.command} label={run.label} (run_id={run.config.run_id})")
        started_cell = time.perf_counter()
        if run.command == "embed":  # single kind; standalone runs choose with --kinds
            code = _cmd_embed(run.config, kinds=("document",))
        elif run.command == "evaluate":  # split comes from the cell's eval.split
            code = _cmd_evaluate(run.config)
        elif run.command == "stats":  # plain UPR dataset statistics
            code = _cmd_stats(run.config)
        elif run.command == "stats-pairs":  # the omct pairwise-data stats (stats --pairs)
            code = _cmd_stats(run.config, pairs=True)
        elif run.command == "split":  # sweep cells never re-split silently (--force only via CLI)
            code = _cmd_split(run.config, force=False)
        else:  # probe-aux
            code = _cmd_probe_aux(run.config)
        if code != 0:
            logger.error(
                "Sweep cell %d/%d (%s, label=%s) exited with code %d — aborting the sweep",
                index,
                total,
                run.command,
                run.label,
                code,
            )
            return code
        print(f"  [{index}/{total}] done in {time.perf_counter() - started_cell:.1f}s")
    print(f"Sweep finished: {total} run(s), {time.perf_counter() - started:.1f}s total")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "sweep":  # sweeps resolve their configs themselves
            return _cmd_sweep(args.file)
        config = load_config(args.config)
        if args.command == "stats":
            return _cmd_stats(config, pairs=args.pairs)
        if args.command == "split":
            return _cmd_split(config, force=args.force)
        if args.command == "embed":
            return _cmd_embed(config, kinds=_parse_kinds(args.kinds))
        if args.command == "evaluate":
            return _cmd_evaluate(config, split=args.split)
        if args.command == "probe-aux":
            return _cmd_probe_aux(config)
        if args.command == "sweep":
            return _cmd_sweep(args.file)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc.filename or exc)
        return 1
    except ValueError as exc:  # includes pydantic ValidationError and YAML errors
        logger.error("Invalid config or data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
