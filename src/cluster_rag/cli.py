"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Commands: stats (--omct for the OMCT pairwise data), split, embed
(Phase 2: dataset embedding through the disk cache), labelembed (Phase 4:
the zero-shot label-text artifact), probe-omct (Phase 2: the
OMCT pairwise embedding probe), analyze-omct (the text-processing track's
Stage-0 pair-score artifact), evaluate (Phase 3: any predictor on a
fixed val/test split), sweep (Phases 2.5+: run a Python sweep file's resolved
cells). Run from the repository root so relative
paths in the config resolve. Commands write their artifacts to
``results/<run_id>/`` (frozen resolved config plus a JSON report), so
``run_id`` must be set in the config for any command. When the command's
comparison table is set (``paths.experiment_table_upr`` /
``paths.experiment_table_omct``, or the text-processing track's
``paths.experiment_table_text_processing_omct`` / ``_upr``), ``embed``,
``labelembed``, ``evaluate``, ``probe-omct`` and ``analyze-omct`` also append
one row per run (per transform for probe-omct, per model for analyze-omct) to
it, upserted by run key — the file comes
from the command's (dataset, track) pair, the section heading from the run's
``table.sweep_label`` when the run came from a sweep, else
``## <UPR|OMCT> — <command>`` (Phase 3.5).
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
from cluster_rag.config import AnalysisConfig, AppConfig, ChunkerSpec, load_config
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
from cluster_rag.decision import CachedDecider, SystemOneDecider
from cluster_rag.embed import KINDS, build_embedder
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.embed.random import RandomEmbedder
from cluster_rag.embed_artifact import (
    EmbeddingArtifact,
    LabelEmbeddingArtifact,
    load_embedding_artifact,
    load_label_embedding_artifact,
)
from cluster_rag.eval.buckets import assign_label_buckets, labels_by_bucket
from cluster_rag.eval.multilabel import apply_decision_rule
from cluster_rag.eval.report import build_label_report
from cluster_rag.experiment_table import (
    ANALYZE_OMCT_ARMS_COLUMNS,
    ANALYZE_OMCT_ARMS_KEY_COLUMNS,
    ANALYZE_OMCT_COLUMNS,
    ANALYZE_OMCT_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_COLUMNS,
    ANALYZE_OMCT_STAGE3_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS,
    DECIDE_OMCT_COLUMNS,
    DECIDE_OMCT_KEY_COLUMNS,
    EMBED_COLUMNS,
    EMBED_KEY_COLUMNS,
    EVALUATE_COLUMNS,
    EVALUATE_KEY_COLUMNS,
    PROBE_OMCT_COLUMNS,
    PROBE_OMCT_GROUPS_COLUMNS,
    PROBE_OMCT_GROUPS_KEY_COLUMNS,
    PROBE_OMCT_KEY_COLUMNS,
    append_table_rows,
    intro_for,
    section_heading,
    table_path,
    table_target,
)
from cluster_rag.omct_analysis import (
    BASELINE_ARM,
    BASELINE_CHUNKER,
    DECISION_ARM,
    DECISION_RULE_FIXED_CUT,
    BaselineArm,
    ChunkedSide,
    ReportingSpec,
    build_analysis_report,
    build_dataset_view,
    build_stage3_grid,
    compare_arms,
    decision_rule_block,
    score_decision_arm,
    score_model_arms,
    score_stage3_model,
    shortlist_rerank_block,
    side_cache_versions,
)
from cluster_rag.omct_decision import (
    DecisionArm,
    run_decision_arm,
    score_decision_grid,
)
from cluster_rag.omct_probe import ProbeCacheOptions, run_probe
from cluster_rag.pairs import PairLoadReport, load_pairs, summarize_pairs
from cluster_rag.predictors import build_predictor, predictor_requirements
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    build_preprocess_spec,
    question_transform_version,
)
from cluster_rag.sweep import COMMANDS as SWEEP_COMMANDS
from cluster_rag.sweep import SHARED_EXPERIMENTS_DIR
from cluster_rag.text_processors.aggregation import parse_aggregations
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.combining import COMBINER_LOGISTIC, build_combiner
from cluster_rag.text_processors.features import (
    FeatureEmbedders,
    PairFeatureExtractor,
    ProfileUnits,
)

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
        "--omct",
        action="store_true",
        help=(
            "Summarize the OMCT pairwise dataset (PLAN section 2), not the UPR "
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
        help="Path to a YAML config with an embed: section (or sweep-resolved configs; "
        "experiment values live in experiments/*.py sweeps).",
    )
    embed.add_argument(
        "--kinds",
        type=str,
        default=None,
        help="Comma-separated encode kinds to embed for; overrides the config's "
        "embed_kinds (default there: document). One matrix per kind is written; "
        "the cache separates them by prefix.",
    )

    labelembed = subparsers.add_parser(
        "labelembed",
        help="Embed one text per dataset label for zero-shot scoring (Phase 4).",
        description=(
            "Build the zero-shot label-embedding artifact: render one text per "
            "label from the dataset's sorted unique labels (config-driven via "
            "the label_texts: section — variant 'name' or a template with a "
            "{label} placeholder), embed them under the model's card-verified "
            "label-kind prefix through the disk cache, and write "
            "results/<run_id>/label_embeddings.npz plus meta.json. Consumed by "
            "the 'zeroshot' predictor and by 'fusion' with c > 0 through "
            "eval.label_embedding_run_id. Run ids follow the grammar "
            "upr-labelembed-<model>[-<variant>]."
        ),
    )
    labelembed.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config with embed:, label_texts: (or sweep-resolved "
        "configs; experiment values live in experiments/*.py sweeps).",
    )

    probe_omct = subparsers.add_parser(
        "probe-omct",
        help="OMCT pairwise-data probe: AUC + question retrieval per embedder (Phase 2).",
        description=(
            "Embed paragraphs and GTI questions from the OMCT pairwise dataset "
            "with the configured backend and compute cosine-similarity AUC (judged "
            "0 vs 1) plus hit@k/recall@k question retrieval in two scopes (all "
            "questions / same document), under the configured question transforms "
            "(default 'strip_block'; add 'keep' for an explicit comparison), with "
            "truncation diagnostics against the model context. Auxiliary evidence "
            "only: results are written to results/omct-<run_id>/ — the omct- dataset "
            "prefix is added by the command itself (AGENTS.md run-id grammar)."
        ),
    )
    probe_omct.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with embed: and paths.data_external set.",
    )

    analyze_omct = subparsers.add_parser(
        "analyze-omct",
        help="Text-processing track: the Stage-0 pair-score artifact (analyze-omct).",
        description=(
            "The text-processing track's Stage-0 artifact "
            "(TEXT_PROCESSING_PLAN.md, Stage 0.1): score every judged OMCT pair "
            "and a seeded, proportional sample of the *unjudged* same-document "
            "cells (the control, subset B) with the configured models, and report "
            "the per-document strata, the score distributions per label and the "
            "negative-enrichment measurement (the rank percentile of the verified "
            "negatives next to the positives and the control). The control's "
            "provenance (seed, sampling rule, per-document quotas, the excluded "
            "judged cells) is recorded, so a later stage re-uses exactly the same "
            "cells. Writes results/omct-<run_id>/analysis.json — the omct- dataset "
            "prefix is added by the command itself (AGENTS.md run-id grammar); the "
            "row goes to the track's own table."
        ),
    )
    analyze_omct.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with an analysis: section (or sweep-resolved "
        "configs; experiment values live in experiments_text_processing/*.py sweeps).",
    )

    decide_omct = subparsers.add_parser(
        "decide-omct",
        help="Text-processing track: the decision-model probe (Stage 3.5).",
        description=(
            "The text-processing track's Stage 3.5 probe "
            "(TEXT_PROCESSING_PLAN.md, Stage 3.5.1): score the reviewed OMCT "
            "pairs (and the unjudged-proxy control) with the configured "
            "decision models — Ollama's decision-model type, answered with "
            "probabilities — and report the pair AUC, the within-document AUC, "
            "the rank percentiles, the control side and the calibration curve, "
            "beside the synthetic competence control that separates 'the "
            "reviewed set is at the ceiling' from 'the model is the wrong "
            "shape'. The model is pinned by id AND digest (the run fails on a "
            "mismatch) and every call is cached. Writes "
            "results/omct-<run_id>/decisions.json — the omct- dataset prefix is "
            "added by the command itself (AGENTS.md run-id grammar); the row "
            "goes to the track's own table."
        ),
    )
    decide_omct.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with an analysis.decision: section (or "
        "sweep-resolved configs; experiment values live in "
        "experiments_text_processing/*.py sweeps).",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate any predictor on val or test (Phase 3 metrics).",
        description=(
            "Load the dataset and the fixed split ids (never re-split), build the "
            "configured predictor (eval.predictor: trivial baselines, Phase 4 "
            "retrieval predictors) from train context only, score the eval split "
            "and report P@k / R@k / nDCG@k, micro/macro/sample-F1 at the configured "
            "decision rule, label coverage and shortlist recall@m — each per "
            "frequency bucket (head/torso/tail/unseen) and overall. Embedding-"
            "dependent predictors consume eval.embedding_run_id (a prior "
            "upr-embed-* run): its artifact is validated, id-aligned to the splits "
            "and recorded in the metrics lineage. Writes "
            "results/<run_id>/metrics.json. Exit criterion: one command evaluates "
            "any predictor on val or test."
        ),
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config with an eval: section (or sweep-resolved configs; "
        "experiment values live in experiments/*.py sweeps).",
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
            "with typed cells — EvalCell/EmbedCell/LabelEmbedCell/ProbeCell/SweepCell — "
            "and define_sweep(cells, dataset=..., label=...)) and execute its cells in "
            "order through the existing command functions (embed / labelembed / "
            "evaluate / probe-omct / stats / stats-omct / split): one run folder per "
            "cell with the frozen resolved config, exactly as single runs. Embed "
            "cells run with the default kinds ('document'). The sweep file only "
            "produces configs; the first failing cell aborts the sweep with its "
            "exit code."
        ),
    )
    sweep.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the sweep file, e.g. experiments/baseline_eval_sweep.py "
        "(the text-processing track's sweeps live in experiments_text_processing/).",
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


def _append_command_rows(
    config: AppConfig,
    command: str,
    *,
    columns: Sequence[str],
    key_columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
    groups: bool = False,
) -> None:
    """Append one command's rows to its comparison table (command-derived).

    The command decides the ``(dataset, track)`` pair and therefore the file:
    the main pipeline's commands write to the per-dataset tables, the
    text-processing track's commands to the track's own pair of tables
    (AGENTS.md "Runs and where things live"). An unset path skips the append
    (throwaway configs in tests).
    """
    path = table_path(config.paths, command)
    if path is None:
        return
    target = table_target(command)
    append_table_rows(
        path,
        heading=section_heading(
            target.dataset,
            command=command,
            sweep_label=config.table.sweep_label,
            groups=groups,
        ),
        columns=columns,
        key_columns=key_columns,
        intro=intro_for(target.dataset, target.track),
        rows=rows,
    )


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

    _append_command_rows(
        config,
        "embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
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


def _cmd_labelembed(config: AppConfig) -> int:
    """Run labelembed: embed one text per dataset label (zero-shot artifact, Phase 4).

    Labels are the dataset's sorted unique labels (no catalog file, PLAN.md
    section 7); the per-label texts come from the config's ``label_texts:``
    section (variant ``name`` or a ``{label}`` template). Rendered texts go
    through the same preprocessing spec and disk cache as dataset texts, under
    the model's card-verified ``label`` prefix. Writes
    ``results/<run_id>/label_embeddings.npz`` (``labels`` + ``vectors_label``,
    row order = sorted label order) plus ``meta.json`` with the full lineage;
    consumed through ``eval.label_embedding_run_id``.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for "
            "the `labelembed` command"
        )
    if config.label_texts is None:
        raise ValueError(
            "the label_texts: section (variant 'name' or a {label} template) is "
            "required for the `labelembed` command"
        )
    report = load_jsonl(config.paths.data_raw)
    if report.errors:
        logger.warning("%d malformed line(s) dropped before label embedding", report.n_invalid)
    labels = sorted({label for record in report.records for label in record.labels})
    if not labels:
        raise ValueError("the dataset has no labels; nothing to embed")
    texts = [config.label_texts.render_text(label) for label in labels]

    spec = build_preprocess_spec(config.preprocess)
    processed = spec.apply(texts)
    preprocess_meta = spec.describe(texts)

    embedder = build_embedder(config.embed)
    info = embedder.info
    if config.cache.enabled:
        embedder = CachedEmbedder(embedder, config.paths.cache, preprocessing_version=spec.version)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)

    def _cache_counters() -> tuple[int, int, int]:
        return (
            getattr(embedder, "cache_hits", 0),
            getattr(embedder, "cache_misses", 0),
            getattr(embedder, "n_stored", 0),
        )

    vectors = embedder.encode(processed, "label")
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(labels)
        or (info.dimension is not None and vectors.shape[1] != info.dimension)
    ):
        raise ValueError(
            f"label embedding produced shape {vectors.shape}, expected "
            f"({len(labels)}, {info.dimension}); refusing to write the artifact"
        )
    np.savez_compressed(
        results_dir / "label_embeddings.npz",
        labels=np.asarray(labels),
        vectors_label=vectors,
    )

    totals = _cache_counters()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": "labelembed",
        "n_labels": len(labels),
        "label_texts": {
            "variant": config.label_texts.variant,
            "template": config.label_texts.template,
            "label_space": "sorted unique labels of the dataset (no catalog file)",
            "rendered_example": texts[0],
        },
        "source_sha256": file_sha256(config.paths.data_raw),
        "n_malformed_lines": report.n_invalid,
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
        "label_stats": {
            "dimension": int(vectors.shape[1]),
            "l2_rows": float(np.linalg.norm(vectors, axis=1).mean()) if len(vectors) else 1.0,
        },
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _append_command_rows(
        config,
        "labelembed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "backend": info.backend,
                "model": info.model_id,
                "model_version": (info.revision_or_digest or "")[:12],
                "dim": info.dimension,
                "ctx": info.context_tokens,
                "kinds": "label",
                "preprocess": spec.variant,
                "n_texts": len(labels),
                "seconds": payload["total_seconds"],
                "cache_hits": totals[0],
                "cache_misses": totals[1],
            }
        ],
    )

    print(
        f"Embedded {len(labels)} label text(s) with {info.backend}:{info.model_id} -> {results_dir}"
    )
    print(
        f"  label_texts: {config.label_texts.variant}"
        + (f" ({config.label_texts.template})" if config.label_texts.template else "")
        + f", dimension: {info.dimension}, preprocess: {spec.variant}"
    )
    print(
        f"  cache: {totals[0]} hit(s), {totals[1]} miss(es), {totals[2]} stored, "
        f"{payload['total_seconds']}s total"
    )
    return 0


def _cmd_probe_omct(config: AppConfig) -> int:
    """Run probe-omct: the Phase 2 auxiliary evidence run (results/omct-<run_id>/)."""
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for probe-omct"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run the omct probe")

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
        "command": "probe-omct",
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
    _append_command_rows(
        config,
        "probe-omct",
        columns=PROBE_OMCT_COLUMNS,
        key_columns=PROBE_OMCT_KEY_COLUMNS,
        rows=variant_rows,
    )
    _append_command_rows(
        config,
        "probe-omct",
        columns=PROBE_OMCT_GROUPS_COLUMNS,
        key_columns=PROBE_OMCT_GROUPS_KEY_COLUMNS,
        rows=group_rows,
        groups=True,
    )

    print(f"OMCT probe: {omct_run_id} with {embedder_info.backend}:{embedder_info.model_id}")
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


def _within_ci(comparison: dict[str, Any] | None) -> str | None:
    """The CI of the within-document AUC difference (the transferable part)."""
    if not comparison:
        return None
    paired = comparison.get("auc_decomposition", {}).get("within_group_paired")
    if not paired:
        return None
    return f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]"


def _analyze_omct_row(
    model: dict[str, Any],
    *,
    run_id: str,
    arm_name: str,
    arm: dict[str, Any],
    comparison: dict[str, Any] | None,
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the analysis: the model, the arm and its numbers."""
    info = model["embedder"]
    distributions = arm["score_distributions"]
    enrichment = arm["negative_enrichment"]["rank_percentile"]
    paired = comparison["paired_auc"] if comparison else None
    paragraph_side = arm["sides"]["paragraph"]
    n_texts = paragraph_side.get("n_texts") or 0
    n_chunks = paragraph_side.get("n_chunks") or 0
    row: dict[str, Any] = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info["backend"],
        "model": info["model_id"],
        "model_version": (info["revision_or_digest"] or "")[:12],
        "dim": info["dimension"],
        "ctx": info["context_tokens"],
        "arm": arm_name,
        "chunker": paragraph_side.get("chunker_slug") or paragraph_side.get("chunker_name"),
        "aggregation": (arm.get("aggregation") or {}).get("spec"),
        "transform": arm["sides"]["question"].get("transform"),
        "pair_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "macro_auc": _fmt(arm["pair_auc"].get("macro"), 4),
        "within_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("within_group_auc"), 4),
        "cross_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("cross_group_auc"), 4),
        "within_d_auc": _fmt(
            (comparison or {}).get("auc_decomposition", {}).get("within_group_delta"), 4
        ),
        "within_d_auc_ci": _within_ci(comparison),
        "d_auc": _fmt(paired["delta_auc"], 4) if paired else None,
        "d_auc_ci": (f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None),
        "deshift_auc": (_fmt(comparison["deshifted_auc"]["overall"], 4) if comparison else None),
        "floor_gap_delta": (_fmt(comparison["floor_gap"]["delta"], 4) if comparison else None),
        "ctrl_mean": _fmt(distributions["control"].get("mean"), 4),
        "ctrl_mean_delta": (
            _fmt(comparison["control_shift"]["mean_delta"], 4) if comparison else None
        ),
        "ctrl_p50": _fmt(distributions["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50": _fmt(enrichment["control"].get("quantiles", {}).get("0.5"), 4),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "chunks_per_text": (round(n_chunks / n_texts, 2) if n_texts else None),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    return row


def _cached_embedder(embedder: Any, config: AppConfig, version: str) -> Any:
    """The embedder of one side, cached under that side's version slot."""
    if not config.cache.enabled:
        return embedder
    return CachedEmbedder(embedder, config.paths.cache, preprocessing_version=version)


def _chunker_with_embedder(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
    *,
    threshold: float | None = None,
    base_version: str | None = None,
    cache: bool = True,
) -> tuple[Any, Any]:
    """Build a chunker and the embedder of its chunks.

    The semantic chunker finds its boundaries from the **atom** embeddings, so
    it takes the ``Embedder`` port: the CLI wires the cached backend in here
    (the package itself never imports a model client). The atoms go through
    the atoms chunker's cache version, the chunks through the chunker's own
    version — which carries the swept threshold, so two thresholds never
    share cached chunks.

    ``base_version`` overrides the version slot the chunks are cached under
    (the paragraph side's preprocessing version by default; the question side
    uses the question transform's version).

    ``cache=False`` returns the embedder as it is: the Stage-3 random-vector
    floor passes its own (uncached) embedder here, so the random vectors never
    enter the model's embedding cache.
    """
    params = dict(chunker_spec.params)
    if threshold is not None:
        params["threshold"] = threshold

    def _side(version: str) -> Any:
        """One side's embedder: the cached backend, or the embedder as it is."""
        return _cached_embedder(embedder, config, version) if cache else embedder

    if chunker_spec.name == "semantic":
        atom_params = {"split_clauses": bool(params.get("atom_split_clauses", False))}
        atom_chunker = build_chunker("atoms", atom_params)
        atom_version, _ = side_cache_versions(
            atom_chunker, spec, question_transform, question_block_markers
        )
        chunker = build_chunker(
            "semantic",
            params,
            embedder=_side(atom_version),
        )
    else:
        chunker = build_chunker(chunker_spec.name, params)
    if base_version is None:
        paragraph_version, _ = side_cache_versions(
            chunker, spec, question_transform, question_block_markers
        )
    else:
        paragraph_version = chunker.cache_version(base_version)
    return chunker, _side(paragraph_version)


def _chunked_side(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    question_chunker_spec: ChunkerSpec | None,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
) -> ChunkedSide:
    """The chunked paragraph side and, for the symmetric variants, the question side."""
    chunker, paragraph_embedder = _chunker_with_embedder(
        config, embedder, chunker_spec, spec, question_transform, question_block_markers
    )
    question_chunker = None
    question_embedder = None
    if question_chunker_spec is not None:
        question_chunker, question_embedder = _chunker_with_embedder(
            config,
            embedder,
            question_chunker_spec,
            spec,
            question_transform,
            question_block_markers,
        )
    return ChunkedSide(
        chunker=chunker,
        paragraph_embedder=paragraph_embedder,
        question_chunker=question_chunker,
        question_embedder=question_embedder,
    )


def _swept_sides(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
    thresholds: Sequence[float],
) -> list[ChunkedSide]:
    """One chunked side per point of the swept grid (the semantic threshold)."""
    sides: list[ChunkedSide] = []
    for threshold in thresholds:
        chunker, paragraph_embedder = _chunker_with_embedder(
            config,
            embedder,
            chunker_spec,
            spec,
            question_transform,
            question_block_markers,
            threshold=threshold,
        )
        sides.append(
            ChunkedSide(
                chunker=chunker,
                paragraph_embedder=paragraph_embedder,
                threshold=threshold,
            )
        )
    return sides


def _stage3_extractor(
    config: AppConfig,
    embedder: Any,
    analysis: AnalysisConfig,
    spec: PreprocessSpec,
    view: Any,
    *,
    paragraph_embedder: Any,
    question_embedder: Any,
    feature_embedder: Any | None = None,
) -> PairFeatureExtractor:
    """The Stage-3 feature pack, with the cached backends wired in.

    The pack takes the ``Embedder`` **port** (the same precedent as the
    semantic chunker): the package never imports a model client, the CLI
    wires the cached backends in here. Every side has its own cache-version
    slot, so they are separate ports: the whole paragraph (the baseline arm's
    paragraph slot), the whole question (the baseline arm's question slot),
    the chunks (the chunker's slot) and the Stage-3.3 chunk profile's four
    unit sides (the sentences and the word pieces, on each side).

    ``feature_embedder`` is the **Stage-3 random-vector floor** (CS-6d): when
    it is given, every embedding side of the pack is that embedder instead of
    the model's (cached) backend, so the floor replaces the embedding
    geometry and nothing else. It is never cached, so the random vectors never
    enter the model's embedding cache.
    """
    features = analysis.stage3.features
    random_mode = feature_embedder is not None
    paragraph_chunker, chunk_embedder = _chunker_with_embedder(
        config,
        feature_embedder if random_mode else embedder,
        features.paragraph_chunker,
        spec,
        analysis.question_transform,
        analysis.question_block_markers,
        cache=not random_mode,
    )
    question_chunks_embedder = None
    if features.question_chunker is not None:
        question_version = question_transform_version(
            analysis.question_transform, analysis.question_block_markers
        )
        _, question_chunks_embedder = _chunker_with_embedder(
            config,
            feature_embedder if random_mode else embedder,
            features.question_chunker,
            spec,
            analysis.question_transform,
            analysis.question_block_markers,
            base_version=question_version,
            cache=not random_mode,
        )
    # The chunk profile's units (the sentences and the word pieces). The
    # paragraph side is cached under the preprocessing version, the question
    # side under the question transform's version (the profile's question
    # units are cut from the transformed question).
    question_version = question_transform_version(
        analysis.question_transform, analysis.question_block_markers
    )
    sentence_chunker = build_chunker("units", {"kind": "sentence"})
    ngram_chunker = build_chunker(
        "units", {"kind": "ngram", "n_words": features.profile_ngram_size}
    )
    profile_units = ProfileUnits(sentences=sentence_chunker, ngrams=ngram_chunker)
    question_embedding_texts = dict(
        zip(
            view.questions,
            apply_question_transform(
                list(view.questions),
                analysis.question_transform,
                analysis.question_block_markers,
            ),
            strict=True,
        )
    )

    def _side(version: str) -> Any:
        """One side's embedder: the cached model backend, or the random one."""
        if random_mode:
            return feature_embedder
        return _cached_embedder(embedder, config, version)

    return PairFeatureExtractor(
        FeatureEmbedders(
            paragraph=feature_embedder if random_mode else paragraph_embedder,
            question=feature_embedder if random_mode else question_embedder,
            chunks=chunk_embedder,
            question_chunks=question_chunks_embedder,
            paragraph_sentences=_side(sentence_chunker.cache_version(spec.version)),
            question_sentences=_side(sentence_chunker.cache_version(question_version)),
            paragraph_ngrams=_side(ngram_chunker.cache_version(spec.version)),
            question_ngrams=_side(ngram_chunker.cache_version(question_version)),
        ),
        paragraph_chunker=paragraph_chunker,
        top_k=features.top_k,
        question_embedding_texts=question_embedding_texts,
        groups=features.groups,
        profile_units=profile_units,
        profile_top_k=features.profile_top_k,
    )


def _stage3_reporting_table_values(stage3: dict[str, Any]) -> dict[str, Any]:
    """The reporting's table cells: the calibration, the threshold and the rule.

    The rule's cells are the **two-part** variant (the cosine's margin *and*
    the arm's margin below their paragraph's best candidate — CS-6d's
    "retrieve with the cosine, rank with the learned arm") and the
    control-role floor's harm at the same reach: the rule's own numbers
    against the floor, in the rule's own units.
    """
    reporting = stage3.get("reporting") or {}
    calibration = reporting.get("calibration") or {}
    arm_calibration = calibration.get("arm") or {}
    cosine_calibration = calibration.get("cosine") or {}
    thresholds = reporting.get("thresholds") or {}
    per_document = thresholds.get("per_document") or {}
    recalls = [
        ((block.get("holdout") or {}).get("operating_point") or {}).get("recall")
        for block in per_document.values()
    ]
    usable = [value for value in recalls if value is not None]
    macro_recall = sum(usable) / len(usable) if usable else None
    out_of_fold = thresholds.get("out_of_fold") or {}
    median_operating = out_of_fold.get("operating_point_at_median") or {}
    variants = (reporting.get("safe_negative") or {}).get("variants") or {}
    rule = variants.get("both_margins") or {}
    pooled = rule.get("pooled") or {}
    floor = rule.get("floor_arm") or {}
    matched = floor.get("harm_at_matched_coverage") or {}
    return {
        "ece_arm": _fmt((arm_calibration.get("pooled") or {}).get("ece"), 4),
        "ece_arm_macro": _fmt(arm_calibration.get("macro_ece"), 4),
        "ece_cos": _fmt((cosine_calibration.get("pooled") or {}).get("ece"), 4),
        "brier_arm": _fmt((arm_calibration.get("pooled") or {}).get("brier"), 4),
        "thr_criterion": thresholds.get("criterion"),
        "thr_macro_recall": _fmt(macro_recall, 4),
        "thr_oof_recall": _fmt(median_operating.get("recall"), 4),
        "rule_cov": _fmt(pooled.get("coverage"), 4),
        "rule_harm": _fmt(pooled.get("harm"), 4),
        "rule_floor_harm": _fmt(matched.get("harm"), 4),
    }


def _stage3_table_row(
    model: dict[str, Any],
    *,
    run_id: str,
    stage3: dict[str, Any],
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the Stage-3 arm: the model, the arm and its numbers."""
    info = model["embedder"]
    comparison = stage3["comparison"]
    decomposition = comparison["auc_decomposition"]
    paired = comparison["paired_auc"]
    within_paired = decomposition.get("within_group_paired") or {}
    control = comparison["control_side"]["arm"]
    enrichment = comparison["negative_enrichment"]["rank_percentile"]
    feature_pack = stage3["feature_pack"]
    floors = stage3["permutation_floor"]
    hit_at_k = comparison["hit_at_k"]
    row = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info["backend"],
        "model": info["model_id"],
        "model_version": (info["revision_or_digest"] or "")[:12],
        "dim": info["dimension"],
        "ctx": info["context_tokens"],
        "groups": "+".join(feature_pack.get("groups") or []) or None,
        "features": feature_pack.get("n_features"),
        "combiner": (stage3.get("combiner") or {}).get("name"),
        "within_auc": _fmt(decomposition["arm"].get("within_group_auc"), 4),
        "within_d_auc": _fmt(decomposition.get("within_group_delta"), 4),
        "within_d_auc_ci": (
            f"[{within_paired['ci_low']:+.4f}, {within_paired['ci_high']:+.4f}]"
            if within_paired
            else None
        ),
        "pooled_auc": _fmt(comparison["pair_auc"].get("arm"), 4),
        "pooled_d_auc": _fmt(paired.get("delta_auc"), 4),
        "pooled_d_auc_ci": (
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None
        ),
        "macro_auc": _fmt(comparison["pair_auc"].get("macro", {}).get("arm"), 4),
        "perm_auc_within": _fmt(floors.get("within_document", {}).get("chance"), 4),
        "ctrl_role_auc_within": _fmt(floors.get("within_document", {}).get("control_role"), 4),
        "ctrl_mean": _fmt(control.get("mean"), 4),
        "ctrl_rank_p50": _fmt(control.get("rank_p50"), 4),
        "ctrl_rank_p50_delta": _fmt(comparison["control_side"].get("rank_p50_delta"), 4),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "floor_gap": _fmt(comparison["floor_gap"]["arm"].get("standardized"), 4),
        "hit1_pos": _fmt(hit_at_k["arm"]["label1"].get("hit@1"), 4),
        "hit1_pos_ref": _fmt(hit_at_k["reference"]["label1"].get("hit@1"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    if stage3.get("reporting"):
        row.update(_stage3_reporting_table_values(stage3))
    return row


def _decision_arm_row(
    model: dict[str, Any],
    *,
    run_id: str,
    arm_name: str,
    arm: dict[str, Any],
    comparison: dict[str, Any] | None,
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the decision arm (CS-7b).

    The row fills both table shapes' keys (the arms shape and the Stage-3
    shape), so the decision arm's numbers show up whichever shape the run's
    section uses. The Stage-3-only keys (the feature groups, the combiner, the
    permutation floors) are **n/a** for a decision arm: it is not a learned
    arm, it is a frozen zero-shot scorer.
    """
    info = model["model"]
    distributions = arm["score_distributions"]
    enrichment = arm["negative_enrichment"]["rank_percentile"]
    paired = comparison["paired_auc"] if comparison else None
    decomposition = comparison.get("auc_decomposition", {}) if comparison else {}
    within_paired = decomposition.get("within_group_paired") or {}
    hit_at_k = comparison["hit_at_k"] if comparison else {}
    rule = (model.get("rule") or {}).get("pooled") or {}
    fixed = rule.get(DECISION_RULE_FIXED_CUT) or {}
    counterpart = fixed.get("counterpart") or {}
    matched = counterpart.get("harm_at_matched_coverage") or {}
    row: dict[str, Any] = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info.get("backend"),
        "model": info.get("model_id"),
        "model_version": (info.get("digest") or "")[:12],
        "dim": None,
        "ctx": None,
        "arm": arm_name,
        "chunker": None,
        "aggregation": None,
        "transform": "strip_block",
        "groups": None,
        "features": None,
        "combiner": None,
        "pair_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "pooled_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "pooled_d_auc": _fmt(paired.get("delta_auc"), 4) if paired else None,
        "pooled_d_auc_ci": (
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None
        ),
        "macro_auc": _fmt(arm["pair_auc"].get("macro"), 4),
        "within_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("within_group_auc"), 4),
        "cross_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("cross_group_auc"), 4),
        "within_d_auc": _fmt(decomposition.get("within_group_delta"), 4),
        "within_d_auc_ci": (
            f"[{within_paired['ci_low']:+.4f}, {within_paired['ci_high']:+.4f}]"
            if within_paired
            else None
        ),
        "d_auc": _fmt(paired.get("delta_auc"), 4) if paired else None,
        "d_auc_ci": (f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None),
        "deshift_auc": (_fmt(comparison["deshifted_auc"]["overall"], 4) if comparison else None),
        "floor_gap_delta": (_fmt(comparison["floor_gap"]["delta"], 4) if comparison else None),
        "floor_gap": _fmt((comparison or {}).get("floor_gap", {}).get("arm"), 4),
        "ctrl_mean": _fmt(distributions["control"].get("mean"), 4),
        "ctrl_mean_delta": (
            _fmt(comparison["control_shift"]["mean_delta"], 4) if comparison else None
        ),
        "ctrl_p50": _fmt(distributions["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50": _fmt(enrichment["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50_delta": (
            _fmt(comparison["control_shift"].get("rank_p50_delta"), 4) if comparison else None
        ),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "chunks_per_text": None,
        "perm_auc_within": None,
        "ctrl_role_auc_within": None,
        "hit1_pos": _fmt((hit_at_k.get("arm") or {}).get("label1", {}).get("hit@1"), 4),
        "hit1_pos_ref": _fmt((hit_at_k.get("reference") or {}).get("label1", {}).get("hit@1"), 4),
        # the decision rule: the fixed cut's reach and harm, and the
        # label-blind counterpart's harm at the same reach
        "rule_cov": _fmt(fixed.get("coverage"), 4),
        "rule_harm": _fmt(fixed.get("harm"), 4),
        "rule_floor_harm": _fmt(matched.get("harm"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    return row


def _cmd_analyze_omct(config: AppConfig) -> int:
    """Run analyze-omct: the text-processing track's Stage-0 artifact.

    Writes ``results/omct-<run_id>/{config.yaml, analysis.json}``: the
    per-pair scores of the judged pairs, the control set (the sampled
    unjudged same-document cells) with their scores and their provenance, the
    per-document strata with their base rates, the score distributions per
    label and the negative-enrichment measurement — per model. The command
    adds the ``omct-`` dataset prefix itself (AGENTS.md run-id grammar), and
    the row lands in the track's own table file (the command-derived
    (dataset, track) mapping).

    When the config asks for structural arms (a chunker with parameters plus
    aggregation rules) or for extra question transforms, the same run also
    scores those arms and compares every one of them with the full-text
    baseline arm — paired (the same pairs) and with the control-side
    counterpart next to the discrimination number.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.analysis is None:
        raise ValueError(
            "the analysis: section (the models to score with and the control design) "
            "is required for the `analyze-omct` command"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run analyze-omct")
    if not config.analysis.embedders:
        raise ValueError("analysis.embedders must name at least one model to score with")

    analysis = config.analysis
    pair_report = load_pairs(config.paths.data_external)
    spec = build_preprocess_spec(config.preprocess)

    baseline_chunker = build_chunker(BASELINE_CHUNKER)
    aggregations = parse_aggregations(analysis.aggregations)
    question_transforms = (analysis.question_transform, *analysis.question_transforms)
    bootstrap_seed = analysis.bootstrap.seed if analysis.bootstrap.seed is not None else config.seed

    view = build_dataset_view(
        pair_report.records,
        control_size=analysis.control_size,
        seed=config.seed,
        sampling_rule=analysis.sampling_rule,
    )

    models: list[dict[str, Any]] = []
    for entry in analysis.embedders:
        embedder = build_embedder(entry.embed)
        paragraph_version, _ = side_cache_versions(
            baseline_chunker, spec, analysis.question_transform, analysis.question_block_markers
        )
        paragraph_embedder = _cached_embedder(embedder, config, paragraph_version)
        question_embedders: dict[str, Any] = {}
        for transform in question_transforms:
            _, question_version = side_cache_versions(
                baseline_chunker, spec, transform, analysis.question_block_markers
            )
            question_embedders[transform] = _cached_embedder(embedder, config, question_version)

        # The chunked arms: one fixed chunker, or one side per point of the
        # swept grid (the semantic threshold of Stage 2.3).
        chunked_side = None
        swept_sides: list[ChunkedSide] = []
        if aggregations:
            if analysis.semantic_thresholds:
                swept_sides = _swept_sides(
                    config,
                    embedder,
                    analysis.chunker,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                    analysis.semantic_thresholds,
                )
            else:
                chunked_side = _chunked_side(
                    config,
                    embedder,
                    analysis.chunker,
                    analysis.question_chunker,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                )
        baseline = BaselineArm()
        model_block = score_model_arms(
            view,
            model_name=entry.name,
            paragraph_embedder=paragraph_embedder,
            question_embedders=question_embedders,
            baseline_chunker=baseline_chunker,
            question_transform=analysis.question_transform,
            question_block_markers=analysis.question_block_markers,
            preprocess_spec=spec,
            chunked_side=chunked_side,
            swept_sides=swept_sides,
            aggregations=aggregations,
            semantic_selection=analysis.semantic_selection,
            bootstrap_resamples=analysis.bootstrap.resamples,
            bootstrap_seed=bootstrap_seed,
            baseline=baseline,
        )
        # The learned arm (Stage 3.1): the feature pack + the combiner behind
        # the document-level holdout. The reference is the baseline arm's
        # full-text cosine on the same held-out documents.
        if analysis.stage3 is not None:
            # The Stage-3 random-vector floor (CS-6d): the feature pack's
            # embedding sides are random unit vectors instead of the model's.
            # The run's baseline arm (the full-text cosine) keeps the model's
            # embeddings, so the guard still holds and the arm-vs-cosine
            # comparison stays meaningful.
            feature_embedder = None
            embeddings_provenance: dict[str, Any] = {
                "mode": analysis.stage3.embeddings,
                "note": (
                    "the feature pack's embedding sides come from the run's "
                    "model (through the disk cache)"
                ),
            }
            if analysis.stage3.embeddings == "random":
                feature_embedder = RandomEmbedder(
                    dimension=int(embedder.info.dimension),
                    seed=config.seed,
                )
                embeddings_provenance = feature_embedder.provenance()
            extractor = _stage3_extractor(
                config,
                embedder,
                analysis,
                spec,
                view,
                paragraph_embedder=paragraph_embedder,
                question_embedder=question_embedders[analysis.question_transform],
                feature_embedder=feature_embedder,
            )
            model_block["stage3"] = score_stage3_model(
                view,
                model_name=entry.name,
                extractor=extractor,
                combiner_factory=lambda: build_combiner(
                    analysis.stage3.combiner.name, analysis.stage3.combiner.params
                ),
                reference_similarities=baseline.similarities,
                control_weight=analysis.stage3.control_weight,
                permutation_resamples=analysis.stage3.permutation_resamples,
                seed=config.seed,
                bootstrap_resamples=analysis.bootstrap.resamples,
                bootstrap_seed=bootstrap_seed,
                feature_embeddings=embeddings_provenance,
                # The cosine's calibration map (CS-6e): a logistic regression
                # on the cosine as its single feature, fitted on the fit side
                # only. It is always the logistic regression, whatever the
                # arm's own combiner is: the map is a calibration, not a model.
                calibration_map_factory=lambda: build_combiner(
                    COMBINER_LOGISTIC, {"C": 1.0, "max_iter": 1000, "standardize": True}
                ),
                reporting=ReportingSpec(
                    enabled=analysis.stage3.reporting.enabled,
                    bins=analysis.stage3.reporting.bins,
                    threshold_criterion=analysis.stage3.reporting.threshold_criterion,
                    threshold_quantiles=analysis.stage3.reporting.threshold_quantiles,
                    margin_quantiles=analysis.stage3.reporting.margin_quantiles,
                ),
            )
        models.append(model_block)

    # The decision arm (CS-7b): the decision model scores the **whole**
    # same-document grid, so it becomes an arm of this run — the same cells,
    # the same candidate lists, the same control, and the paired comparison
    # against the full-text cosine. The decision scorer is not an embedding
    # model, so it enters the artifact as its own entry.
    if analysis.decision is not None:
        reporting = ReportingSpec(
            enabled=True,
            bins=analysis.stage3.reporting.bins if analysis.stage3 is not None else 10,
            threshold_criterion=(
                analysis.stage3.reporting.threshold_criterion
                if analysis.stage3 is not None
                else "floor_gap"
            ),
            threshold_quantiles=(
                analysis.stage3.reporting.threshold_quantiles if analysis.stage3 is not None else 99
            ),
            margin_quantiles=(
                analysis.stage3.reporting.margin_quantiles if analysis.stage3 is not None else 49
            ),
        )
        for model_config in analysis.decision.models:
            decider = build_decider(model_config, config)
            arm = DecisionArm(
                name=model_config.name,
                state=model_config.state,
                question_type=model_config.question_type,
                instructions=model_config.instructions,
                options=model_config.options,
                criteria=model_config.criteria,
                positive_option=model_config.positive_option,
                max_input_tokens=model_config.max_input_tokens,
            )
            decision_grid = score_decision_grid(
                view,
                decider,
                arm,
                question_transform=analysis.question_transform,
                question_block_markers=analysis.question_block_markers,
                max_questions_per_call=analysis.decision.max_questions_per_call,
            )
            mismatched_grid = None
            if analysis.decision.mismatched_control:
                mismatched_grid = score_decision_grid(
                    view,
                    decider,
                    arm,
                    question_transform=analysis.question_transform,
                    question_block_markers=analysis.question_block_markers,
                    max_questions_per_call=analysis.decision.max_questions_per_call,
                    mismatched=True,
                )
            decision_block = score_decision_arm(view, decision=decision_grid)
            comparison = compare_arms(
                view,
                arm_name=DECISION_ARM,
                reference_similarities=baseline.similarities,
                arm_similarities=decision_grid.matrix,
                resamples=analysis.bootstrap.resamples,
                seed=bootstrap_seed,
            )
            rule_block = decision_rule_block(
                view,
                grid=build_stage3_grid(view),
                scores=decision_grid.scores,
                mismatched_scores=(None if mismatched_grid is None else mismatched_grid.scores),
                reporting=reporting,
            )
            info = decider.info
            models.append(
                {
                    "name": f"decision-{model_config.name}",
                    "kind": "decision",
                    "model": {
                        "name": model_config.name,
                        "model_id": info.model_id,
                        "digest": info.digest,
                        "backend": info.backend,
                        "host": info.host,
                        "prompt_version": analysis.decision.prompt_version,
                        "state": model_config.state,
                        "question_type": model_config.question_type,
                        "instructions": model_config.instructions,
                        "options": list(model_config.options),
                        "criteria": dict(model_config.criteria),
                        "positive_option": model_config.positive_option,
                        "max_input_tokens": model_config.max_input_tokens,
                        "extra": info.extra,
                    },
                    "arms": {DECISION_ARM: decision_block},
                    "comparisons": {DECISION_ARM: comparison},
                    "rule": rule_block,
                    "shortlist_rerank": shortlist_rerank_block(
                        view,
                        reference_similarities=baseline.similarities,
                        decision_matrix=decision_grid.matrix,
                        ks=analysis.decision.rerank_ks,
                    ),
                    "truncation": {
                        "n_cells": decision_grid.n_cells,
                        "n_truncated": decision_grid.n_truncated,
                        "share": (
                            decision_grid.n_truncated / decision_grid.n_cells
                            if decision_grid.n_cells
                            else None
                        ),
                        "mismatched": {
                            "n_cells": 0 if mismatched_grid is None else mismatched_grid.n_cells,
                            "n_truncated": (
                                0 if mismatched_grid is None else mismatched_grid.n_truncated
                            ),
                        },
                        "note": (
                            "the cells whose paragraph was cut to fit the model's "
                            "input budget (the question and the instructions are "
                            "never cut): a shortened input is never silent"
                        ),
                    },
                    "cache": {
                        "hits": getattr(decider, "cache_hits", 0),
                        "misses": getattr(decider, "cache_misses", 0),
                        "stored": getattr(decider, "n_stored", 0),
                    },
                    "usage": dict(getattr(decider, "usage", {})),
                    "n_calls_planned": decision_grid.n_calls,
                }
            )

    omct_run_id = _omct_run_id(run_id)
    payload = build_analysis_report(
        view,
        models,
        run_id=omct_run_id,
        source=str(config.paths.data_external),
        source_sha256=file_sha256(config.paths.data_external),
        seed=config.seed,
        control_size=analysis.control_size,
        sampling_rule=analysis.sampling_rule,
        question_transform=analysis.question_transform,
        chunker_name=analysis.chunker.name,
        chunker_params=analysis.chunker.params,
        question_chunker_name=(
            None if analysis.question_chunker is None else analysis.question_chunker.name
        ),
        question_chunker_params=(
            {} if analysis.question_chunker is None else analysis.question_chunker.params
        ),
        semantic_thresholds=analysis.semantic_thresholds,
        semantic_selection=analysis.semantic_selection,
        aggregations=[rule.spec for rule in aggregations],
        question_transforms=list(analysis.question_transforms),
        bootstrap={
            "resamples": analysis.bootstrap.resamples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        stage3=(
            None
            if analysis.stage3 is None
            else {
                "features": {
                    "paragraph_chunker": analysis.stage3.features.paragraph_chunker.model_dump(
                        mode="python"
                    ),
                    "question_chunker": (
                        None
                        if analysis.stage3.features.question_chunker is None
                        else analysis.stage3.features.question_chunker.model_dump(mode="python")
                    ),
                    "top_k": analysis.stage3.features.top_k,
                    "groups": list(analysis.stage3.features.groups),
                    "profile_top_k": analysis.stage3.features.profile_top_k,
                    "profile_ngram_size": analysis.stage3.features.profile_ngram_size,
                },
                "combiner": analysis.stage3.combiner.model_dump(mode="python"),
                "control_weight": analysis.stage3.control_weight,
                "permutation_resamples": analysis.stage3.permutation_resamples,
                "embeddings": analysis.stage3.embeddings,
                "reporting": analysis.stage3.reporting.model_dump(mode="python"),
            }
        ),
        baseline_chunker_name=baseline_chunker.name,
        n_valid_pairs=pair_report.n_valid,
        n_malformed_pairs=pair_report.n_invalid,
        total_seconds=time.perf_counter() - started,
    )
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_rows: list[dict[str, Any]] = []
    has_stage3 = False
    for model in models:
        if model.get("kind") == "decision":
            # The decision arm (CS-7b): its own entry, one row per arm.
            for arm_name, arm in (model.get("arms") or {}).items():
                table_rows.append(
                    _decision_arm_row(
                        model,
                        run_id=omct_run_id,
                        arm_name=arm_name,
                        arm=arm,
                        comparison=(model.get("comparisons") or {}).get(arm_name),
                        n_pairs=view.pooled["n_pairs"],
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            continue
        stage3 = model.get("stage3")
        if stage3 is not None:
            has_stage3 = True
            table_rows.append(
                _stage3_table_row(
                    model,
                    run_id=omct_run_id,
                    stage3=stage3,
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
            continue
        arms = model.get("arms", {})
        comparisons = model.get("comparisons", {})
        if not arms:
            table_rows.append(
                _analyze_omct_row(
                    model,
                    run_id=omct_run_id,
                    arm_name=BASELINE_ARM,
                    arm=model,
                    comparison=None,
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
            continue
        table_rows.append(
            _analyze_omct_row(
                model,
                run_id=omct_run_id,
                arm_name=BASELINE_ARM,
                arm=model,
                comparison=None,
                n_pairs=view.pooled["n_pairs"],
                n_control=view.control.n_cells,
                n_docs=view.pooled["n_documents"],
                seconds=payload["total_seconds"],
            )
        )
        for arm_name, arm in arms.items():
            table_rows.append(
                _analyze_omct_row(
                    model,
                    run_id=omct_run_id,
                    arm_name=arm_name,
                    arm=arm,
                    comparison=comparisons.get(arm_name),
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
    has_arms = any(model.get("arms") for model in models)
    has_reporting = any((model.get("stage3") or {}).get("reporting") for model in models)
    if has_stage3:
        columns = (
            ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS if has_reporting else ANALYZE_OMCT_STAGE3_COLUMNS
        )
        key_columns = ANALYZE_OMCT_STAGE3_KEY_COLUMNS
    elif has_arms:
        columns = ANALYZE_OMCT_ARMS_COLUMNS
        key_columns = ANALYZE_OMCT_ARMS_KEY_COLUMNS
    else:
        columns = ANALYZE_OMCT_COLUMNS
        key_columns = ANALYZE_OMCT_KEY_COLUMNS
    _append_command_rows(
        config,
        "analyze-omct",
        columns=columns,
        key_columns=key_columns,
        rows=table_rows,
    )

    print(f"OMCT analysis (Stage 0): {omct_run_id}")
    print(
        f"  grid: {view.pooled['grid_cells']} same-document cells, "
        f"{view.pooled['judged_cells']} judged "
        f"({view.pooled['grid_coverage']:.1%} covered), "
        f"{view.pooled['n_paragraphs']} paragraphs, {view.pooled['n_questions']} questions"
    )
    print(
        f"  control: {view.control.n_cells} unjudged cells (seed {config.seed}, "
        f"rule {analysis.sampling_rule}); "
        f"quotas: "
        + ", ".join(
            f"{document} {quota}" for document, quota in sorted(view.control.quotas.items())
        )
    )
    for model in models:
        if model.get("kind") == "decision":
            # The decision arm (CS-7b): its own entry, printed on its own lines.
            decision_block = model["arms"][DECISION_ARM]
            decision_measurement = decision_block.get("measurement", decision_block)
            decomposition = decision_measurement.get("pair_auc_decomposition") or {}
            rule = (model.get("rule") or {}).get("pooled") or {}
            fixed = rule.get(DECISION_RULE_FIXED_CUT) or {}
            counterpart = fixed.get("counterpart") or {}
            print(
                f"  [decision:{model['model']['name']}] {model['model']['model_id']} "
                f"(digest {model['model']['digest'][:12]}, {model['model']['state']}/"
                f"{model['model']['question_type']}): within-document AUC "
                f"{_fmt(decomposition.get('within_group_auc'), 4)}, pooled AUC "
                f"{_fmt(decision_measurement['pair_auc'].get('overall'), 4)}, macro "
                f"{_fmt(decision_measurement['pair_auc'].get('macro'), 4)}"
            )
            print(
                f"      the rule's fixed cut: the reach "
                f"{_fmt(fixed.get('coverage'), 4)} over the unjudged grid, the harm "
                f"{_fmt(fixed.get('harm'), 4)} on the verified positives; the "
                "label-blind counterpart's harm "
                f"{_fmt((counterpart.get('mean') or {}).get('harm'), 4)}"
            )
            continue
        info = model["embedder"]
        distributions = model["score_distributions"]
        enrichment = model["negative_enrichment"]["rank_percentile"]
        top1 = model["top1_candidate_provenance"]
        print(
            f"  [{model['name']}] {info['backend']}:{info['model_id']} "
            f"(dim {info['dimension']}): pair-AUC "
            f"{model['pair_auc']['overall']:.4f}; control mean "
            f"{distributions['control']['mean']:.3f}, p50 "
            f"{distributions['control']['quantiles']['0.5']:.3f}"
        )
        print(
            f"      rank percentile (0 = top of the candidate list): positives "
            f"{enrichment['label1']['quantiles']['0.5']:.3f}, negatives "
            f"{enrichment['label0']['quantiles']['0.5']:.3f}, control "
            f"{enrichment['control']['quantiles']['0.5']:.3f}"
        )
        print(
            f"      top-1 candidate per paragraph (model-dependent): positive "
            f"{top1['given_positive']}, negative {top1['given_negative']}, unjudged "
            f"{top1['unjudged']} of {top1['n_paragraphs']}"
        )
        for arm_name, comparison in model.get("comparisons", {}).items():
            paired = comparison["paired_auc"]
            shift = comparison["control_shift"]
            gap = comparison["floor_gap"]
            print(
                f"      [{arm_name}] dAUC {paired['delta_auc']:+.4f} "
                f"(95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}], "
                f"P(d>0) {paired['p_delta_positive']:.3f}); control mean "
                f"{shift['mean_delta']:+.4f}; floor gap {gap['delta']:+.4f}; "
                f"de-shifted AUC {comparison['deshifted_auc']['overall']:.4f}"
            )
        stage3 = model.get("stage3")
        if stage3 is not None:
            comparison = stage3["comparison"]
            decomposition = comparison["auc_decomposition"]
            within_paired = decomposition.get("within_group_paired") or {}
            floor = stage3["permutation_floor"]
            print(
                f"      [stage3] out-of-fold: within-document AUC "
                f"{_fmt(decomposition['arm'].get('within_group_auc'), 4)} "
                f"(baseline {_fmt(decomposition['reference'].get('within_group_auc'), 4)}, "
                f"delta {_fmt(decomposition.get('within_group_delta'), 4)}, "
                f"95% CI [{_fmt(within_paired.get('ci_low'), 4)}, "
                f"{_fmt(within_paired.get('ci_high'), 4)}]); pooled AUC "
                f"{_fmt(comparison['pair_auc']['arm'], 4)} "
                f"(delta {_fmt(comparison['paired_auc']['delta_auc'], 4)}); "
                f"chance floor {_fmt(floor['within_document']['chance'], 4)}, "
                f"control-role floor {_fmt(floor['within_document']['control_role'], 4)}"
            )
            groups = stage3["feature_pack"].get("groups") or []
            if groups:
                print(
                    f"        feature groups: {'+'.join(groups)} "
                    f"({stage3['feature_pack'].get('n_features')} features)"
                )
            diagnostics = (stage3.get("feature_diagnostics") or {}).get("per_feature") or {}
            selected = set(stage3["feature_pack"].get("feature_names") or [])
            if diagnostics and selected:
                top = sorted(
                    ((name, entry) for name, entry in diagnostics.items() if name in selected),
                    key=lambda item: (
                        item[1].get("within_auc") if item[1].get("within_auc") is not None else -1.0
                    ),
                    reverse=True,
                )[:3]
                print(
                    "        best features of the arm by their own within-document AUC: "
                    + ", ".join(
                        f"{name} ({_fmt(entry.get('within_auc'), 4)}, "
                        f"control mean {_fmt(entry.get('control', {}).get('mean'), 4)})"
                        for name, entry in top
                    )
                )
            length_check = stage3.get("length_check") or {}
            if length_check:
                arm_pearson = (length_check.get("arm") or {}).get("overall", {}).get("pearson")
                reference_pearson = (
                    (length_check.get("reference") or {}).get("overall", {}).get("pearson")
                )
                shift_pearson = (
                    (length_check.get("rank_shift") or {}).get("overall", {}).get("pearson")
                )
                print(
                    "        length vs control score (Pearson): arm "
                    f"{_fmt(arm_pearson, 4)}, "
                    f"cosine {_fmt(reference_pearson, 4)}, "
                    f"rank shift {_fmt(shift_pearson, 4)}"
                )
            reporting = stage3.get("reporting") or {}
            if reporting:
                calibration = reporting.get("calibration") or {}
                arm_calibration = calibration.get("arm") or {}
                cosine_calibration = calibration.get("cosine") or {}
                print(
                    "        calibration (the reliability diagram's expected "
                    "calibration error): arm "
                    f"{_fmt((arm_calibration.get('pooled') or {}).get('ece'), 4)} "
                    f"(per-document macro {_fmt(arm_calibration.get('macro_ece'), 4)}), "
                    f"cosine {_fmt((cosine_calibration.get('pooled') or {}).get('ece'), 4)} "
                    f"(per-document macro {_fmt(cosine_calibration.get('macro_ece'), 4)}); "
                    "the Brier scores "
                    f"{_fmt((arm_calibration.get('pooled') or {}).get('brier'), 4)} / "
                    f"{_fmt((cosine_calibration.get('pooled') or {}).get('brier'), 4)}"
                )
                thresholds = reporting.get("thresholds") or {}
                for document, block in (thresholds.get("per_document") or {}).items():
                    operating = (block.get("holdout") or {}).get("operating_point") or {}
                    print(
                        f"        threshold ({thresholds.get('criterion')}) for "
                        f"{document[:40]}: {_fmt(block.get('fit_side', {}).get('threshold'), 4)} "
                        f"(the holdout's recall {_fmt(operating.get('recall'), 4)}, "
                        f"precision {_fmt(operating.get('precision'), 4)})"
                    )
                out_of_fold = thresholds.get("out_of_fold") or {}
                median_operating = out_of_fold.get("operating_point_at_median") or {}
                print(
                    "        the pooled out-of-fold cut (mixes three calibrations, "
                    f"not like-for-like): median threshold "
                    f"{_fmt(out_of_fold.get('median_threshold'), 4)}, recall "
                    f"{_fmt(median_operating.get('recall'), 4)}"
                )
                safe_negative = reporting.get("safe_negative") or {}
                for variant, block in (safe_negative.get("variants") or {}).items():
                    pooled = block.get("pooled") or {}
                    floor = block.get("floor_arm") or {}
                    print(
                        f"        safe-negative rule [{variant}]: the unjudged grid's "
                        f"coverage {_fmt(pooled.get('coverage'), 4)} "
                        f"({pooled.get('n_unjudged_cleared')} of "
                        f"{pooled.get('n_unjudged_cells')}), "
                        f"the verified positives' harm {_fmt(pooled.get('harm'), 4)}; "
                        "the control-role floor's harm at the same coverage "
                        f"{_fmt((floor.get('harm_at_matched_coverage') or {}).get('harm'), 4)}"
                    )
            rotations = stage3.get("rotations") or []
            usage: dict[str, Any] = {}
            if rotations:
                usage = (rotations[0].get("feature_usage") or {}).get("per_feature") or {}
            if usage:
                top = sorted(
                    usage.items(),
                    key=lambda item: item[1].get("total_gain", 0.0),
                    reverse=True,
                )[:3]
                print(
                    "        top features by the trees' gain (first rotation): "
                    + ", ".join(
                        f"{name} ({int(entry.get('n_splits', 0))} splits, "
                        f"gain {entry.get('total_gain', 0.0):.3f})"
                        for name, entry in top
                    )
                )
            for rotation in stage3["rotations_summary"]:
                print(
                    f"        holdout {rotation['holdout'][:40]}: arm "
                    f"{_fmt(rotation['arm_pair_auc'], 4)} (baseline "
                    f"{_fmt(rotation['reference_pair_auc'], 4)}, chance floor "
                    f"{_fmt(rotation['chance_floor_mean'], 4)}, control-role floor "
                    f"{_fmt(rotation['control_role_floor_mean'], 4)})"
                )
    print(f"Report written to {results_dir / 'analysis.json'}")
    return 0


def build_decider(model_config: Any, config: AppConfig) -> Any:
    """Build the decision model client for one arm (the CLI's wiring seam).

    The client is the raw ``POST /v1/systemone`` caller (the endpoint is in
    neither the Ollama CLI nor its Python/JS libraries); it checks the pinned
    digest against the server's and **fails the run** on a mismatch. The
    decision models' own disk cache wraps it (its own root under ``cache/``,
    never the embedding cache), so a re-run is reproducible and never
    re-judges a pair.
    """
    # The probe makes thousands of calls; httpx logs one INFO line per request,
    # which would bury the run's own output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = SystemOneDecider(
        model_config,
        host=config.analysis.decision.host if config.analysis and config.analysis.decision else "",
        timeout=(
            config.analysis.decision.timeout
            if config.analysis and config.analysis.decision
            else 600.0
        ),
    )
    prompt_version = (
        config.analysis.decision.prompt_version
        if config.analysis and config.analysis.decision
        else "v1"
    )
    return CachedDecider(client, config.paths.cache, prompt_version=prompt_version)


def _decide_omct_row(
    model_block: dict[str, Any],
    *,
    run_id: str,
    n_pairs: int,
    n_control: int,
    seconds: float,
) -> dict[str, Any]:
    """One row of the decision probe's comparison table."""
    measurement = model_block["measurement"]
    pair_auc = measurement["pair_auc"]
    decomposition = measurement["pair_auc_decomposition"]
    calibration = measurement["calibration"]
    distributions = measurement["score_distributions"]
    competence = model_block.get("competence_control") or {}
    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "model": model_block["model"]["model_id"],
        "model_version": model_block["model"]["digest"][:12],
        "state": model_block["state"],
        "question_type": model_block["question_type"],
        "scope": model_block["scope"],
        "within_auc": _fmt(decomposition.get("within_group_auc"), 4),
        "pooled_auc": _fmt(pair_auc.get("overall"), 4),
        "macro_auc": _fmt(pair_auc.get("macro"), 4),
        "cross_auc": _fmt(decomposition.get("cross_group_auc"), 4),
        "ece": _fmt((calibration.get("pooled") or {}).get("ece"), 4),
        "ece_macro": _fmt(calibration.get("macro_ece"), 4),
        "brier": _fmt((calibration.get("pooled") or {}).get("brier"), 4),
        "pos_mean": _fmt((distributions.get("label1") or {}).get("mean"), 4),
        "neg_mean": _fmt((distributions.get("label0") or {}).get("mean"), 4),
        "ctrl_mean": _fmt((distributions.get("control") or {}).get("mean"), 4),
        "floor_gap": _fmt((measurement.get("floor_gap") or {}).get("gap"), 4),
        "comp_auc": _fmt(competence.get("pair_auc"), 4),
        "comp_pos_mean": _fmt((competence.get("positive") or {}).get("mean"), 4),
        "comp_neg_mean": _fmt((competence.get("negative") or {}).get("mean"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_comp": competence.get("n_cells"),
        "n_calls": model_block.get("n_calls_planned"),
        "trunc": _fmt((model_block.get("truncation") or {}).get("share"), 4),
        "seconds": _fmt(seconds, 3),
    }


def _cmd_decide_omct(config: AppConfig) -> int:
    """Run decide-omct: the track's Stage-3.5 decision-model probe.

    Writes ``results/omct-<run_id>/{config.yaml, decisions.json}``: the
    per-cell probabilities, the measurement blocks (the pair AUC pooled and per
    document, the within-document AUC, the rank percentiles, the control side,
    the calibration curve), the synthetic competence control and the
    provenance (the model id, the pinned digest, the prompt version, the
    cache's hit/miss counters) — per arm.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.analysis is None or config.analysis.decision is None:
        raise ValueError(
            "the analysis.decision: section (the decision models and the prompt "
            "grid) is required for the `decide-omct` command"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run decide-omct")

    decision = config.analysis.decision
    pair_report = load_pairs(config.paths.data_external)
    view = build_dataset_view(
        pair_report.records,
        control_size=config.analysis.control_size,
        seed=config.seed,
        sampling_rule=config.analysis.sampling_rule,
    )

    models: list[dict[str, Any]] = []
    for model_config in decision.models:
        decider = build_decider(model_config, config)
        arm = DecisionArm(
            name=model_config.name,
            state=model_config.state,
            question_type=model_config.question_type,
            instructions=model_config.instructions,
            options=model_config.options,
            criteria=model_config.criteria,
            positive_option=model_config.positive_option,
            max_input_tokens=model_config.max_input_tokens,
        )
        block = run_decision_arm(
            view,
            decider,
            arm,
            scope=decision.scope,
            question_transform=config.analysis.question_transform,
            question_block_markers=config.analysis.question_block_markers,
            competence_control=decision.competence_control,
            competence_pairs=decision.competence_pairs,
            max_questions_per_call=decision.max_questions_per_call,
            bins=decision.bins,
        )
        info = decider.info
        block["model"] = {
            "name": model_config.name,
            "model_id": info.model_id,
            "digest": info.digest,
            "backend": info.backend,
            "host": info.host,
            "prompt_version": decision.prompt_version,
            "extra": info.extra,
        }
        block["cache"] = {
            "hits": getattr(decider, "cache_hits", 0),
            "misses": getattr(decider, "cache_misses", 0),
            "stored": getattr(decider, "n_stored", 0),
        }
        block["usage"] = dict(getattr(decider, "usage", {}))
        models.append(block)

    omct_run_id = _omct_run_id(run_id)
    payload = {
        "run_id": omct_run_id,
        "command": "decide-omct",
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": pair_report.n_valid,
        "n_malformed_pairs": pair_report.n_invalid,
        "seed": config.seed,
        "analysis": {
            "scope": decision.scope,
            "prompt_version": decision.prompt_version,
            "question_transform": config.analysis.question_transform,
            "question_block_markers": list(config.analysis.question_block_markers),
            "control": view.control.describe(),
            "grid": {
                "n_documents": view.pooled["n_documents"],
                "n_paragraphs": view.pooled["n_paragraphs"],
                "n_questions": view.pooled["n_questions"],
                "n_cells": view.pooled["grid_cells"],
                "n_judged_cells": view.pooled["judged_cells"],
                "n_unjudged_cells": view.pooled["unjudged_cells"],
                "coverage": view.pooled["grid_coverage"],
            },
            "strata": list(view.strata),
            "pooled": view.pooled,
            "notes": [
                "OMCT is auxiliary data (PLAN.md section 2): these numbers annotate "
                "method choice, they are never a UPR metric.",
                "The two negative subsets are never pooled: the reviewer-verified "
                "hard negatives (the reviewed set's label 0) and the unjudged-proxy "
                "control cells are separate groups.",
                "The decision models are frozen and zero-shot: they carry no OMCT "
                "label, so nothing leaks. Fine-tuning one on OMCT would break the "
                "track's leakage rule (i) (101 question numbers are positive in one "
                "document and negative in another).",
                "A decision model only scores: it never relabels, curates or "
                "corrects the data (the raw pair file is read-only).",
                "The rank percentiles and the hit@k are measured inside the cells "
                "the run scored for a paragraph (the restricted candidate list), "
                "not inside the paragraph's full same-document question list: the "
                "full list needs the grid scope.",
            ],
        },
        "models": models,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "decisions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    seconds = payload["total_seconds"]
    rows = [
        _decide_omct_row(
            block,
            run_id=omct_run_id,
            n_pairs=view.pooled["n_pairs"],
            n_control=view.control.n_cells,
            seconds=seconds,
        )
        for block in models
    ]
    _append_command_rows(
        config,
        "decide-omct",
        columns=DECIDE_OMCT_COLUMNS,
        key_columns=DECIDE_OMCT_KEY_COLUMNS,
        rows=rows,
    )

    print(f"OMCT decision probe (Stage 3.5): {omct_run_id}")
    print(
        f"  scope: {decision.scope}; {view.pooled['n_pairs']} reviewed pairs, "
        f"{view.control.n_cells} control cells, "
        f"{view.pooled['n_documents']} documents"
    )
    for block in models:
        measurement = block["measurement"]
        decomposition = measurement["pair_auc_decomposition"]
        calibration = measurement["calibration"]
        competence = block.get("competence_control") or {}
        print(
            f"  [{block['arm']}] {block['model']['model_id']} "
            f"(digest {block['model']['digest'][:12]}, {block['state']}/"
            f"{block['question_type']}): within-document AUC "
            f"{_fmt(decomposition.get('within_group_auc'), 4)}, pooled AUC "
            f"{_fmt(measurement['pair_auc'].get('overall'), 4)}, macro "
            f"{_fmt(measurement['pair_auc'].get('macro'), 4)}"
        )
        print(
            f"      calibration: ECE {_fmt((calibration.get('pooled') or {}).get('ece'), 4)} "
            f"(per-document macro {_fmt(calibration.get('macro_ece'), 4)}), the Brier "
            f"{_fmt((calibration.get('pooled') or {}).get('brier'), 4)}; the floor gap "
            f"{_fmt((measurement.get('floor_gap') or {}).get('gap'), 4)}"
        )
        if competence:
            print(
                f"      competence control: AUC {_fmt(competence.get('pair_auc'), 4)} "
                f"(the positives' mean {_fmt((competence.get('positive') or {}).get('mean'), 4)}, "
                f"the negatives' {_fmt((competence.get('negative') or {}).get('mean'), 4)})"
            )
        print(
            f"      calls: {(block.get('usage') or {}).get('n_calls', 0)}; the cache's "
            f"hits {(block.get('cache') or {}).get('hits', 0)}, misses "
            f"{(block.get('cache') or {}).get('misses', 0)}"
        )
    print(f"Report written to {results_dir / 'decisions.json'}")
    return 0


def _cmd_evaluate(config: AppConfig, split: str | None = None) -> int:
    """Run the evaluate command: any predictor on a fixed split (Phase 3 exit criterion).

    Loads the dataset and the fixed split ids (with a provenance sha check, never
    re-splitting), constructs the configured predictor from **train** context only,
    scores the evaluation split and assembles the per-bucket report. Embedding-
    dependent predictors (Phase 4) additionally consume
    ``eval.embedding_run_id``: the artifact is loaded, hard-validated and
    id-aligned (train document vectors only at construction; query vectors at
    score time), and its lineage is recorded in the metrics. Writes
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

    # Phase 4 wiring (PLAN Phase 4 opening D + zero-shot): resolve the
    # embedding-run references against the declared needs of the predictor
    # (params included — fusion needs label vectors only when c > 0). The
    # artifact holds the full dataset's vectors, but only the train rows reach
    # the predictor's construction context and only the eval split's query rows
    # reach score().
    requires, required_kinds, needs_labels = predictor_requirements(
        eval_config.predictor.name, eval_config.predictor.params
    )
    artifact: EmbeddingArtifact | None = None
    label_artifact: LabelEmbeddingArtifact | None = None
    if eval_config.embedding_run_id and not requires:
        logger.warning(
            "eval.embedding_run_id %r is set but predictor %r does not use "
            "embeddings; ignoring it (the run carries no embedding lineage).",
            eval_config.embedding_run_id,
            eval_config.predictor.name,
        )
    elif requires:
        if not eval_config.embedding_run_id:
            raise ValueError(
                f"predictor {eval_config.predictor.name!r} requires embeddings; set "
                "eval.embedding_run_id to a prior UPR embedding run_id "
                "(e.g. 'upr-embed-e5-base-v2') — see PLAN.md Phase 4 opening D"
            )
        artifact = load_embedding_artifact(
            config.paths.results,
            eval_config.embedding_run_id,
            required_kinds=required_kinds,
            source_sha256=source_sha,
        )
    if eval_config.label_embedding_run_id and not needs_labels:
        logger.warning(
            "eval.label_embedding_run_id %r is set but predictor %r does not use "
            "zero-shot label vectors; ignoring it (no lineage is recorded).",
            eval_config.label_embedding_run_id,
            eval_config.predictor.name,
        )
    elif needs_labels:
        if not requires or artifact is None:
            raise ValueError(
                f"predictor {eval_config.predictor.name!r} wants zero-shot label "
                "vectors but does not consume the primary embedding run — label "
                "vectors are only defined inside one model's space; this predictor "
                "combination is unsupported"
            )
        if not eval_config.label_embedding_run_id:
            raise ValueError(
                f"predictor {eval_config.predictor.name!r} requires zero-shot label "
                "vectors; set eval.label_embedding_run_id to a prior UPR "
                "label-embedding run_id (e.g. 'upr-labelembed-jina-v5-about') — "
                "see PLAN.md Phase 4 zero-shot"
            )

    labels = sorted({label for record in full_report.records for label in record.labels})
    if needs_labels:
        assert artifact is not None  # guarded by the needs_labels branch above
        label_artifact = load_label_embedding_artifact(
            config.paths.results,
            eval_config.label_embedding_run_id,
            labels=labels,
            source_sha256=source_sha,
            model_id=artifact.meta["embedder"]["model_id"],
            dimension=artifact.meta["embedder"]["dimension"],
        )
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

    train_vectors = (
        artifact.select("document", ids_by_split["train"])
        if artifact is not None and "document" in required_kinds
        else None  # zero-shot predictors use query vectors only
    )
    predictor = build_predictor(
        eval_config.predictor.name,
        eval_config.predictor.params,
        PredictorContext(
            labels=tuple(labels),
            train_records=tuple(train_records),
            train_document_vectors=train_vectors,
            label_vectors=label_artifact.vectors if label_artifact is not None else None,
        ),
        seed=config.seed,
    )
    texts = [record.text for record in eval_records]
    if artifact is not None:
        query_vectors = artifact.select("query", ids_by_split[split_name])
        scores = predictor.score(texts, query_vectors=query_vectors)
    else:
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
        "embedding_run": artifact.lineage if artifact is not None else None,
        "label_embedding_run": (label_artifact.lineage if label_artifact is not None else None),
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

    decision_cell = {
        "top_k": f"top_k k={decision.k}",
        "score_threshold": f"thr={decision.threshold}",
        "top_k_min_score": f"top_k k={decision.k} min={decision.min_score}",
    }[decision.rule]
    _append_command_rows(
        config,
        "evaluate",
        columns=EVALUATE_COLUMNS,
        key_columns=EVALUATE_KEY_COLUMNS,
        rows=[
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "split": split_name,
                "predictor": eval_config.predictor.name,
                "embedding": (
                    eval_config.embedding_run_id.removeprefix("upr-embed-")
                    if artifact is not None
                    else None
                ),
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

    While the file executes, two directories are importable: the sweep file's
    own directory (sibling modules) and the shared ``experiments/``
    directory, so ``embedder_pool.MODELS`` stays shared no matter where the
    sweep file lives (the text-processing track's sweeps live in
    ``experiments_text_processing/``). Both are removed again afterwards.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("cluster_rag_sweep_file", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load sweep file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The sweep file's own directory first (its priority is the higher one),
    # then the shared experiments directory.
    candidates = [str(path.parent.resolve()), str(SHARED_EXPERIMENTS_DIR)]
    added: list[str] = []
    for entry in candidates:
        if entry in added or entry in sys.path or not Path(entry).is_dir():
            continue
        added.append(entry)
    for entry in added:
        sys.path.insert(0, entry)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import error aborts with its cause
        raise ValueError(f"loading sweep file {path} failed: {exc}") from exc
    finally:
        for entry in added:
            sys.path.remove(entry)
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
        if run.command == "embed":  # kinds come from the cell's resolved config
            code = _cmd_embed(run.config, kinds=run.config.embed_kinds)
        elif run.command == "labelembed":  # the zero-shot label-text artifact
            code = _cmd_labelembed(run.config)
        elif run.command == "evaluate":  # split comes from the cell's eval.split
            code = _cmd_evaluate(run.config)
        elif run.command == "stats":  # plain UPR dataset statistics
            code = _cmd_stats(run.config)
        elif run.command == "stats-omct":  # the omct pairwise-data stats (stats --omct)
            code = _cmd_stats(run.config, omct=True)
        elif run.command == "split":  # sweep cells never re-split silently (--force only via CLI)
            code = _cmd_split(run.config, force=False)
        elif run.command == "probe-omct":
            code = _cmd_probe_omct(run.config)
        elif run.command == "analyze-omct":  # the text-processing track's Stage 0
            code = _cmd_analyze_omct(run.config)
        elif run.command == "decide-omct":  # the text-processing track's Stage 3.5
            code = _cmd_decide_omct(run.config)
        else:  # a command the CLI does not implement: fail loudly, never guess
            raise ValueError(
                f"sweep cell {run.label!r} runs command {run.command!r}, which this CLI "
                "does not implement (known commands: "
                f"{', '.join(sorted(SWEEP_COMMANDS))})"
            )
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
            return _cmd_stats(config, omct=args.omct)
        if args.command == "split":
            return _cmd_split(config, force=args.force)
        if args.command == "embed":
            # The --kinds flag wins over the config's embed_kinds; both end up
            # in the run's meta.json, so the record is unambiguous either way.
            kinds = _parse_kinds(args.kinds) if args.kinds else config.embed_kinds
            return _cmd_embed(config, kinds=kinds)
        if args.command == "labelembed":
            return _cmd_labelembed(config)
        if args.command == "evaluate":
            return _cmd_evaluate(config, split=args.split)
        if args.command == "probe-omct":
            return _cmd_probe_omct(config)
        if args.command == "analyze-omct":
            return _cmd_analyze_omct(config)
        if args.command == "decide-omct":
            return _cmd_decide_omct(config)
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
