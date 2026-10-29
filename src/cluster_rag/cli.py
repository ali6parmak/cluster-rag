"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Commands: stats (--omct for the OMCT pairwise data), split, embed
(Phase 2: dataset embedding through the disk cache), labelembed (Phase 4:
the zero-shot label-text artifact), probe-omct (Phase 2: the
OMCT pairwise embedding probe), analyze-omct (the text-processing track's
Stage-0 pair-score artifact), decide-omct (the track's Stage-3.5 decision
probe), analyze-upr (the track's UPR-side analysis artifact), evaluate
(Phase 3: any predictor on a fixed val/test split), sweep (Phases 2.5+: run a
Python sweep file's resolved cells). Run from the repository root so relative
paths in the config resolve. Commands write their artifacts to
``results/<run_id>/`` (frozen resolved config plus a JSON report), so
``run_id`` must be set in the config for any command. When the command's
comparison table is set (``paths.experiment_table_upr`` /
``paths.experiment_table_omct``, or the text-processing track's
``paths.experiment_table_text_processing_omct`` / ``_upr``), ``embed``,
``labelembed``, ``evaluate``, ``probe-omct``, ``analyze-omct`` and
``analyze-upr`` also append
one row per run (per transform for probe-omct, per model for analyze-omct, per
arm for analyze-upr) to
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    Record,
    build_label_matrix,
    build_splits,
    load_jsonl,
    load_split_ids,
    select_records_by_ids,
)
from cluster_rag.dataset_stats import build_dataset_stats, file_sha256
from cluster_rag.decision import CachedDecider, SystemOneDecider
from cluster_rag.embed import KINDS, build_embedder
from cluster_rag.embed.base import token_capability
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.embed.random import RandomEmbedder
from cluster_rag.embed_artifact import (
    EmbeddingArtifact,
    LabelEmbeddingArtifact,
    load_embedding_artifact,
    load_label_embedding_artifact,
)
from cluster_rag.eval.buckets import assign_label_buckets, labels_by_bucket
from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    decision_f1_metrics,
    paired_metric_bootstrap,
    ranking_metrics,
)
from cluster_rag.eval.report import build_label_report
from cluster_rag.experiment_table import (
    ANALYZE_OMCT_ARMS_COLUMNS,
    ANALYZE_OMCT_ARMS_KEY_COLUMNS,
    ANALYZE_OMCT_COLUMNS,
    ANALYZE_OMCT_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_COLUMNS,
    ANALYZE_OMCT_STAGE3_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS,
    ANALYZE_UPR_COLUMNS,
    ANALYZE_UPR_KEY_COLUMNS,
    DECIDE_OMCT_COLUMNS,
    DECIDE_OMCT_KEY_COLUMNS,
    DECIDE_UPR_COLUMNS,
    DECIDE_UPR_KEY_COLUMNS,
    EMBED_COLUMNS,
    EMBED_KEY_COLUMNS,
    EVALUATE_COLUMNS,
    EVALUATE_KEY_COLUMNS,
    PROBE_OMCT_COLUMNS,
    PROBE_OMCT_GROUPS_COLUMNS,
    PROBE_OMCT_GROUPS_KEY_COLUMNS,
    PROBE_OMCT_KEY_COLUMNS,
    RERANK_UPR_COLUMNS,
    RERANK_UPR_KEY_COLUMNS,
    append_table_rows,
    intro_for,
    section_heading,
    table_path,
    table_target,
)
from cluster_rag.label_catalog import LabelEntry, load_label_catalog
from cluster_rag.omct_analysis import (
    BASELINE_ARM,
    BASELINE_CHUNKER,
    DECISION_ARM,
    DECISION_RULE_FIXED_CUT,
    BaselineArm,
    ChunkedSide,
    LateSide,
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
from cluster_rag.rerank.base import KIND_LISTWISE
from cluster_rag.sweep import COMMANDS as SWEEP_COMMANDS
from cluster_rag.sweep import SHARED_EXPERIMENTS_DIR
from cluster_rag.text_processors.aggregation import aggregate_chunk_similarities, parse_aggregations
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.combining import COMBINER_LOGISTIC, build_combiner
from cluster_rag.text_processors.decision.upr_probe import measure_upr_arm
from cluster_rag.text_processors.decision.upr_union import z_max_union
from cluster_rag.text_processors.features import (
    FeatureEmbedders,
    PairFeatureExtractor,
    ProfileUnits,
)
from cluster_rag.text_processors.features.upr_backoff import (
    control_labels,
    sibling_backoff,
    unsupported_labels,
)
from cluster_rag.text_processors.features.upr_features import (
    UprFeatureEmbedders,
    UprFeatureExtractor,
    UprLabel,
)
from cluster_rag.text_processors.rerank.upr_rerank import (
    ORDER_TEXT_FIRST,
    candidate_ceiling,
    candidate_columns,
    fill_restricted_scores,
    label_documents,
    shifted_documents,
)
from cluster_rag.upr_analysis import (
    UprView,
    aggregation_of,
    arm_block,
    arm_name,
    build_upr_report,
    build_upr_view,
    chunked_arm_scores,
    chunks_block,
    encode_chunks,
    encode_late_chunks,
    fusion_components,
    per_text_block,
    symmetric_arm_scores,
)
from cluster_rag.upr_decision import (
    UprDecisionSample,
    build_upr_decision_report,
    run_upr_competence_control,
    run_upr_decision_arm,
    run_upr_listwise_stage,
    run_upr_rule_stage,
    run_upr_union_stage,
    sample_texts,
    score_upr_grid,
    upr_per_text_block,
)
from cluster_rag.upr_learning import score_upr_learned_arms
from cluster_rag.upr_rerank import (
    RerankSources,
    build_candidate_requests,
    build_upr_rerank_report,
    candidate_set_matrix,
    paired_arm_comparison,
    run_upr_rerank_arm,
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
            "(TEXT_PROCESSING_PLAN.md → the archive docs/text_processing_plan_archive.md, "
            "Stage 0.1): score every judged OMCT pair "
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
            "(TEXT_PROCESSING_PLAN.md → the archive docs/text_processing_plan_archive.md, "
            "Stage 3.5.1): score the reviewed OMCT "
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

    analyze_upr = subparsers.add_parser(
        "analyze-upr",
        help="Text-processing track: the UPR analysis artifact (analyze-upr).",
        description=(
            "The text-processing track's UPR artifact "
            "(TEXT_PROCESSING_PLAN.md → the archive docs/text_processing_plan_archive.md, "
            "CS-8a): score the evaluated split with the "
            "main pipeline's own predictor (the whole-text reference arm) and "
            "report the UPR-side diagnostics — the gold cells' rank percentiles "
            "inside their own text's label list, the hit@1 / hit@10, the non-gold "
            "cells as the background (fully enumerable on UPR: no control sample, "
            "no seed), the background's level and rank shift beside every headline "
            "number, the score distributions per bucket and per label, the texts' "
            "length buckets and the top-1 provenance. The reference arm must "
            "reproduce the matching `evaluate` run field by field (the guard). "
            "Writes results/<run_id>/analysis.json; the row goes to the track's "
            "own UPR table."
        ),
    )
    analyze_upr.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with eval: and upr_analysis: sections (or "
        "sweep-resolved configs; experiment values live in "
        "experiments_text_processing/*.py sweeps).",
    )

    decide_upr = subparsers.add_parser(
        "decide-upr",
        help="Text-processing track: the UPR decision-model probe (decide-upr).",
        description=(
            "The text-processing track's UPR decision probe "
            "(TEXT_PROCESSING_PLAN.md → the archive docs/text_processing_plan_archive.md, "
            "CS-9): the decision models answer the "
            "UPR-native question ('is this text about <label>?') per label, and "
            "the probe measures what their probability buys — the decision view "
            "and the ranking view, the gold side and the background side, the "
            "calibration and the probability compression — with the synthetic "
            "competence control and the label-blind counterpart beside every "
            "arm. The models are measurement instruments: they never enter the "
            "pipeline. Writes results/<run_id>/decisions.json; the row goes to "
            "the track's own UPR table."
        ),
    )
    decide_upr.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with eval: and upr_analysis: sections (or "
        "sweep-resolved configs; experiment values live in "
        "experiments_text_processing/*.py sweeps).",
    )

    rerank_upr = subparsers.add_parser(
        "rerank-upr",
        help="Text-processing track: the UPR zero-shot reranker probe (rerank-upr).",
        description=(
            "The text-processing track's UPR reranker probe "
            "(TEXT_PROCESSING_PLAN.md → the archive docs/text_processing_plan_archive.md, "
            "CS-15): the zero-shot rerankers RE-ORDER "
            "an existing candidate set (the decision model's top-k, the union's "
            "top-k, or the champion's top-k as the control), and the run compares "
            "their ordering with the ordering that produced the set — the same "
            "candidate set, two orderings, paired over the texts. The set's own "
            "ceiling is measured beside every arm, the label-blind counterpart "
            "is the yardstick, and the cost (the pairs per text, the pairs per "
            "second, the wall time, the peak GPU memory) is part of the "
            "deliverable. Writes results/<run_id>/rerank.json; the row goes to "
            "the track's own UPR table."
        ),
    )
    rerank_upr.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config with eval: and upr_analysis: sections (or "
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
    catalog = _upr_label_catalog(config) if config.label_texts.variant == "description" else {}
    texts: list[str] = []
    n_fallback = 0
    for label in labels:
        entry = catalog.get(label)
        description = entry.description if entry else None
        if config.label_texts.variant == "description" and not description:
            description = label
            n_fallback += 1
        texts.append(config.label_texts.render_text(label, description))

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
            "label_space": (
                "the dataset's sorted unique labels"
                + (
                    ""
                    if config.label_texts.variant != "description"
                    else " (the catalog's descriptions, with the name as the fallback)"
                )
            ),
            "rendered_example": texts[0],
            "rendered_texts": texts,
            "catalog_path": (
                str(config.paths.label_catalog)
                if config.label_texts.variant == "description" and config.paths.label_catalog
                else None
            ),
            "n_fallback_to_name": n_fallback,
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
        "chunker": (
            paragraph_side.get("side_slug")
            or paragraph_side.get("chunker_slug")
            or paragraph_side.get("chunker_name")
        ),
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


def _late_side(
    config: AppConfig,
    embedder: Any,
    analysis: AnalysisConfig,
) -> LateSide:
    """The late-chunked side (Stage 2.7): the chunker, the embedder, the poolings.

    The late side embeds the **whole document once** and pools the
    token-level embeddings per chunk, so it needs an embedder that can
    return token-level embeddings. The capability is checked here, before any
    work: a backend without it (Ollama's ``/api/embed`` returns one vector
    per text and nothing else) gets a clear message instead of a failure
    deep inside the run.

    The side uses the **raw** backend, not the cached one: the token-level
    pass is not cached (the cache keys on the chunk's text, and a
    late-chunked vector is a function of the whole document), so the cache
    wrapper would only add a layer without a hit.
    """
    if analysis.late_chunker is None:
        raise ValueError("the late-chunked side needs analysis.late_chunker")
    if token_capability(embedder) is None:
        backend = getattr(getattr(embedder, "info", None), "backend", "unknown")
        raise ValueError(
            "the late-chunked arm needs token-level embeddings; the backend "
            f"{backend!r} cannot return them (Ollama's /api/embed returns one "
            "vector per text and nothing else), so the arm is HF-only"
        )
    chunker = build_chunker(analysis.late_chunker.name, analysis.late_chunker.params)
    question_chunker = None
    if analysis.late_question_chunker is not None:
        question_chunker = build_chunker(
            analysis.late_question_chunker.name, analysis.late_question_chunker.params
        )
    return LateSide(
        chunker=chunker,
        paragraph_embedder=embedder,
        question_chunker=question_chunker,
        question_embedder=embedder if question_chunker is not None else None,
        poolings=tuple(analysis.late_pooling),
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
        header_side = None
        late_side = None
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
            if analysis.header_chunker is not None:
                # The chunk header (Stage 2.6): a second chunked side whose
                # chunks carry the paragraph's own first sentence, so the
                # header is measured against the no-header arm of this run.
                header_side = _chunked_side(
                    config,
                    embedder,
                    analysis.header_chunker,
                    None,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                )
        if analysis.late_chunker is not None:
            # The late chunking (Stage 2.7): the document embedded once and
            # the token-level embeddings pooled per chunk. Its arms run
            # beside the run's early-chunked arms, so the late geometry is
            # measured against the early one in the same run.
            late_side = _late_side(config, embedder, analysis)
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
            header_side=header_side,
            late_side=late_side,
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


def build_upr_decider(model_config: Any, config: AppConfig) -> Any:
    """Build the UPR probe's decision model client (the CLI's wiring seam).

    The same raw ``POST /v1/systemone`` caller as the OMCT probe (the
    endpoint is in neither the Ollama CLI nor its Python/JS libraries): it
    checks the pinned digest against the server's and **fails the run** on a
    mismatch, and the decision models' own disk cache wraps it (its own root
    under ``cache/``, never the embedding cache), so a re-run is reproducible
    and never re-judges a cell.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    decision = config.upr_analysis.decision if config.upr_analysis else None
    client = SystemOneDecider(
        model_config,
        host=decision.host if decision else "",
        timeout=decision.timeout if decision else 600.0,
    )
    prompt_version = decision.prompt_version if decision else "v1"
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


def _cmd_decide_upr(config: AppConfig) -> int:
    """Run decide-upr: the track's UPR decision-model probe (CS-9).

    The decision models are **measurement instruments** on the UPR side: they
    answer the UPR-native question ("is this text about *<label>*?") per
    label, and the probe measures what their probability buys. They never
    enter the pipeline.

    Writes ``results/<run_id>/{config.yaml, decisions.json}``: the sample, the
    arms' measurement blocks (the decision view, the ranking view, the gold
    side, the background side, the calibration and the probability
    compression), the **label-blind counterpart** beside every arm, the
    synthetic competence control and the provenance (the model id, the pinned
    digest, the prompt version, the cache's counters) — per arm.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.upr_analysis is None:
        raise ValueError(
            "the upr_analysis: section (the decision models and the prompt grid) "
            "is required for the `decide-upr` command"
        )
    decision = config.upr_analysis.decision
    if not decision.models:
        raise ValueError(
            "the upr_analysis.decision: section carries no model: the `decide-upr` "
            "command needs at least one pinned decision model"
        )
    if config.eval is None:
        raise ValueError(
            "the eval: section (the decision rule) is required for the `decide-upr` command"
        )
    setup = _prepare_evaluation(config, config.eval.split)
    view = build_upr_view(
        setup.records,
        setup.labels,
        setup.buckets,
        config.upr_analysis.length_buckets,
    )
    if decision.scope == "sample":
        sample = sample_texts(view, size=decision.sample_size, seed=decision.seed)
    else:
        sample = UprDecisionSample(
            texts=tuple(record.text for record in setup.records),
            labels=tuple(setup.labels),
            gold=view.gold,
            rows=tuple(range(view.n_texts)),
        )

    arms: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def _per_text(matrix: np.ndarray) -> dict[str, Any]:
        """The arm's per-text summary (what a paired comparison needs)."""
        return upr_per_text_block(
            matrix,
            sample.gold,
            list(setup.labels),
            view.labels_by_bucket,
            decision=eval_decision(config),
            ks=config.eval.metrics.ks,
        )

    # The pipeline's own scoring of the **same** cells (the champion's scores
    # restricted to the sample's texts), so every probe number has its
    # reference beside it on the same texts.
    sample_rows = np.asarray(sample.rows, dtype=np.int64)
    bucket_masks = _upr_bucket_masks(view)
    reference_block = None
    if sample_rows.size:
        reference_block = measure_upr_arm(
            np.asarray(setup.scores, dtype=np.float64)[sample_rows],
            sample.gold,
            decision=eval_decision(config),
            ks=config.eval.metrics.ks,
            shortlist_ms=config.eval.metrics.shortlist_ms,
            bins=decision.bins,
            bucket_masks=bucket_masks,
        )
    # The **fit side** of the safe-negative rule (CS-9b): the train split's
    # grid, scored with the very same arm, so no operating point is ever
    # fitted on the evaluated split.
    fit_sample: UprDecisionSample | None = None
    if decision.rule.enabled and setup.train_records:
        fit_sample = UprDecisionSample(
            texts=tuple(record.text for record in setup.train_records),
            labels=tuple(setup.labels),
            gold=build_label_matrix(setup.train_records, setup.labels),
            rows=tuple(range(len(setup.train_records))),
        )
    for model_config in decision.models:
        decider = build_upr_decider(model_config, config)
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
        block, arm_matrix = run_upr_decision_arm(
            sample,
            decider,
            arm,
            decision=eval_decision(config),
            ks=config.eval.metrics.ks,
            shortlist_ms=config.eval.metrics.shortlist_ms,
            max_questions_per_call=decision.max_questions_per_call,
            bins=decision.bins,
            bucket_masks=bucket_masks,
        )
        block["per_text"] = _per_text(arm_matrix)
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
        if decision.competence_control:
            block["competence_control"] = run_upr_competence_control(
                sample,
                decider,
                arm,
                n_pairs=decision.competence_pairs,
                max_questions_per_call=decision.max_questions_per_call,
                bins=decision.bins,
            )
        blind_matrix: np.ndarray | None = None
        if decision.mismatched_control:
            blind_block, blind_matrix = run_upr_decision_arm(
                sample,
                decider,
                arm,
                decision=eval_decision(config),
                ks=config.eval.metrics.ks,
                shortlist_ms=config.eval.metrics.shortlist_ms,
                max_questions_per_call=decision.max_questions_per_call,
                bins=decision.bins,
                mismatched=True,
                bucket_masks=bucket_masks,
            )
            blind_block["per_text"] = _per_text(blind_matrix)
            block["label_blind_counterpart"] = blind_block
        if decision.listwise.enabled:
            # The **listwise** second stage (CS-12): one choice question per
            # text over the round-1 top-N candidates. The label-blind
            # counterpart presents the *shifted* labels (the wrong pairing,
            # the same pipeline).
            descriptions = {
                name: entry.description
                for name, entry in _upr_label_catalog(config).items()
                if entry.description
            }
            n_labels = len(setup.labels)
            shifted = [setup.labels[(column + 1) % n_labels] for column in range(n_labels)]
            listwise_block, listwise_matrix = run_upr_listwise_stage(
                sample,
                decider,
                round_one=arm_matrix,
                option_labels=list(setup.labels),
                descriptions=descriptions,
                criteria_source=decision.listwise.criteria,
                n_candidates=decision.listwise.candidates,
                task=decision.listwise.instructions,
                question_text=decision.listwise.question_text,
                max_input_tokens=model_config.max_input_tokens,
                decision=eval_decision(config),
                ks=config.eval.metrics.ks,
                shortlist_ms=config.eval.metrics.shortlist_ms,
                bucket_masks=bucket_masks,
            )
            block["listwise"] = {
                key: value for key, value in listwise_block.items() if key != "measurement"
            }
            listwise_arm = {
                "arm": f"{model_config.name}-listwise",
                "state": arm.state,
                "question_type": "choice",
                "instructions": decision.listwise.instructions,
                "options": decision.listwise.candidates,
                "criteria": decision.listwise.criteria,
                "positive_option": None,
                "max_input_tokens": model_config.max_input_tokens,
                "label_blind": False,
                "n_cells": sample.n_cells,
                "n_calls_planned": listwise_block["n_calls_planned"],
                "model": {
                    "name": f"{model_config.name}-listwise",
                    "model_id": info.model_id,
                    "digest": info.digest,
                    "backend": info.backend,
                    "host": info.host,
                    "prompt_version": decision.listwise.prompt_version,
                    "extra": info.extra,
                },
                "cache": {
                    "hits": getattr(decider, "cache_hits", 0),
                    "misses": getattr(decider, "cache_misses", 0),
                    "stored": getattr(decider, "n_stored", 0),
                },
                "usage": dict(getattr(decider, "usage", {})),
                "measurement": listwise_block["measurement"],
            }
            listwise_arm["per_text"] = _per_text(listwise_matrix)
            listwise_blind = None
            if blind_matrix is not None:
                blind_listwise_block, blind_listwise_matrix = run_upr_listwise_stage(
                    sample,
                    decider,
                    round_one=blind_matrix,
                    option_labels=shifted,
                    descriptions=descriptions,
                    criteria_source=decision.listwise.criteria,
                    n_candidates=decision.listwise.candidates,
                    task=decision.listwise.instructions,
                    question_text=decision.listwise.question_text,
                    max_input_tokens=model_config.max_input_tokens,
                    decision=eval_decision(config),
                    ks=config.eval.metrics.ks,
                    shortlist_ms=config.eval.metrics.shortlist_ms,
                    bucket_masks=bucket_masks,
                )
                listwise_blind = {
                    **listwise_arm,
                    "arm": f"{model_config.name}-listwise-label-blind",
                    "label_blind": True,
                    "n_calls_planned": blind_listwise_block["n_calls_planned"],
                    "measurement": blind_listwise_block["measurement"],
                }
                listwise_blind["per_text"] = _per_text(blind_listwise_matrix)
                block["listwise"]["label_blind_counterpart"] = {
                    key: value
                    for key, value in blind_listwise_block.items()
                    if key != "measurement"
                }
            arms[f"{model_config.name}-listwise"] = listwise_arm
            if listwise_blind is not None:
                arms[f"{model_config.name}-listwise-label-blind"] = listwise_blind
        if decision.rule.enabled:
            block.update(
                run_upr_rule_stage(
                    sample,
                    decider,
                    arm,
                    arm_matrix=arm_matrix,
                    blind_matrix=blind_matrix,
                    fit_sample=fit_sample,
                    champion_scores=np.asarray(setup.scores, dtype=np.float64)[sample_rows],
                    fixed_cut=decision.rule.fixed_cut,
                    threshold_quantiles=decision.rule.threshold_quantiles,
                    margin_quantiles=decision.rule.margin_quantiles,
                    rerank_ks=decision.rule.rerank_ks,
                    decision=eval_decision(config),
                    ks=config.eval.metrics.ks,
                    shortlist_ms=config.eval.metrics.shortlist_ms,
                    max_questions_per_call=decision.max_questions_per_call,
                )
            )
        if decision.union:
            # The **union** as a number (CS-14): the per-row z-max of the
            # champion's and the decision model's scores, with the blind
            # control beside it. A measurement, not a component.
            champion_scores = np.asarray(setup.scores, dtype=np.float64)[sample_rows]
            union_block, union_matrix = run_upr_union_stage(
                arm_matrix,
                sample.gold,
                champion_scores=champion_scores,
                blind_matrix=blind_matrix,
                decision=eval_decision(config),
                ks=config.eval.metrics.ks,
                shortlist_ms=config.eval.metrics.shortlist_ms,
            )
            block["union"] = union_block
            arms[f"{model_config.name}-union"] = {
                "arm": f"{model_config.name}-union",
                "state": arm.state,
                "question_type": "union",
                "instructions": (
                    "the per-row z-max of the champion's and the decision model's scores"
                ),
                "options": 0,
                "criteria": {},
                "positive_option": None,
                "max_input_tokens": model_config.max_input_tokens,
                "label_blind": False,
                "n_cells": sample.n_cells,
                "n_calls_planned": 0,
                "model": {
                    "name": f"{model_config.name}-union",
                    "model_id": info.model_id,
                    "digest": info.digest,
                    "backend": info.backend,
                    "host": info.host,
                    "prompt_version": decision.prompt_version,
                    "extra": info.extra,
                },
                "cache": {"hits": 0, "misses": 0, "stored": 0},
                "usage": {},
                "measurement": measure_upr_arm(
                    union_matrix,
                    sample.gold,
                    decision=eval_decision(config),
                    ks=config.eval.metrics.ks,
                    shortlist_ms=config.eval.metrics.shortlist_ms,
                    bins=decision.bins,
                    bucket_masks=bucket_masks,
                    probability_scores=False,
                ),
            }
        arms[model_config.name] = block
        seconds = round(time.perf_counter() - started, 3)
        rows.extend(
            _decide_upr_rows(
                block,
                blind=None,
                run_id=run_id,
                split=setup.split,
                scope=decision.scope,
                n_texts=sample.n_texts,
                seconds=seconds,
            )
        )
        if decision.mismatched_control:
            rows.extend(
                _decide_upr_rows(
                    block,
                    blind=block["label_blind_counterpart"],
                    run_id=run_id,
                    split=setup.split,
                    scope=decision.scope,
                    n_texts=sample.n_texts,
                    seconds=seconds,
                )
            )

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    payload = build_upr_decision_report(
        sample,
        arms,
        run_id=run_id,
        split=setup.split,
        source=str(config.paths.data_raw),
        source_sha256=setup.source_sha256,
        scope=decision.scope,
        seed=decision.seed,
        config_block={
            "scope": decision.scope,
            "sample_size": decision.sample_size,
            "seed": decision.seed,
            "prompt_version": decision.prompt_version,
            "competence_control": decision.competence_control,
            "competence_pairs": decision.competence_pairs,
            "mismatched_control": decision.mismatched_control,
            "bins": decision.bins,
            "max_questions_per_call": decision.max_questions_per_call,
            "models": [model.model_dump() for model in decision.models],
            "decision_rule": eval_decision(config),
            "reference_arm": {
                "predictor": setup.predictor.describe(),
                "note": (
                    "the pipeline's own scoring of the probe's cells (the "
                    "champion's scores restricted to the sample's texts), so "
                    "every probe number has its reference beside it on the "
                    "same texts"
                ),
            },
            "notes": [
                "the decision models are frozen and zero-shot: they carry no "
                "UPR label, so nothing leaks into the fit side.",
                "a decision model only scores: it never relabels, curates or corrects the data.",
                "the non-gold labels' true positive rate is unmeasured, so "
                "these numbers are a reach and a cost, never a precision "
                "claim; the label-blind counterpart is the yardstick that "
                "says how much of the cost is the model's own label-blind "
                "signal.",
            ],
        },
        reference=reference_block,
        total_seconds=round(time.perf_counter() - started, 3),
    )
    (results_dir / "decisions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _append_command_rows(
        config,
        "decide-upr",
        columns=DECIDE_UPR_COLUMNS,
        key_columns=DECIDE_UPR_KEY_COLUMNS,
        rows=rows,
    )

    print(f"UPR decision probe (CS-9): {run_id}")
    print(
        f"  scope: {decision.scope}; {sample.n_texts} texts, {sample.n_labels} labels, "
        f"{sample.n_cells} cells ({int(sample.gold.sum())} gold)"
    )
    for name, block in arms.items():
        measurement = block["measurement"]
        print(
            f"  [{name}] {block['model']['model_id']} "
            f"(digest {block['model']['digest'][:12]}, {block['state']}/"
            f"{block['question_type']}): micro-F1 "
            f"{_fmt(measurement['decision']['micro_f1'], 4)}, nDCG@5 "
            f"{_fmt(measurement['ranking'].get('ndcg@5'), 4)}, shortlist R@50 "
            f"{_fmt(measurement['ranking'].get('shortlist_recall@50'), 4)}"
        )
        calibration = measurement["calibration"]
        if "ece" in calibration:
            print(
                f"      calibration: ECE {_fmt(calibration['ece'], 4)}, the "
                f"Brier {_fmt(calibration['brier'], 4)}; the compression: the "
                f"gold's mean probability {_fmt(measurement['compression']['gold_mean'], 4)} "
                f"against the background's "
                f"{_fmt(measurement['compression']['background_mean'], 4)}"
            )
        else:
            print(f"      calibration: {calibration.get('note', 'n/a')}")
    print(f"Report written to {results_dir / 'decisions.json'}")
    return 0


def _decide_upr_rows(
    block: dict[str, Any],
    *,
    blind: dict[str, Any] | None,
    run_id: str,
    split: str,
    scope: str,
    n_texts: int,
    seconds: float,
) -> list[dict[str, Any]]:
    """The table rows of one arm (the arm itself and its label-blind counterpart)."""
    rows: list[dict[str, Any]] = []
    pairs = [(block, False)]
    if blind is not None:
        pairs.append((blind, True))
    for source, label_blind in pairs:
        measurement = source["measurement"]
        decision = measurement["decision"]
        ranking = measurement["ranking"]
        compression = measurement["compression"]
        competence = (source.get("competence_control") or {}).get("measurement") or {}
        rows.append(
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "split": split,
                "model": source.get("arm", ""),
                "model_version": (source.get("model") or {}).get("model_id"),
                "state": source.get("state"),
                "question_type": source.get("question_type"),
                "scope": scope,
                "label_blind": "yes" if label_blind else "no",
                "micro-F1": _fmt(decision["micro_f1"], 4),
                "macro-F1": _fmt(decision["macro_f1"], 4),
                "sample-F1": _fmt(decision["sample_f1"], 4),
                "coverage": _fmt(decision["coverage"], 3),
                "nDCG@5": _fmt(ranking.get("ndcg@5"), 4),
                "shortlist R@50": _fmt(ranking.get("shortlist_recall@50"), 4),
                "hit@1": _fmt(measurement["gold"].get("hit@1"), 4),
                "gold_rank_p50": _fmt(measurement["gold"].get("rank_p50"), 4),
                "bg_mean": _fmt(measurement["background"].get("mean"), 4),
                "bg_rank_p50": _fmt(measurement["background"].get("rank_p50"), 4),
                "ece": _fmt(measurement["calibration"].get("ece"), 4),
                "brier": _fmt(measurement["calibration"].get("brier"), 4),
                "gold_prob_mean": _fmt(compression.get("gold_mean"), 4),
                "bg_prob_mean": _fmt(compression.get("background_mean"), 4),
                "comp_auc": _fmt(competence.get("pair_auc"), 4),
                "n_texts": n_texts,
                "n_cells": source.get("n_cells", ""),
                "n_calls": source.get("n_calls_planned", ""),
                "seconds": seconds,
            }
        )
    return rows


@dataclass(frozen=True)
class EvalSetup:
    """The evaluated split, the predictor and its score matrix (shared wiring).

    The ``evaluate`` command and the text-processing track's ``analyze-upr``
    command share this wiring, so the track's whole-text **reference arm** *is*
    the pipeline's own scoring of the same split — not a re-implementation of
    it. That is what makes the field-by-field guard (the reference arm against
    the matching ``evaluate`` run) possible at all.

    Attributes:
        split: The evaluated split's name (``val`` or ``test``).
        records: The evaluated records (the split's order).
        train_records: The train split's records (the only context a
            predictor may see).
        labels: The label space (the sorted unique labels).
        buckets: The ``label -> bucket`` mapping (head / torso / tail /
            unseen).
        predictor: The configured predictor.
        artifact: The embedding artifact (``None`` for embedding-free
            predictors).
        label_artifact: The zero-shot label artifact (``None``).
        scores: The predictor's ``(n_texts, n_labels)`` score matrix.
        gold: The bool gold matrix of the same shape.
        source_sha256: The dataset's sha256 (the splits' provenance).
        provenances: The split files' provenance (per split).
        ids_by_split: The split id lists (per split).
        dataset_counts: The full-dataset label counts (the bucket sizes).
        train_counts: The train split's label counts (the unseen labels).
    """

    split: str
    records: list[Record]
    train_records: list[Record]
    labels: list[str]
    buckets: dict[str, str]
    predictor: Any
    artifact: EmbeddingArtifact | None
    label_artifact: LabelEmbeddingArtifact | None
    scores: np.ndarray
    gold: np.ndarray
    source_sha256: str
    provenances: dict[str, dict[str, Any]]
    ids_by_split: dict[str, list[str]]
    dataset_counts: Counter[str]
    train_counts: Counter[str]
    context: PredictorContext
    query_vectors: np.ndarray | None


def _cmd_rerank_upr(config: AppConfig) -> int:
    """Run rerank-upr: the track's zero-shot reranker probe (CS-15).

    The rerankers are **scorers over a fixed candidate set**: they re-order an
    existing set of labels, and the run compares their ordering with **the
    ordering that produced the set** (the decision model's own probabilities,
    the union's own z-max, the champion's own scores). A reranker can
    re-order a set, never enlarge it, so the candidate set is the experiment's
    independent variable and the run's **ceiling** is measured beside every
    arm.

    Writes ``results/<run_id>/{config.yaml, rerank.json}``: the candidate set
    and its ceiling, the incumbent's own numbers, every arm's measurement
    (the decision view, the ranking view, the per-bucket view), the
    **label-blind counterpart** beside every arm, the **paired** differences
    over the texts and the **cost** block (the pairs per text, the pairs per
    second, the wall time and the peak GPU memory).
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.upr_analysis is None:
        raise ValueError(
            "the upr_analysis: section (the rerankers and the candidate set) is "
            "required for the `rerank-upr` command"
        )
    rerank = config.upr_analysis.rerank
    if not rerank.enabled or not rerank.models:
        raise ValueError(
            "the upr_analysis.rerank: section is required for the `rerank-upr` "
            "command: it carries the pinned rerankers and the candidate set"
        )
    if config.eval is None:
        raise ValueError(
            "the eval: section (the decision rule) is required for the `rerank-upr` command"
        )
    setup = _prepare_evaluation(config, config.eval.split)
    view = build_upr_view(
        setup.records,
        setup.labels,
        setup.buckets,
        config.upr_analysis.length_buckets,
    )
    gold = np.asarray(view.gold, dtype=bool)
    labels = list(setup.labels)
    texts = [record.text for record in setup.records]
    bucket_masks = _upr_bucket_masks(view)
    decision_rule = eval_decision(config)

    # The **reference arm** and the guard: the champion's own scoring is the
    # main pipeline's scoring, so it must reproduce the matching `evaluate`
    # run field by field (the track's guard, the same one the `analyze-upr`
    # command applies).
    reference_block = arm_block(
        view,
        setup.scores,
        decision=decision_rule,
        ks=config.eval.metrics.ks,
        shortlist_ms=config.eval.metrics.shortlist_ms,
        arm=BASELINE_ARM,
        predictor=setup.predictor.describe(),
        chunker="full_text",
        aggregation=None,
    )
    guard = _check_reference_arm(config, config.upr_analysis.reference_eval_run, reference_block)

    champion = np.asarray(setup.scores, dtype=np.float64)
    sources = RerankSources(
        champion=champion,
        decision=np.zeros_like(champion),
        union=np.zeros_like(champion),
    )
    needs_decision = rerank.candidate_set in ("decision", "union")
    decision_matrix: np.ndarray | None = None
    if needs_decision:
        decision_matrix = _rerank_decision_matrix(config, setup, view)
        sources = RerankSources(
            champion=champion,
            decision=decision_matrix,
            union=z_max_union(champion, decision_matrix),
        )
    source = candidate_set_matrix(sources, rerank.candidate_set)
    columns = candidate_columns(source, rerank.candidate_k)
    n_predicted = _rerank_n_predicted(decision_rule, len(texts))
    ceiling = candidate_ceiling(
        columns,
        gold,
        n_predicted=n_predicted,
        labels_by_bucket=view.labels_by_bucket,
        bucket_masks=bucket_masks,
    )
    incumbent_measurement = measure_upr_arm(
        source,
        gold,
        decision=decision_rule,
        ks=config.eval.metrics.ks,
        shortlist_ms=config.eval.metrics.shortlist_ms,
        bucket_masks=bucket_masks,
        probability_scores=(rerank.candidate_set == "decision"),
    )
    incumbent_block = per_text_block(
        source,
        gold,
        apply_decision_rule(
            source,
            rule=decision_rule["rule"],
            k=decision_rule.get("k"),
            threshold=decision_rule.get("threshold"),
            min_score=decision_rule.get("min_score"),
        ),
        labels,
        view.labels_by_bucket,
        ks=config.eval.metrics.ks,
    )

    catalog = _upr_label_catalog(config)
    descriptions = {name: entry.description for name, entry in catalog.items() if entry.description}

    arms: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for model_config in rerank.models:
        client = build_reranker(model_config, config)
        for variant in rerank.label_documents:
            documents = label_documents(labels, descriptions, variant)
            blind_documents = shifted_documents(documents)
            for order in rerank.orders:
                if model_config.kind == KIND_LISTWISE and order != ORDER_TEXT_FIRST:
                    logger.info(
                        "Skipping the %r order for the listwise model %r: its API "
                        "takes one query and many documents",
                        order,
                        model_config.name,
                    )
                    continue
                for blind, side in ((False, documents), (True, blind_documents)):
                    if blind and not rerank.label_blind:
                        continue
                    block, fused = run_upr_rerank_arm(
                        texts=texts,
                        gold=gold,
                        columns=columns,
                        incumbent=source,
                        reranker=client,
                        model_config=model_config,
                        documents=side,
                        label_document=variant,
                        order=order,
                        label_blind=blind,
                        decision=decision_rule,
                        ks=config.eval.metrics.ks,
                        shortlist_ms=config.eval.metrics.shortlist_ms,
                        bucket_masks=bucket_masks,
                        benchmark_requests=config.upr_analysis.rerank.benchmark_requests,
                    )
                    block["label_document"] = variant
                    block["comparison_vs_incumbent"] = paired_arm_comparison(
                        fused,
                        source,
                        gold,
                        labels,
                        view.labels_by_bucket,
                        decision=decision_rule,
                        ks=config.eval.metrics.ks,
                        resamples=rerank.bootstrap.resamples,
                        seed=rerank.bootstrap.seed if rerank.bootstrap.seed is not None else 42,
                    )
                    block["per_text"] = per_text_block(
                        fused,
                        gold,
                        apply_decision_rule(
                            fused,
                            rule=decision_rule["rule"],
                            k=decision_rule.get("k"),
                            threshold=decision_rule.get("threshold"),
                            min_score=decision_rule.get("min_score"),
                        ),
                        labels,
                        view.labels_by_bucket,
                        ks=config.eval.metrics.ks,
                    )
                    info = client.info
                    block["model_info"] = {
                        "name": model_config.name,
                        "model_id": info.model_id,
                        "revision": info.revision,
                        "kind": info.kind,
                        "backend": info.backend,
                        "extra": info.extra,
                    }
                    arms[block["arm"]] = block
                    rows.append(
                        _rerank_upr_row(
                            block,
                            run_id=run_id,
                            split=setup.split,
                            candidate_set=rerank.candidate_set,
                            n_texts=len(texts),
                            seconds=round(time.perf_counter() - started, 3),
                        )
                    )

    incumbent_arm = {
        "arm": f"incumbent-{rerank.candidate_set}",
        "candidate_set": rerank.candidate_set,
        "measurement": incumbent_measurement,
        "per_text": incumbent_block,
        "note": (
            "the incumbent: the ordering that produced the candidate set (the "
            "decision model's own probabilities, the union's own z-max or the "
            "champion's own scores). Every arm of this run is compared against "
            "it, with the candidate set held fixed."
        ),
    }
    arms[incumbent_arm["arm"]] = incumbent_arm

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    payload = build_upr_rerank_report(
        run_id=run_id,
        split=setup.split,
        source=str(config.paths.data_raw),
        source_sha256=setup.source_sha256,
        n_texts=len(texts),
        n_labels=len(labels),
        candidate_set=rerank.candidate_set,
        candidate_k=rerank.candidate_k,
        arms=arms,
        incumbent=incumbent_measurement,
        ceiling=ceiling,
        guard=guard,
        reference=reference_block,
        config_block={
            "candidate_set": rerank.candidate_set,
            "candidate_k": rerank.candidate_k,
            "label_documents": list(rerank.label_documents),
            "orders": list(rerank.orders),
            "label_blind": rerank.label_blind,
            "models": [model.model_dump(mode="python") for model in rerank.models],
            "decision_rule": decision_rule,
            "bootstrap": rerank.bootstrap.model_dump(mode="python"),
            "decision_model": (
                {
                    "name": config.upr_analysis.decision.models[0].name,
                    "model_id": config.upr_analysis.decision.models[0].model_id,
                    "digest": config.upr_analysis.decision.models[0].digest,
                }
                if needs_decision and config.upr_analysis.decision.models
                else None
            ),
            "notes": [
                "the rerankers are frozen and zero-shot: they carry no UPR "
                "label, and the label side is the frozen catalog (label-side "
                "knowledge, the same kind the zero-shot label vectors use).",
                "a reranker re-orders a candidate set, never enlarges it: the "
                "candidate set is held fixed and only the ordering changes.",
                "the comparison is paired over the texts, and the label-blind "
                "counterpart is the yardstick that says whether a gain is the "
                "model's label reading.",
                "the raw data is never touched: a reranker only scores.",
            ],
        },
        total_seconds=round(time.perf_counter() - started, 3),
    )
    (results_dir / "rerank.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _append_command_rows(
        config,
        "rerank-upr",
        columns=RERANK_UPR_COLUMNS,
        key_columns=RERANK_UPR_KEY_COLUMNS,
        rows=rows,
    )

    print(f"UPR rerank probe (CS-15): {run_id}")
    print(
        f"  candidate set: {rerank.candidate_set} (k={rerank.candidate_k}); "
        f"{len(texts)} texts, {len(labels)} labels"
    )
    print(
        f"  the set's ceiling: micro-F1 {ceiling['micro_f1_ceiling']:.4f} "
        f"(the gold cells inside the set: {ceiling['n_gold_cells_in_set']}/"
        f"{ceiling['n_gold_cells']})"
    )
    for name, block in arms.items():
        if name.startswith("incumbent-"):
            measurement = block["measurement"]
            print(
                f"  [incumbent] micro-F1 {measurement['decision']['micro_f1']:.4f}, "
                f"nDCG@5 {measurement['ranking'].get('ndcg@5', 0.0):.4f}"
            )
            continue
        measurement = block["measurement"]
        comparison = block["comparison_vs_incumbent"]["pooled"]["micro_f1"]
        cost = block["cost"]
        print(
            f"  [{name}] micro-F1 {measurement['decision']['micro_f1']:.4f}, "
            f"nDCG@5 {measurement['ranking'].get('ndcg@5', 0.0):.4f}; "
            f"vs the incumbent {comparison['delta']:+.4f} "
            f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]; "
            f"{cost['n_pairs']} pairs, {cost['seconds']:.1f}s, "
            f"{cost['peak_gpu_mb']:.0f} MB"
        )
    print(f"Report written to {results_dir / 'rerank.json'}")
    return 0


def _rerank_n_predicted(decision_rule: Mapping[str, Any], n_texts: int) -> int:
    """How many cells the decision rule predicts (the top-k rule's k per text)."""
    if decision_rule.get("rule") != "top_k":
        raise ValueError(
            "the rerank probe's ceiling needs the top_k decision rule (the "
            f"rule {decision_rule.get('rule')!r} predicts a variable number of cells)"
        )
    return int(decision_rule.get("k") or 0) * n_texts


def _rerank_decision_matrix(config: AppConfig, setup: EvalSetup, view: UprView) -> np.ndarray:
    """The decision model's score matrix (the cache hits of CS-9b's grid).

    The candidate sets of the ``decision`` and ``union`` runs come from the
    decision model, so its grid is scored here with the very same arm (and
    the very same cache keys) as CS-9b's: the calls are cache hits, and the
    artifact records the cache's counters.

    Raises:
        ValueError: If the decision section carries no model.
    """
    decision = config.upr_analysis.decision if config.upr_analysis else None
    if decision is None or not decision.models:
        raise ValueError(
            "the upr_analysis.decision: section (a pinned decision model) is "
            "required for the 'decision' and 'union' candidate sets: they are "
            "the decision model's candidates"
        )
    model_config = decision.models[0]
    decider = build_upr_decider(model_config, config)
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
    sample = UprDecisionSample(
        texts=tuple(record.text for record in setup.records),
        labels=tuple(setup.labels),
        gold=view.gold,
        rows=tuple(range(view.n_texts)),
    )
    matrix, _cost = score_upr_grid(
        sample,
        decider,
        arm,
        max_questions_per_call=decision.max_questions_per_call,
    )
    return matrix


def build_reranker(model_config: Any, config: AppConfig) -> Any:
    """Build the reranker's client (the CLI's wiring seam).

    The Hugging Face client behind the :class:`~cluster_rag.rerank.base.Reranker`
    port, with the rerankers' **own** disk cache wrapped around it (its own
    root under ``cache/``, never the embedding cache), so a re-run is
    reproducible and never re-scores a pair.
    """
    from cluster_rag.rerank.cache import CachedReranker
    from cluster_rag.rerank.hf import HFReranker

    client = HFReranker(model_config)
    shape = {
        "max_length": model_config.max_length,
        "max_documents": model_config.max_documents,
        "max_query_length": model_config.max_query_length,
        "max_doc_length": model_config.max_doc_length,
        "template": model_config.template,
        "padding_side": model_config.padding_side,
        "batch_size": model_config.batch_size,
    }
    return CachedReranker(client, config.paths.cache, shape=shape)


def _rerank_upr_row(
    block: dict[str, Any],
    *,
    run_id: str,
    split: str,
    candidate_set: str,
    n_texts: int,
    seconds: float,
) -> dict[str, Any]:
    """One row of the rerank probe's comparison table."""
    measurement = block["measurement"]
    comparison = block["comparison_vs_incumbent"]["pooled"]
    cost = block["cost"]
    info = block.get("model_info", {})
    probe = cost.get("throughput_probe") or {}
    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "split": split,
        "candidate_set": candidate_set,
        "model": info.get("model_id", block.get("model", "")),
        "model_version": str(info.get("revision", ""))[:12],
        "kind": block.get("kind", ""),
        "order": block.get("order", ""),
        "label_document": block.get("label_document", ""),
        "label_blind": block.get("label_blind", False),
        "micro-F1": _fmt(measurement["decision"]["micro_f1"], 4),
        "macro-F1": _fmt(measurement["decision"]["macro_f1"], 4),
        "nDCG@5": _fmt(measurement["ranking"].get("ndcg@5"), 4),
        "shortlist R@50": _fmt(measurement["ranking"].get("shortlist_recall@50"), 4),
        "hit@1": _fmt(measurement["gold"].get("hit@1"), 4),
        "d_micro-F1": _fmt(comparison["micro_f1"]["delta"], 4),
        "d_micro-F1_lo": _fmt(comparison["micro_f1"]["ci_low"], 4),
        "d_micro-F1_hi": _fmt(comparison["micro_f1"]["ci_high"], 4),
        "d_nDCG@5": _fmt(comparison["ndcg"]["delta"], 4),
        "d_nDCG@5_lo": _fmt(comparison["ndcg"]["ci_low"], 4),
        "d_nDCG@5_hi": _fmt(comparison["ndcg"]["ci_high"], 4),
        "n_pairs": cost.get("n_pairs"),
        "pairs_per_text": _fmt(cost.get("n_pairs_per_text"), 2),
        "pairs_per_second": _fmt(probe.get("pairs_per_second"), 1),
        "seconds": _fmt(cost.get("seconds"), 2),
        # the peak memory of the arm's own forward passes, or the probe's when
        # the arm was served entirely from the cache (the probe always runs)
        "peak_gpu_mb": _fmt(
            max(float(cost.get("peak_gpu_mb") or 0.0), float(probe.get("peak_gpu_mb") or 0.0)),
            1,
        ),
        "n_texts": n_texts,
    }


def _prepare_evaluation(config: AppConfig, split: str) -> EvalSetup:
    """Load the split, build the predictor from train context, score the split.

    The shared wiring of the ``evaluate`` command and of the track's
    ``analyze-upr`` command: the dataset and the fixed split ids (with the
    provenance sha check — never re-split silently), the predictor's
    construction context (**train** only), the embedding artifacts (validated
    and id-aligned) and the score matrix.

    Args:
        config: The resolved app config (the ``eval:`` section is required).
        split: The split to evaluate (``val`` or ``test``).

    Returns:
        The :class:`EvalSetup` of the run.

    Raises:
        ValueError: If the ``eval:`` section is missing, an embedding run is
            not set for an embedding-dependent predictor, the splits do not
            match the dataset, or the predictor's score matrix has the wrong
            shape.
    """
    if config.eval is None:
        raise ValueError(
            "the eval: section (predictor, decision) is required for the `evaluate` command"
        )
    eval_config = config.eval

    full_report = load_jsonl(config.paths.data_raw)
    if full_report.errors:
        logger.warning("%d malformed line(s) dropped before evaluating", full_report.n_invalid)
    source_sha = file_sha256(config.paths.data_raw)

    ids_by_split: dict[str, list[str]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    for name in (split, "train"):
        ids, provenance = load_split_ids(config.paths.data_splits, name)
        recorded = provenance.get("source_sha256")
        if recorded and recorded != source_sha:
            raise ValueError(
                f"{name}_ids.json was built from a different data source (sha256 mismatch): "
                "the fixed splits and the dataset disagree; do not re-split silently."
            )
        ids_by_split[name] = ids
        provenances[name] = provenance

    eval_records = select_records_by_ids(full_report.records, ids_by_split[split])
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
    context = PredictorContext(
        labels=tuple(labels),
        train_records=tuple(train_records),
        train_document_vectors=train_vectors,
        label_vectors=label_artifact.vectors if label_artifact is not None else None,
    )
    predictor = build_predictor(
        eval_config.predictor.name,
        eval_config.predictor.params,
        context,
        seed=config.seed,
    )
    texts = [record.text for record in eval_records]
    query_vectors: np.ndarray | None = None
    if artifact is not None:
        query_vectors = artifact.select("query", ids_by_split[split])
        scores = predictor.score(texts, query_vectors=query_vectors)
    else:
        scores = predictor.score(texts)
    expected_shape = (len(eval_records), len(labels))
    if scores.shape != expected_shape:
        raise ValueError(
            f"predictor {eval_config.predictor.name!r} returned scores of shape {scores.shape}, "
            f"expected {expected_shape} (n_eval_records, n_labels)"
        )

    return EvalSetup(
        split=split,
        records=eval_records,
        train_records=train_records,
        labels=labels,
        buckets=buckets,
        predictor=predictor,
        artifact=artifact,
        label_artifact=label_artifact,
        scores=scores,
        gold=build_label_matrix(eval_records, labels),
        source_sha256=source_sha,
        provenances=provenances,
        ids_by_split=ids_by_split,
        dataset_counts=dataset_counts,
        train_counts=train_counts,
        context=context,
        query_vectors=query_vectors,
    )


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

    The wiring itself lives in :func:`_prepare_evaluation`, which the
    text-processing track's ``analyze-upr`` command shares — so the track's
    whole-text reference arm is this command's own scoring.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.eval is None:
        raise ValueError(
            "the eval: section (predictor, decision) is required for the `evaluate` command"
        )
    eval_config = config.eval
    setup = _prepare_evaluation(config, split or eval_config.split)

    decision = eval_config.decision
    predicted = apply_decision_rule(
        setup.scores,
        rule=decision.rule,
        k=decision.k,
        threshold=decision.threshold,
        min_score=decision.min_score,
    )
    report = build_label_report(
        setup.scores,
        setup.gold,
        setup.labels,
        setup.buckets,
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
        "split": setup.split,
        "source": str(config.paths.data_raw),
        "source_sha256": setup.source_sha256,
        "split_provenance": setup.provenances,
        "n_records": len(setup.records),
        "n_records_train_context": len(setup.train_records),
        "n_labels": len(setup.labels),
        "seed": config.seed,
        "predictor": setup.predictor.describe(),
        "embedding_run": setup.artifact.lineage if setup.artifact is not None else None,
        "label_embedding_run": (
            setup.label_artifact.lineage if setup.label_artifact is not None else None
        ),
        "decision": decision.model_dump(),
        "metrics_config": eval_config.metrics.model_dump(),
        "buckets": {
            "thresholds": {
                "head_min": config.stats.frequency_head_min,
                "torso_min": config.stats.frequency_torso_min,
            },
            "n_labels_per_bucket": {
                name: len(subset) for name, subset in labels_by_bucket(setup.buckets).items()
            },
            "labels_by_bucket": labels_by_bucket(setup.buckets),
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
                "split": setup.split,
                "predictor": eval_config.predictor.name,
                "embedding": (
                    eval_config.embedding_run_id.removeprefix("upr-embed-")
                    if setup.artifact is not None
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
                "n_texts": len(setup.records),
                "seconds": payload["total_seconds"],
            }
        ],
    )

    ranking = overall["ranking"]
    print(
        f"Evaluated {len(setup.records)} record(s) [{setup.split}] with predictor "
        f"'{eval_config.predictor.name}' ({len(setup.labels)} label columns)"
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


def _upr_arm_rows(
    arms: Mapping[str, dict[str, Any]],
    *,
    run_id: str,
    split: str,
    seconds: float,
) -> list[dict[str, Any]]:
    """One table row per arm of the run (the track's UPR table shape)."""
    rows: list[dict[str, Any]] = []
    for name, block in arms.items():
        overall = block["metrics"]["overall"]
        chunks = block.get("chunks_per_text") or {}
        rows.append(
            {
                "run_id": run_id,
                "date": date.today().isoformat(),
                "split": split,
                "arm": name,
                "predictor": (block.get("predictor") or {}).get("name"),
                "chunker": block.get("chunker"),
                "aggregation": block.get("aggregation"),
                "micro-F1": _fmt(overall["f1"]["micro_f1"], 4),
                "macro-F1": _fmt(overall["f1"]["macro_f1"], 4),
                "sample-F1": _fmt(overall["f1"]["sample_f1"], 4),
                "coverage": _fmt(overall["coverage"]["coverage"], 3),
                "P@5": _fmt(overall["ranking"].get("precision@5"), 4),
                "R@10": _fmt(overall["ranking"].get("recall@10"), 4),
                "nDCG@5": _fmt(overall["ranking"].get("ndcg@5"), 4),
                "shortlist R@50": _fmt(overall["shortlist"].get("recall@50"), 4),
                "hit@1": _fmt(block["gold"].get("hit@1"), 4),
                "hit@10": _fmt(block["gold"].get("hit@10"), 4),
                "gold_rank_p50": _fmt(block["gold"].get("rank_p50"), 4),
                "bg_rank_p50": _fmt(block["background"].get("rank_p50"), 4),
                "bg_mean": _fmt(block["background"].get("mean"), 4),
                "bg_mean_delta": _fmt(block["shift"].get("background_mean_delta"), 4),
                "bg_rank_delta": _fmt(block["shift"].get("background_rank_p50_delta"), 4),
                "chunks_per_text": _fmt(chunks.get("mean"), 2),
                "n_texts": overall["n_records"],
                "seconds": seconds,
            }
        )
    return rows


def _check_reference_arm(
    config: AppConfig,
    reference_run_id: str | None,
    reference_block: dict[str, Any],
) -> dict[str, Any] | None:
    """The guard: the whole-text arm must reproduce the ``evaluate`` run.

    The UPR analogue of ``tools/verify_analysis_baseline.py``: the track's
    whole-text reference arm is the main pipeline's own scoring, so it must
    match the matching ``evaluate`` run's report **field by field**. A
    mismatch means the run's numbers cannot be compared with the pipeline's.

    Args:
        config: The resolved app config (the results folder).
        reference_run_id: The ``evaluate`` run to compare with (``None``
            disables the check).
        reference_block: The reference arm's block (its ``metrics``).

    Returns:
        The guard's outcome (recorded in the artifact), or ``None`` when the
        check was not requested.

    Raises:
        ValueError: If the check fails — a mismatch is never a warning.
    """
    if not reference_run_id:
        return None
    path = config.paths.results / reference_run_id / "metrics.json"
    if not path.is_file():
        raise ValueError(
            f"the reference evaluate run {reference_run_id!r} has no metrics artifact "
            f"({path}); the guard cannot check the whole-text arm"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("report")
    actual = reference_block["metrics"]["overall"]
    differences: list[str] = []
    if expected is None:
        differences.append("the reference run has no report block")
    else:
        if expected.get("overall") != actual:
            differences.append("the overall report differs")
        if expected.get("per_bucket") != reference_block["metrics"]["per_bucket"]:
            differences.append("the per-bucket report differs")
    if payload.get("n_records") != actual.get("n_records"):
        differences.append("the record count differs")
    if payload.get("n_labels") != actual.get("n_labels"):
        differences.append("the label count differs")
    if differences:
        raise ValueError(
            f"the whole-text reference arm does not reproduce the evaluate run "
            f"{reference_run_id!r}: {'; '.join(differences)}"
        )
    return {
        "reference_run": reference_run_id,
        "checked": ["overall", "per_bucket", "n_records", "n_labels"],
        "matches": True,
    }


def _upr_chunker(
    config: AppConfig,
    embedder: Any,
    spec: ChunkerSpec,
    preprocess_version: str,
    *,
    threshold: float | None = None,
) -> tuple[Any, Any]:
    """A chunker and the (cached) embedder of its chunks.

    The chunking version goes into the **existing** cache version slot (the
    chunker's own ``cache_version``), so the cache key layout never changes
    and two differently configured chunkers never share cached chunks. The
    semantic chunker finds its boundaries from the **atom** embeddings, so it
    takes the ``Embedder`` port: the atoms are cached under the atoms
    chunker's own version (so a swept threshold re-uses them).

    Args:
        config: The resolved app config (the cache root).
        embedder: The raw backend (the CLI wires it in).
        spec: The chunker's config-facing spec.
        preprocess_version: The preprocessing version the texts carry.
        threshold: The semantic chunker's threshold (overrides the spec's).

    Returns:
        The chunker and the embedder of its chunks.
    """
    params = dict(spec.params)
    if threshold is not None:
        params["threshold"] = threshold
    if spec.name == "semantic":
        atom_chunker = build_chunker("atoms", {"split_clauses": False})
        atom_version = atom_chunker.cache_version(preprocess_version)
        chunker = build_chunker(
            "semantic", params, embedder=_cached_embedder(embedder, config, atom_version)
        )
    else:
        chunker = build_chunker(spec.name, params)
    return chunker, _cached_embedder(embedder, config, chunker.cache_version(preprocess_version))


def _check_embedder_matches_artifact(config: AppConfig, setup: EvalSetup) -> Any:
    """Build the chunked arms' embedder and check it against the index's space.

    The chunked arms' chunk vectors must live in the **same space** as the
    embedding artifact the index is built from (the train document vectors and
    the label vectors). A mismatch would silently score the chunks against a
    different geometry, so it is refused with the fix named.

    Returns:
        The (uncached) backend.
    """
    if config.embed is None:
        raise ValueError(
            "the chunked arms need an embed: section (the chunks are encoded by the "
            "run): set embed.backend / embed.model_id / embed.prefixes to the "
            "embedding run's model"
        )
    if setup.artifact is None:
        raise ValueError(
            "the chunked arms need an embedding run (eval.embedding_run_id): the "
            "chunks are compared with the index built from it"
        )
    embedder = build_embedder(config.embed)
    meta = setup.artifact.meta["embedder"]
    if embedder.info.model_id != meta["model_id"]:
        raise ValueError(
            f"the embed: section's model {embedder.info.model_id!r} is not the "
            f"embedding run's model {meta['model_id']!r}: the chunks would be "
            "encoded in a different space than the index"
        )
    prefixes = dict(config.embed.prefixes.model_dump())
    if prefixes != dict(meta["prefixes"]):
        raise ValueError(
            f"the embed: section's prefixes {prefixes} are not the embedding run's "
            f"prefixes {dict(meta['prefixes'])}: the chunks would be encoded with a "
            "different prompt than the index's vectors"
        )
    return embedder


def _upr_label_texts(setup: EvalSetup) -> list[str]:
    """The label texts the zero-shot vectors were built from.

    The label artifact's own metadata names the variant and the template (and,
    since the CS-11 catalog, records the **rendered** texts), so the label
    side is reconstructed from the artifact that produced the vectors (never
    from a guess).
    """
    if setup.label_artifact is None:
        raise ValueError(
            "the chunked arms' label side needs the zero-shot label vectors: set "
            "eval.label_embedding_run_id to a prior upr-labelembed-* run"
        )
    label_texts = setup.label_artifact.meta["label_texts"]
    rendered = label_texts.get("rendered_texts")
    if rendered:
        if len(rendered) != len(setup.labels):
            raise ValueError(
                f"the label artifact carries {len(rendered)} rendered texts for "
                f"{len(setup.labels)} labels: the label spaces disagree"
            )
        return [str(text) for text in rendered]
    variant = label_texts.get("variant", "name")
    template = label_texts.get("template")
    if variant == "name":
        return list(setup.labels)
    if not template:
        raise ValueError(
            f"the label artifact's variant is {variant!r} but it carries no template: "
            "the label texts cannot be reconstructed"
        )
    return [template.replace("{label}", label) for label in setup.labels]


def _upr_family_context(
    config: AppConfig,
    setup: EvalSetup,
    label_embedding_run_id: str,
) -> PredictorContext:
    """A predictor context with a **different** zero-shot label artifact.

    The label-side transform (the bare label name vs the config template) is a
    config choice of the *label* side, so a family may point at its own prior
    label run. The artifact is loaded and validated exactly like the run's own
    (the same model, the same dimension, the same label space).
    """
    if setup.artifact is None:
        raise ValueError("the family's label run needs the primary embedding run")
    label_artifact = load_label_embedding_artifact(
        config.paths.results,
        label_embedding_run_id,
        labels=setup.labels,
        source_sha256=setup.source_sha256,
        model_id=setup.artifact.meta["embedder"]["model_id"],
        dimension=setup.artifact.meta["embedder"]["dimension"],
    )
    return PredictorContext(
        labels=setup.context.labels,
        train_records=setup.context.train_records,
        train_document_vectors=setup.context.train_document_vectors,
        label_vectors=label_artifact.vectors,
    )


def _upr_chunked_arms(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    analysis_config: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Score every configured chunked arm family (CS-8b).

    The arms are the track's **ladder** on UPR: the text side chunked, the
    chunk-level scores folded with the family's aggregation rules. The
    families are the two the reviewer named — the **pair-scoring** family
    (the text's chunks against the zero-shot label vectors) and the
    **retrieval** family (each chunk is a query against the train-only index)
    — plus the **symmetric** family (the label side chunked too).

    The pairing is by construction: every arm is scored in the same run as
    the whole-text reference arm, on the same texts, with the same predictor
    and the same decision rule.

    Returns:
        The arms' blocks (keyed by the arm's name) and the families' metadata.
    """
    if not analysis_config.arms:
        return {}, {}
    embedder = _check_embedder_matches_artifact(config, setup)
    spec = build_preprocess_spec(config.preprocess)
    texts = [record.text for record in setup.records]
    arms: dict[str, dict[str, Any]] = {}
    families: dict[str, Any] = {}

    for family in analysis_config.arms:
        threshold = family.semantic_threshold
        semantic_selection = None
        if family.chunker.name == "semantic" and analysis_config.semantic_thresholds:
            threshold, semantic_selection = _select_semantic_threshold(
                config,
                setup,
                analysis_config,
                embedder,
                family.chunker,
            )
        chunker, chunk_embedder = _upr_chunker(
            config, embedder, family.chunker, spec.version, threshold=threshold
        )
        if family.late_pooling is not None:
            text_side, pooling_reports = encode_late_chunks(
                embedder,
                chunker,
                texts,
                "query",
                pooling=family.late_pooling,
                reference_vectors=setup.query_vectors,
            )
            family_meta_late = {
                "late_pooling": family.late_pooling,
                "token_coverage_mean": float(
                    np.mean([report["token_coverage"] for report in pooling_reports])
                )
                if pooling_reports
                else 0.0,
                "n_chunks_without_token": sum(
                    report["n_chunks_without_token"] for report in pooling_reports
                ),
                "pooling_consistency": {
                    "cosine_with_the_model_embedding_mean": float(
                        np.mean(
                            [
                                report["cosine_with_the_model_embedding"]
                                for report in pooling_reports
                                if "cosine_with_the_model_embedding" in report
                            ]
                        )
                    )
                    if any(
                        "cosine_with_the_model_embedding" in report for report in pooling_reports
                    )
                    else None,
                    "note": (
                        "the cosine between the text's pooled token vector and the "
                        "model's own embedding of the same text: the check that the "
                        "token-level path lives in the model's own space"
                    ),
                },
                "note": (
                    "the late chunking: the text is embedded once at the token level "
                    "and the chunks are pooled out of its tokens. It needs a backend "
                    "that returns token-level embeddings (HF only), so the arm is "
                    "within-model: the early-chunked arm of the same run is its "
                    "reference"
                ),
            }
        else:
            text_side = encode_chunks(chunk_embedder, chunker, texts, "query")
            family_meta_late = None
        label_side = None
        if family.label_chunker is not None:
            label_chunker, label_embedder = _upr_chunker(
                config, embedder, family.label_chunker, spec.version
            )
            label_side = encode_chunks(
                label_embedder, label_chunker, _upr_label_texts(setup), "label"
            )
        # The family's own label vectors (the label-side transform: the bare
        # name vs the config template is a *config* choice, and this knob lets
        # one run compare the two variants side by side).
        family_context = setup.context
        if family.label_embedding_run_id:
            family_context = _upr_family_context(config, setup, family.label_embedding_run_id)
        # The family's own whole-text score: the second input of the
        # local+global blend (exp06). It is the *same geometry* as the arm's
        # chunk-level scores, so the blend mixes two scales of one signal.
        family_predictor = build_predictor(
            family.predictor, family.params, family_context, seed=config.seed
        )
        family_global = (
            np.asarray(
                family_predictor.score(texts, query_vectors=setup.query_vectors),
                dtype=np.float64,
            )
            if setup.query_vectors is not None
            else None
        )
        family_meta: dict[str, Any] = {
            "name": family.name,
            "chunker": chunker.slug,
            "chunker_version": chunker.version,
            "predictor": family.predictor,
            "params": dict(family.params),
            "aggregation_rules": list(family.aggregations),
            "n_chunks": text_side.n_chunks,
            "chunks_per_text": chunks_block(text_side.chunks_per_text()),
            "semantic_threshold": threshold,
            "semantic_selection": semantic_selection,
            "label_embedding_run_id": family.label_embedding_run_id,
            "late": family_meta_late,
            "whole_text_agrees_with_reference": (
                bool(np.allclose(family_global, setup.scores))
                if family_global is not None
                else None
            ),
        }
        if label_side is not None:
            family_meta["label_chunker"] = label_side.n_chunks
            family_meta["label_chunks_per_text"] = chunks_block(label_side.chunks_per_text())
            family_meta["label_degeneracy"] = {
                "n_labels": label_side.n_texts,
                "n_labels_with_more_than_one_chunk": sum(
                    1 for count in label_side.chunks_per_text() if count > 1
                ),
                "note": (
                    "the UPR labels are 1-3 words, so the label side is almost always "
                    "one chunk: the symmetric arms are the null-check, and this "
                    "counter says how degenerate they are"
                ),
            }
        families[family.name] = family_meta

        for aggregation_spec in family.aggregations:
            rule = aggregation_of(aggregation_spec)
            name = arm_name(family.name, rule)
            if label_side is not None:
                scores = symmetric_arm_scores(
                    text_side,
                    label_side,
                    rule,
                    global_scores=family_global,
                )
            else:
                scores = chunked_arm_scores(
                    family_predictor, text_side, rule, global_scores=family_global
                )
            arms[name] = arm_block(
                view,
                scores,
                decision=eval_decision(config),
                ks=config.eval.metrics.ks,
                shortlist_ms=config.eval.metrics.shortlist_ms,
                reference_scores=setup.scores,
                chunks_per_text=text_side.chunks_per_text(),
                arm=name,
                predictor={"name": family.predictor, **dict(family.params)},
                chunker=chunker.slug,
                aggregation=rule.spec,
                per_label_distributions=False,
                paired=_upr_paired(config, view, scores, setup.scores),
            )
            if analysis_config.weight_grid.enabled and family.predictor == "fusion":
                refit = _upr_refit_weights(
                    config,
                    setup,
                    view,
                    analysis_config,
                    family,
                    text_side,
                    rule,
                    name,
                    family_predictor,
                    family_global,
                )
                if refit is not None:
                    arms[refit["name"]] = refit["block"]
                    family_meta.setdefault("weight_refits", {})[rule.slug] = refit["report"]
    return arms, families


def _upr_refit_weights(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    analysis_config: Any,
    family: Any,
    text_side: Any,
    rule: Any,
    name: str,
    family_predictor: Any,
    family_global: np.ndarray | None,
) -> dict[str, Any] | None:
    """Re-fit the fusion's weights on val for the chunked components (CS-8b).

    The grid explores the two free weights (``a`` and ``c``; ``b`` stays 1,
    the Phase-4 convention) **per chunk, before the fold** — so each
    combination is a different folding, not a different linear combination of
    the folded scores. The selection criterion is the run's decision rule's
    micro-F1 on **val** (the plan's protocol: val is for tuning), so the
    re-fitted arm's numbers are optimistically biased — the arm's name
    carries the ``_refit`` suffix and the artifact records the grid.

    The components are the pipeline's own, and the blend is checked against
    the family's predictor (the champion's weights) before the grid is
    explored, so the refit cannot silently use a different scoring path.
    """
    grid = analysis_config.weight_grid
    params = dict(family.params)
    k = int(params.get("k", 10))
    power = float(params.get("power", 1.0))
    components = fusion_components(setup.context, text_side.vectors, k=k, power=power)
    base_a = float(params.get("a", 0.0))
    base_b = float(params.get("b", 1.0))
    base_c = float(params.get("c", 0.0))
    base_blend = (
        base_a * components["knn"] + base_b * components["proto"] + base_c * components["zeroshot"]
    )
    predictor_scores = np.asarray(
        family_predictor.score(text_side.flat_chunks(), query_vectors=text_side.vectors),
        dtype=np.float64,
    )
    if not np.allclose(base_blend, predictor_scores, atol=1e-4):
        raise ValueError(
            "the fusion components do not reproduce the predictor's chunk scores "
            f"(max difference {float(np.max(np.abs(base_blend - predictor_scores))):.6f}): "
            "the weight re-fit would use a different scoring path than the pipeline's"
        )
    per_combination: list[dict[str, Any]] = []
    best: tuple[float, float, np.ndarray] | None = None
    for weight_a in grid.a:
        for weight_c in grid.c:
            blended = (
                weight_a * components["knn"]
                + 1.0 * components["proto"]
                + weight_c * components["zeroshot"]
            )
            folded = aggregate_chunk_similarities(
                blended, text_side.group_rows(), rule, global_similarities=family_global
            )
            criterion = _upr_fit_criterion(folded, view.gold, config.eval.decision, "micro_f1")
            per_combination.append({"a": weight_a, "c": weight_c, "micro_f1": criterion})
            if best is None or criterion > best[0]:
                best = (criterion, weight_a, weight_c, folded)
    assert best is not None
    criterion, best_a, best_c, best_scores = best
    block = arm_block(
        view,
        best_scores,
        decision=eval_decision(config),
        ks=config.eval.metrics.ks,
        shortlist_ms=config.eval.metrics.shortlist_ms,
        reference_scores=setup.scores,
        chunks_per_text=text_side.chunks_per_text(),
        arm=f"{name}_refit",
        predictor={"name": "fusion", "a": best_a, "b": 1.0, "c": best_c, "k": k},
        chunker=family.chunker.name,
        aggregation=rule.spec,
        per_label_distributions=False,
        paired=_upr_paired(config, view, best_scores, setup.scores),
    )
    report = {
        "criterion": "micro_f1 at the run's decision rule, on val",
        "selection_note": (
            "the weights are re-fitted on val, so the arm's val numbers are "
            "optimistically biased — the un-refitted arm is the comparable one"
        ),
        "grid": {"a": list(grid.a), "c": list(grid.c), "b": 1.0},
        "n_combinations": len(per_combination),
        "per_combination": per_combination,
        "chosen": {"a": best_a, "b": 1.0, "c": best_c, "micro_f1": criterion},
    }
    return {"name": f"{name}_refit", "block": block, "report": report}


def _upr_label_sides(
    setup: EvalSetup,
    catalog: Mapping[str, LabelEntry],
) -> list[UprLabel]:
    """The label space as the pack's label sides.

    Every label carries its **name** (the taxonomy term — the lexical
    features compare the text with it) and the text that is **embedded** for
    it (the zero-shot label vector's own text: the bare name or the config
    template). The catalog's description and parent are attached when the
    catalog has them.
    """
    embedded_texts = _upr_label_texts(setup)
    sides: list[UprLabel] = []
    for label, text in zip(setup.labels, embedded_texts, strict=True):
        entry = catalog.get(label)
        sides.append(
            UprLabel(
                name=label,
                text=text,
                description=entry.description if entry else None,
                parent=entry.parent if entry else None,
            )
        )
    return sides


def _upr_label_catalog(config: AppConfig) -> dict[str, LabelEntry]:
    """The optional label catalog (``paths.label_catalog``).

    An absent catalog is not an error: the pack's two catalog-gated feature
    groups are then unavailable and the artifact records the gap.
    """
    path = config.paths.label_catalog
    if path is None:
        return {}
    catalog = load_label_catalog(path)
    logger.info("the label catalog %s carries %d entries", path, len(catalog))
    return catalog


def _upr_bucket_masks(view: UprView) -> dict[str, np.ndarray]:
    """The label buckets' column masks (the calibration's per-bucket view)."""
    masks: dict[str, np.ndarray] = {}
    for bucket, members in view.labels_by_bucket.items():
        columns = np.asarray(
            [view.labels.index(label) for label in members],
            dtype=np.int64,
        )
        mask = np.zeros(view.n_labels, dtype=bool)
        mask[columns] = True
        masks[bucket] = mask
    return masks


def _upr_length_masks(view: UprView) -> dict[str, np.ndarray]:
    """The text-length buckets' row masks (the calibration's per-length view)."""
    masks: dict[str, np.ndarray] = {}
    for index, name in enumerate(view.length_bucket_names):
        mask = np.zeros(view.n_texts, dtype=bool)
        for position, bucket in enumerate(view.length_bucket_index):
            if bucket == index:
                mask[position] = True
        masks[name] = mask
    return masks


def _upr_learned_arms(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    analysis_config: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The learned arm and its three floors (CS-8d).

    The arm is **fitted on train** (the train gold cells plus the train
    non-gold cells as down-weighted presumed negatives) and **scored on the
    evaluated split** (val), in the same run as the whole-text reference arm
    — so the pairing is by construction.

    The three floors (the chance, the control-role and the random-vector one)
    are measured on the same statistic as the headline: the UPR harness's own
    decision-view and ranking-view metrics.
    """
    learned = analysis_config.learned
    if not learned.enabled:
        return {}, {}
    embedder = _check_embedder_matches_artifact(config, setup)
    spec = build_preprocess_spec(config.preprocess)
    catalog = _upr_label_catalog(config)
    labels = _upr_label_sides(setup, catalog)

    chunker, chunk_embedder = _upr_chunker(config, embedder, learned.chunker, spec.version)
    text_embedder = _cached_embedder(embedder, config, spec.version)
    label_embedder = _cached_embedder(embedder, config, spec.version)
    # The chunk profile's units (the sentences and the word pieces) on the text
    # side; the label side is one unit (the CS-8b verdict: the UPR labels are
    # 1-3 words, so the label side is the identity).
    profile_units = None
    sentence_embedder = None
    ngram_embedder = None
    if "chunk_profile" in learned.groups:
        sentence_chunker = build_chunker("units", {"kind": "sentence"})
        ngram_chunker = build_chunker(
            "units", {"kind": "ngram", "n_words": learned.profile_ngram_size}
        )
        profile_units = ProfileUnits(sentences=sentence_chunker, ngrams=ngram_chunker)
        sentence_embedder = _cached_embedder(
            embedder, config, sentence_chunker.cache_version(spec.version)
        )
        ngram_embedder = _cached_embedder(
            embedder, config, ngram_chunker.cache_version(spec.version)
        )
    extractor = UprFeatureExtractor(
        UprFeatureEmbedders(
            text=text_embedder,
            chunks=chunk_embedder,
            labels=label_embedder,
            text_sentences=sentence_embedder,
            text_ngrams=ngram_embedder,
        ),
        labels=labels,
        text_chunker=chunker,
        top_k=learned.top_k,
        groups=learned.groups,
        catalog_available=bool(catalog),
        profile_units=profile_units,
        profile_top_k=learned.profile_top_k,
    )
    # The random-vector floor: the pack's embedding sides are seeded random
    # unit vectors of the model's dimension (never written to the cache).
    random_extractor = None
    if learned.floors.random_vectors:
        dimension = int(embedder.info.dimension or 0)
        if dimension <= 0:
            raise ValueError(
                "the random-vector floor needs the model's dimension; the embed: "
                "section's backend does not report one"
            )
        random_embedder = RandomEmbedder(dimension=dimension, seed=learned.floors.random_seed)
        random_extractor = UprFeatureExtractor(
            UprFeatureEmbedders(
                text=random_embedder,
                chunks=random_embedder,
                labels=random_embedder,
                text_sentences=random_embedder,
                text_ngrams=random_embedder,
            ),
            labels=labels,
            text_chunker=chunker,
            top_k=learned.top_k,
            groups=learned.groups,
            catalog_available=bool(catalog),
            profile_units=profile_units,
            profile_top_k=learned.profile_top_k,
        )

    train_texts = [record.text for record in setup.train_records]
    eval_texts = [record.text for record in setup.records]
    train_gold = build_label_matrix(setup.train_records, setup.labels)
    seed = learned.seed if learned.seed is not None else config.seed
    decision_train, decision_eval = _upr_decision_feature(config, setup, view, learned)
    rerank_train, rerank_eval = _upr_rerank_feature(
        config, setup, view, learned, decision_train, decision_eval
    )
    result = score_upr_learned_arms(
        train_texts=train_texts,
        train_gold=train_gold,
        eval_texts=eval_texts,
        eval_gold=view.gold,
        extractor=extractor,
        combiner_factory=lambda: build_combiner(learned.combiner.name, learned.combiner.params),
        decision=eval_decision(config),
        ks=config.eval.metrics.ks,
        shortlist_ms=config.eval.metrics.shortlist_ms,
        control_weight=learned.control_weight,
        permutation_resamples=learned.floors.permutation_resamples,
        random_extractor=random_extractor,
        seed=seed,
        arm_name=f"learned_{learned.combiner.name}",
        reporting=learned.reporting.model_dump(),
        bucket_masks=_upr_bucket_masks(view),
        length_masks=_upr_length_masks(view),
        decision_train=decision_train,
        rerank_train=rerank_train,
        rerank_eval=rerank_eval,
        decision_eval=decision_eval,
    )
    chunks_per_text = extractor.chunks_per_text(eval_texts)
    arms: dict[str, dict[str, Any]] = {}
    for name, scores in result.scores.items():
        arms[name] = arm_block(
            view,
            scores,
            decision=eval_decision(config),
            ks=config.eval.metrics.ks,
            shortlist_ms=config.eval.metrics.shortlist_ms,
            reference_scores=setup.scores,
            chunks_per_text=chunks_per_text,
            arm=name,
            predictor={
                "name": "learned",
                "combiner": learned.combiner.name,
                "params": dict(learned.combiner.params),
                "groups": list(learned.groups),
            },
            chunker=chunker.slug,
            aggregation=None,
            # The per-label distributions of the main arm only: they say which
            # labels the arm finds (the unseen bucket's question), and the
            # floors' per-label view would be noise.
            per_label_distributions=name == result.meta["arm"],
            paired=_upr_paired(config, view, scores, setup.scores),
        )
    meta = dict(result.meta)
    meta["catalog"] = {
        "path": str(config.paths.label_catalog) if config.paths.label_catalog else None,
        "n_entries": len(catalog),
        "note": (
            "the label-description coverage and the label-type features need a "
            "label catalog (a name / description / parent per label). Without "
            "one the pack runs without those two groups and the gap is stated "
            "here."
        ),
    }
    backoff_meta = _upr_backoff_arms(config, setup, view, learned, result, arms, labels)
    if backoff_meta:
        meta["backoff"] = backoff_meta
    return arms, meta


def _upr_decision_feature(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    learned: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The decision model's probability matrix for the pack (the track's CS-13).

    The feature is **cell-local**: the model reads the text against the
    label's name and answers a probability. It is scored at the edge (the
    model is a client) and the matrix is passed into the pack, which stays a
    pure core.

    The **fit side** is scored too: the combiner is fitted on train, so the
    feature must exist for the train cells as well (and the train split is
    where the eight held-out labels have **no** positives -- the transfer the
    hypothesis is about).

    Returns:
        The fit side's and the evaluated split's matrices (``None`` when the
        feature is off).
    """
    feature = learned.decision_feature
    if not feature.enabled or feature.model is None:
        return None, None
    decider = build_upr_decider(feature.model, config)
    arm = DecisionArm(
        name=feature.model.name,
        state=feature.model.state,
        question_type=feature.model.question_type,
        instructions=feature.model.instructions,
        options=feature.model.options,
        criteria=feature.model.criteria,
        positive_option=feature.model.positive_option,
        max_input_tokens=feature.model.max_input_tokens,
    )
    labels = tuple(setup.labels)
    train_sample = UprDecisionSample(
        texts=tuple(record.text for record in setup.train_records),
        labels=labels,
        gold=build_label_matrix(setup.train_records, setup.labels),
        rows=tuple(range(len(setup.train_records))),
    )
    eval_sample = UprDecisionSample(
        texts=tuple(record.text for record in setup.records),
        labels=labels,
        gold=view.gold,
        rows=tuple(range(view.n_texts)),
    )
    train_matrix, _ = score_upr_grid(
        train_sample,
        decider,
        arm,
        max_questions_per_call=feature.max_questions_per_call,
        mismatched=feature.label_blind,
    )
    eval_matrix, _ = score_upr_grid(
        eval_sample,
        decider,
        arm,
        max_questions_per_call=feature.max_questions_per_call,
        mismatched=feature.label_blind,
    )
    return train_matrix, eval_matrix


def _upr_rerank_feature(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    learned: Any,
    decision_train: np.ndarray | None,
    decision_eval: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The reranker's score matrix for the pack (the track's CS-16).

    The feature is **cell-local**: the reranker reads the text against the
    label's document and answers a score. It is scored at the edge (the
    reranker is a client) and the matrix is passed into the pack, which stays
    a pure core.

    The **fit side** is scored too: the combiner is fitted on train, so the
    feature must exist for the train cells as well (and the train split is
    where the eight held-out labels have **no** positives — the transfer the
    CS-13 hypothesis was about).

    Two shapes:

    - ``full``: every (text, label) cell is scored (68 pairs per text);
    - ``restricted``: only the candidate set's cells are scored (10 pairs per
      text) and the cells outside it are filled with the **pre-registered**
      fill value (the row's minimum scored value).

    Returns:
        The fit side's and the evaluated split's matrices (``None`` when the
        feature is off).
    """
    feature = learned.rerank_feature
    if not feature.enabled or feature.model is None:
        return None, None
    model_config = feature.model
    client = build_reranker(model_config, config)
    catalog = _upr_label_catalog(config)
    descriptions = {name: entry.description for name, entry in catalog.items() if entry.description}
    documents = label_documents(list(setup.labels), descriptions, feature.label_document)
    if feature.label_blind:
        documents = shifted_documents(documents)

    train_texts = [record.text for record in setup.train_records]
    eval_texts = [record.text for record in setup.records]

    if feature.shape == "full":
        train_matrix = _rerank_full_matrix(
            client, model_config, train_texts, documents, feature.order
        )
        eval_matrix = _rerank_full_matrix(
            client, model_config, eval_texts, documents, feature.order
        )
        return train_matrix, eval_matrix

    # The restricted shape: the candidate set's cells only, the rest filled.
    if feature.candidate_set == "decision" and (decision_train is None or decision_eval is None):
        raise ValueError(
            "the restricted rerank feature needs the decision model's score "
            "matrix (its candidate set): enable the decision feature as well"
        )
    sources = {
        "decision": (decision_train, decision_eval),
        "champion": (
            np.asarray(setup.context.train_scores, dtype=np.float64)
            if hasattr(setup.context, "train_scores")
            else None,
            np.asarray(setup.scores, dtype=np.float64),
        ),
    }
    if feature.candidate_set == "union":
        raise ValueError(
            "the restricted rerank feature's 'union' candidate set is not wired: "
            "the union needs the decision model's grid (the 'decision' set)"
        )
    train_source, eval_source = sources[feature.candidate_set]
    if train_source is None or eval_source is None:
        raise ValueError(
            f"the restricted rerank feature's candidate set {feature.candidate_set!r} "
            "has no score matrix on the fit side"
        )
    train_matrix = _rerank_restricted_matrix(
        client,
        model_config,
        train_texts,
        documents,
        train_source,
        feature.candidate_k,
        feature.order,
    )
    eval_matrix = _rerank_restricted_matrix(
        client,
        model_config,
        eval_texts,
        documents,
        eval_source,
        feature.candidate_k,
        feature.order,
    )
    return train_matrix, eval_matrix


def _rerank_full_matrix(
    client: Any,
    model_config: Any,
    texts: Sequence[str],
    documents: Sequence[str],
    order: str,
) -> np.ndarray:
    """Every (text, label) cell's score (the full-grid shape)."""
    per_text = [list(documents) for _ in texts]
    requests = build_candidate_requests(texts, per_text, kind=model_config.kind, order=order)
    scores = client.score(requests)
    matrix = np.zeros((len(texts), len(documents)), dtype=np.float64)
    if model_config.kind == KIND_LISTWISE:
        for row, row_scores in enumerate(scores):
            matrix[row, :] = row_scores
        return matrix
    flat = [value for row in scores for value in row]
    cursor = 0
    for row in range(len(texts)):
        size = len(documents)
        matrix[row, :] = flat[cursor : cursor + size]
        cursor += size
    return matrix


def _rerank_restricted_matrix(
    client: Any,
    model_config: Any,
    texts: Sequence[str],
    documents: Sequence[str],
    source: np.ndarray,
    candidate_k: int,
    order: str,
) -> np.ndarray:
    """The candidate set's cells' scores, the rest filled with the row's floor."""
    columns = candidate_columns(np.asarray(source, dtype=np.float64), candidate_k)
    per_text = [[documents[column] for column in row] for row in columns]
    requests = build_candidate_requests(texts, per_text, kind=model_config.kind, order=order)
    scores = client.score(requests)
    if model_config.kind == KIND_LISTWISE:
        per_candidate = [list(row) for row in scores]
    else:
        flat = [value for row in scores for value in row]
        per_candidate = []
        cursor = 0
        for row in columns:
            size = len(row)
            per_candidate.append(flat[cursor : cursor + size])
            cursor += size
    return fill_restricted_scores(columns, per_candidate, len(documents))


def _upr_backoff_arms(
    config: AppConfig,
    setup: EvalSetup,
    view: UprView,
    learned: Any,
    result: Any,
    arms: dict[str, dict[str, Any]],
    labels: Sequence[Any],
) -> dict[str, Any]:
    """The **sibling-max backoff** arms (the track's CS-11).

    The catalog's **parent** channel: a label with no fit-side support (zero
    gold cells on the train split) inherits the score of a taxonomy sibling.
    The **control** is the point — the same rule applied to the fit side's
    rarest *supported* labels must **not** help them, or the honest claim
    changes to "another correlated signal".

    Which labels get the backoff and which labels form the control are
    decided on the **fit side** (the train split's own gold counts), so
    nothing about the evaluated split enters the choice.
    """
    if not learned.backoff.enabled:
        return {}
    parents = {label.name: label.parent for label in labels if label.parent}
    if not parents:
        raise ValueError(
            "the sibling-max backoff needs the catalog's parent field: the label "
            "catalog is absent or carries no parent (paths.label_catalog)"
        )
    train_gold = build_label_matrix(setup.train_records, setup.labels)
    unsupported = unsupported_labels(train_gold, setup.labels)
    control = (
        control_labels(train_gold, setup.labels, n=len(unsupported))
        if learned.backoff.control
        else ()
    )
    main_arm = result.meta["arm"]
    main_scores = result.scores[main_arm]
    report: dict[str, Any] = {
        "unsupported_labels": list(unsupported),
        "control_labels": list(control),
        "n_unsupported": len(unsupported),
        "n_control": len(control),
        "note": (
            "the sibling-max backoff: the targets are the labels with NO "
            "fit-side support (decided on the train split) and the control's "
            "labels are the fit side's rarest SUPPORTED ones (matched in "
            "rarity). A label without a supported sibling is a measured null "
            "by construction."
        ),
        "per_lam": {},
    }
    for lam in learned.backoff.lams:
        variants = (("backoff", unsupported), ("backoff_control", control))
        for suffix, targets in variants:
            if not targets:
                continue
            backoff_scores, detail = sibling_backoff(
                main_scores,
                setup.labels,
                parents,
                train_gold,
                targets=targets,
                lam=lam,
            )
            name = f"{main_arm}_{suffix}_l{lam:g}"
            arms[name] = arm_block(
                view,
                backoff_scores,
                decision=eval_decision(config),
                ks=config.eval.metrics.ks,
                shortlist_ms=config.eval.metrics.shortlist_ms,
                reference_scores=setup.scores,
                chunks_per_text=None,
                arm=name,
                predictor={
                    "name": "learned_backoff",
                    "combiner": learned.combiner.name,
                    "params": dict(learned.combiner.params),
                    "groups": list(learned.groups),
                    "variant": suffix,
                },
                chunker=None,
                aggregation=None,
                per_label_distributions=True,
                # the paired comparison is against the **unbacked-off** arm of
                # the same run: the backoff is the only difference
                paired=_upr_paired(config, view, backoff_scores, main_scores),
            )
            report["per_lam"].setdefault(f"{lam:g}", {})[suffix] = {
                "arm": name,
                "targets": list(targets),
                **detail,
            }
    return report


def _select_semantic_threshold(
    config: AppConfig,
    setup: EvalSetup,
    analysis_config: Any,
    embedder: Any,
    chunker_spec: ChunkerSpec,
) -> tuple[float, dict[str, Any]]:
    """The fit-side selection of the semantic chunker's threshold (CS-8b).

    The threshold is chosen on the **fit side only** — the train split — with
    the fit side's own label-scoring metric, and the **cross-fit** over the
    train records reports what the choice is worth on records that did not
    pick it (the UPR analogue of OMCT's document rotation).

    The selection uses the **pair-scoring** geometry (the chunked cosine
    against the label vectors): the retrieval geometry's fit-side score would
    contain the text's own train example (a self-match), which is a leak the
    track's rules forbid.

    Returns:
        The selected threshold and the selection's report.
    """
    grid = tuple(analysis_config.semantic_thresholds)
    spec = build_preprocess_spec(config.preprocess)
    train_texts = [record.text for record in setup.train_records]
    gold = build_label_matrix(setup.train_records, setup.labels)
    label_vectors = setup.context.label_vectors
    if label_vectors is None:
        raise ValueError(
            "the semantic threshold's fit-side selection needs the zero-shot label "
            "vectors: set eval.label_embedding_run_id to a prior upr-labelembed-* run"
        )
    folds = _upr_folds(len(train_texts), analysis_config.semantic_folds, config.seed)
    per_threshold: list[dict[str, Any]] = []
    scores_by_threshold: dict[float, np.ndarray] = {}
    for candidate in grid:
        chunker, chunk_embedder = _upr_chunker(
            config, embedder, chunker_spec, spec.version, threshold=candidate
        )
        side = encode_chunks(chunk_embedder, chunker, train_texts, "query")
        chunk_scores = (
            np.asarray(side.vectors, dtype=np.float64)
            @ np.asarray(label_vectors, dtype=np.float64).T
        )
        folded = aggregate_chunk_similarities(
            chunk_scores, side.group_rows(), aggregation_of("max")
        )
        scores_by_threshold[candidate] = folded
        per_threshold.append(
            {
                "threshold": candidate,
                "criterion": _upr_fit_criterion(
                    folded, gold, config.eval.decision, analysis_config.semantic_criterion
                ),
                "n_chunks": side.n_chunks,
            }
        )
    chosen = max(per_threshold, key=lambda entry: entry["criterion"])["threshold"]
    per_fold: list[dict[str, Any]] = []
    for index, rows in enumerate(folds):
        held_out = np.asarray(rows, dtype=np.int64)
        others = np.asarray(
            [row for position, fold in enumerate(folds) if position != index for row in fold],
            dtype=np.int64,
        )
        fold_choice = max(
            per_threshold,
            key=lambda entry: _upr_fit_criterion(
                scores_by_threshold[entry["threshold"]][others],
                gold[others],
                config.eval.decision,
                analysis_config.semantic_criterion,
            ),
        )["threshold"]
        per_fold.append(
            {
                "fold": index,
                "n_records": int(held_out.size),
                "chosen_threshold": fold_choice,
                "criterion_held_out": _upr_fit_criterion(
                    scores_by_threshold[fold_choice][held_out],
                    gold[held_out],
                    config.eval.decision,
                    analysis_config.semantic_criterion,
                ),
                "criterion_reference": _upr_fit_criterion(
                    scores_by_threshold[chosen][held_out],
                    gold[held_out],
                    config.eval.decision,
                    analysis_config.semantic_criterion,
                ),
            }
        )
    report = {
        "criterion": analysis_config.semantic_criterion,
        "criterion_note": (
            "the fit side's own label-scoring metric on the train split, at the "
            "run's decision rule; the pair-scoring geometry (the chunked cosine "
            "against the label vectors), because the retrieval geometry's fit-side "
            "score would contain the text's own train example"
        ),
        "grid": list(grid),
        "per_threshold": per_threshold,
        "chosen_threshold": chosen,
        "folds": per_fold,
        "n_folds": len(folds),
        "n_train_records": len(train_texts),
        "selection": "the whole fit side (train) — the folds only report the transfer",
    }
    return chosen, report


def _upr_folds(n_records: int, n_folds: int, seed: int) -> list[list[int]]:
    """The fit-side cross-fit folds over the train records (seeded, deterministic)."""
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_records)
    return [sorted(int(row) for row in order[index::n_folds]) for index in range(n_folds)]


def _upr_fit_criterion(
    scores: np.ndarray,
    gold: np.ndarray,
    decision: Any,
    criterion: str,
) -> float:
    """The fit side's label-scoring metric (the threshold selection's criterion)."""
    predicted = apply_decision_rule(
        scores,
        rule=decision.rule,
        k=decision.k,
        threshold=decision.threshold,
        min_score=decision.min_score,
    )
    metrics = decision_f1_metrics(predicted, gold)
    if criterion == "micro_f1":
        return float(metrics["micro_f1"])
    ranking = ranking_metrics(scores, gold, (5,))
    return float(ranking["ndcg@5"])


def eval_decision(config: AppConfig) -> dict[str, Any]:
    """The run's decision rule as a mapping (the arm blocks' input)."""
    if config.eval is None:
        raise ValueError("the eval: section is required")
    return config.eval.decision.model_dump()


def _upr_paired(
    config: AppConfig,
    view: UprView,
    scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict[str, Any]:
    """The arm's **paired** comparison against the whole-text reference arm.

    The track's discipline (a paired interval, never two independent
    numbers): the resampling unit is the **text**, so both scorers are
    judged on the same resampled texts. The decision view (the micro-F1 at
    the run's decision rule) and the ranking view (the nDCG@5) are both
    reported, with the interval and the share of positive resamples — pooled
    **and per text-length bucket** (the hypothesis is about the long texts,
    and the pooled interval would hide them).
    """
    decision = eval_decision(config)
    bootstrap = config.upr_analysis.bootstrap
    seed = bootstrap.seed if bootstrap.seed is not None else config.seed
    left = np.asarray(scores, dtype=np.float64)
    right = np.asarray(reference_scores, dtype=np.float64)
    block: dict[str, Any] = {
        "micro_f1": paired_metric_bootstrap(
            left,
            right,
            view.gold,
            metric="micro_f1",
            decision=decision,
            resamples=bootstrap.resamples,
            seed=seed,
        ),
        "ndcg5": paired_metric_bootstrap(
            left,
            right,
            view.gold,
            metric="ndcg",
            k=5,
            resamples=bootstrap.resamples,
            seed=seed,
        ),
        "per_length_bucket": {},
    }
    for index, name in enumerate(view.length_bucket_names):
        rows = np.asarray(
            [
                position
                for position, bucket in enumerate(view.length_bucket_index)
                if bucket == index
            ],
            dtype=np.int64,
        )
        if rows.size < 2:
            continue
        block["per_length_bucket"][name] = {
            "n_texts": int(rows.size),
            "micro_f1": paired_metric_bootstrap(
                left[rows],
                right[rows],
                view.gold[rows],
                metric="micro_f1",
                decision=decision,
                resamples=bootstrap.resamples,
                seed=seed,
            ),
        }
    return block


def _cmd_analyze_upr(config: AppConfig) -> int:
    """Run the analyze-upr command: the track's UPR artifact (CS-8a).

    The text-processing track's UPR counterpart of ``analyze-omct``: the
    whole-text **reference arm** (the main pipeline's own scoring of the
    split — the same predictor, the same embedding runs, the same decision
    rule) beside the track's chunked arms, with the UPR-side diagnostics:
    the gold cells' rank percentiles inside their own text's label list, the
    hit@1 / hit@10, the **non-gold cells as the background** (fully
    enumerable on UPR: no control sample, no seed), the background's level
    and rank shift beside every headline number, the score distributions per
    bucket and per label, the texts' length buckets and the top-1 provenance.

    The reference arm must reproduce the matching ``evaluate`` run field by
    field (the guard, ``upr_analysis.reference_eval_run``).

    Writes ``results/<run_id>/{config.yaml, analysis.json}`` and the track's
    UPR table rows.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.eval is None:
        raise ValueError(
            "the eval: section (predictor, decision) is required for the `analyze-upr` command"
        )
    if config.upr_analysis is None:
        raise ValueError("the upr_analysis: section is required for the `analyze-upr` command")
    eval_config = config.eval
    analysis_config = config.upr_analysis
    setup = _prepare_evaluation(config, eval_config.split)
    decision = eval_config.decision.model_dump()

    view = build_upr_view(
        setup.records,
        setup.labels,
        setup.buckets,
        analysis_config.length_buckets,
    )
    reference_block = arm_block(
        view,
        setup.scores,
        decision=decision,
        ks=eval_config.metrics.ks,
        shortlist_ms=eval_config.metrics.shortlist_ms,
        arm=BASELINE_ARM,
        predictor=setup.predictor.describe(),
        chunker="full_text",
        aggregation=None,
    )
    guard = _check_reference_arm(config, analysis_config.reference_eval_run, reference_block)
    arms: dict[str, dict[str, Any]] = {BASELINE_ARM: reference_block}
    chunked_arms, families = _upr_chunked_arms(config, setup, view, analysis_config)
    arms.update(chunked_arms)
    learned_arms, learned_meta = _upr_learned_arms(config, setup, view, analysis_config)
    arms.update(learned_arms)

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    payload = build_upr_report(
        view,
        arms,
        run_id=run_id,
        split=setup.split,
        source=str(config.paths.data_raw),
        source_sha256=setup.source_sha256,
        predictor=setup.predictor.describe(),
        embedding_run=setup.artifact.lineage if setup.artifact is not None else None,
        label_embedding_run=(
            setup.label_artifact.lineage if setup.label_artifact is not None else None
        ),
        decision=decision,
        metrics_config=eval_config.metrics.model_dump(),
        length_edges=analysis_config.length_buckets,
        guard=guard,
        families=families,
        learned=learned_meta,
        total_seconds=round(time.perf_counter() - started, 3),
    )
    (results_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _append_command_rows(
        config,
        "analyze-upr",
        columns=ANALYZE_UPR_COLUMNS,
        key_columns=ANALYZE_UPR_KEY_COLUMNS,
        rows=_upr_arm_rows(
            arms,
            run_id=run_id,
            split=setup.split,
            seconds=payload["total_seconds"],
        ),
    )

    overall = reference_block["metrics"]["overall"]
    print(
        f"Analyzed {view.n_texts} record(s) [{setup.split}] with predictor "
        f"'{eval_config.predictor.name}' ({view.n_labels} labels, "
        f"{view.n_gold_cells} gold cells, {view.n_background_cells} background cells)"
    )
    print(
        f"  reference arm: micro-F1 {overall['f1']['micro_f1']:.4f}, "
        f"macro-F1 {overall['f1']['macro_f1']:.4f}, "
        f"sample-F1 {overall['f1']['sample_f1']:.4f}, "
        f"coverage {overall['coverage']['coverage']:.3f}, "
        f"P@5 {overall['ranking'].get('precision@5', 0.0):.4f}, "
        f"R@10 {overall['ranking'].get('recall@10', 0.0):.4f}, "
        f"nDCG@5 {overall['ranking'].get('ndcg@5', 0.0):.4f}, "
        f"shortlist R@50 {overall['shortlist'].get('recall@50', 0.0):.4f}"
    )
    print(
        f"  gold: rank p50 {reference_block['gold']['rank_p50']}, "
        f"hit@1 {reference_block['gold']['hit@1']}, "
        f"hit@10 {reference_block['gold']['hit@10']}; "
        f"background: rank p50 {reference_block['background']['rank_p50']}, "
        f"mean {reference_block['background']['mean']}"
    )
    if guard is not None:
        print(f"  guard: the reference arm reproduces {guard['reference_run']} field by field")
    print(f"Artifact written to {results_dir / 'analysis.json'}")
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
        elif run.command == "analyze-upr":  # the text-processing track's UPR side
            code = _cmd_analyze_upr(run.config)
        elif run.command == "decide-upr":  # the track's UPR decision probe (CS-9)
            code = _cmd_decide_upr(run.config)
        elif run.command == "rerank-upr":  # the track's UPR reranker probe (CS-15)
            code = _cmd_rerank_upr(run.config)
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
        if args.command == "analyze-upr":
            return _cmd_analyze_upr(config)
        if args.command == "decide-upr":
            return _cmd_decide_upr(config)
        if args.command == "rerank-upr":
            return _cmd_rerank_upr(config)
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
