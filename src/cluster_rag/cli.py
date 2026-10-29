"""Command-line entrypoint, installed as the ``cluster-rag`` console script.

Invocation (target CLI in AGENTS.md):

    uv run cluster-rag stats    --config configs/base.yaml
    python -m cluster_rag.cli stats    --config configs/base.yaml   # equivalent

Commands: stats (--omct for the OMCT pairwise data), split, embed
(Phase 2: dataset embedding through the disk cache), labelembed (Phase 4:
the zero-shot label-text artifact), probe-omct (Phase 2: the
OMCT pairwise embedding probe), analyze-omct (the text-processing track's
Stage-0 pair-score artifact), decide-omct (the track's Stage-3.5 decision
probe), analyze-upr (the track's UPR-side analysis artifact), decide-upr (the
track's UPR-side decision probe), rerank-upr (the track's zero-shot reranker
probe), hitmap-upr (the track's hit map: which words carry the signal),
evaluate
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

This module is the **thin shell**: the argument parser and the dispatch. The
commands themselves live in ``cluster_rag.commands`` — one module per family
(``stats``, ``embed``, ``omct``, ``upr``, ``sweep``) plus the shared helpers in
``commands._common``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from cluster_rag import __version__
from cluster_rag.commands.embed import _cmd_embed, _cmd_labelembed, _parse_kinds
from cluster_rag.commands.omct import _cmd_analyze_omct, _cmd_decide_omct, _cmd_probe_omct
from cluster_rag.commands.stats import _cmd_split, _cmd_stats
from cluster_rag.commands.sweep import _cmd_sweep
from cluster_rag.commands.upr import (
    _cmd_analyze_upr,
    _cmd_decide_upr,
    _cmd_evaluate,
    _cmd_hitmap_upr,
    _cmd_rerank_upr,
)
from cluster_rag.config import load_config

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

    hitmap_upr = subparsers.add_parser(
        "hitmap-upr",
        help="Text-processing track: the hit map (hitmap-upr, CS-18).",
        description=(
            "The text-processing track's hit map "
            "(TEXT_PROCESSING_PLAN.md section 7.1, CS-18): which words carry "
            "the signal. Two versions in one run — the LEXICAL map (the "
            "lexical feature's own exact per-word decomposition, which maps the "
            "label side and costs no model call) and the ABLATION map (the "
            "decision model's own sentence- or clause-level ablation: one unit "
            "of the text is removed and the model is asked again, which maps "
            "the text side and is the only version that perturbs the input, so "
            "the only one that says anything causal). The decision model is "
            "the same pinned model as the CS-13 feature, and the label-blind "
            "counterpart runs beside the ablation. Writes results/<run_id>/"
            "hitmap.json; the row goes to the track's own UPR table."
        ),
    )
    hitmap_upr.add_argument(
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
        if args.command == "hitmap-upr":
            return _cmd_hitmap_upr(config)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc.filename or exc)
        return 1
    except ValueError as exc:  # includes pydantic ValidationError and YAML errors
        logger.error("Invalid config or data: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
