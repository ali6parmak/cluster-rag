"""The UPR-side commands: ``evaluate``, ``analyze-upr``, ``decide-upr``,
``rerank-upr``, ``hitmap-upr``.

``evaluate`` is the Phase-3 harness (any predictor on a fixed val/test
split); the other four are the text-processing track's UPR artifacts — the
analysis, the decision probe, the zero-shot reranker probe and the **hit map**
(CS-18: which words carry the signal).
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from cluster_rag.commands._common import (
    _append_command_rows,
    _cached_embedder,
    _check_reference_arm,
    _fmt,
    _require_run_id,
    _upr_label_catalog,
    _warn_upr_prefix,
    _write_resolved_config,
)
from cluster_rag.config import AppConfig, ChunkerSpec
from cluster_rag.data import (
    Record,
    build_label_matrix,
    load_jsonl,
    load_split_ids,
    select_records_by_ids,
)
from cluster_rag.dataset_stats import file_sha256
from cluster_rag.decision import CachedDecider, SystemOneDecider
from cluster_rag.embed import build_embedder
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
    ANALYZE_UPR_COLUMNS,
    ANALYZE_UPR_KEY_COLUMNS,
    DECIDE_UPR_COLUMNS,
    DECIDE_UPR_KEY_COLUMNS,
    EVALUATE_COLUMNS,
    EVALUATE_KEY_COLUMNS,
    HITMAP_UPR_COLUMNS,
    HITMAP_UPR_KEY_COLUMNS,
    RERANK_UPR_COLUMNS,
    RERANK_UPR_KEY_COLUMNS,
)
from cluster_rag.label_catalog import LabelEntry
from cluster_rag.omct_analysis import BASELINE_ARM
from cluster_rag.omct_decision import DecisionArm, score_cells
from cluster_rag.predictors import build_predictor, predictor_requirements
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.preprocess import build_preprocess_spec
from cluster_rag.rerank.base import KIND_LISTWISE
from cluster_rag.text_processors.aggregation import aggregate_chunk_similarities
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.combining import build_combiner
from cluster_rag.text_processors.decision.ablation_map import (
    ablatable_summary,
    build_ablation_cells,
    build_variants,
    label_bucket_map,
    label_linked_units,
    measure_ablation,
    text_units,
)
from cluster_rag.text_processors.decision.upr_probe import measure_upr_arm
from cluster_rag.text_processors.decision.upr_union import z_max_union
from cluster_rag.text_processors.features import ProfileUnits
from cluster_rag.text_processors.features.lexical_map import (
    aggregate_lexical_map,
    build_lexical_map,
    build_text_side_map,
    exactness_check,
    label_word_classes,
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
from cluster_rag.text_processors.surrogate import (
    agreement_with_carrier,
    build_design_matrix,
    build_vocabulary,
    fit_ridge_surrogate,
    name_word_share,
    surrogate_predictions,
    surrogate_quality,
    topic_attribution,
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


def _cmd_hitmap_upr(config: AppConfig) -> int:
    """Run hitmap-upr: the track's hit map (CS-18).

    Two versions, one run:

    - **Version A — the lexical map.** The lexical feature's own per-word
      decomposition: ``lex_label_coverage`` is the *average* over a label
      name's content words of how much of each word the text has, so each
      word's contribution is **exact**. This maps the **label side** and it is
      a map of the *word-overlap feature*, never of the embedding model and
      never of the boosted-tree ensemble.
    - **Version B — the ablation map.** The decision model's own
      sentence- or clause-level ablation: one unit of the text is removed and
      the model is asked again. This maps the **text side**, and it is the only
      version that **perturbs** the input, so it is the only one that says
      anything causal. The **label-blind counterpart** runs beside it.

    The decision model is the very same pinned model as the CS-13 feature (the
    ``upr_analysis.decision`` section), so the map and the feature are read
    against one model and the intact-text calls are that run's cache hits.

    Writes ``results/<run_id>/{config.yaml, hitmap.json}``: the text set's
    shape (how many texts are ablatable at all), the lexical map and its
    aggregates, the ablation's three paired reads with their intervals, the
    label-blind counterpart beside each, the falsification flags and the cost.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    _warn_upr_prefix(run_id)
    if config.upr_analysis is None:
        raise ValueError("the upr_analysis: section is required for the `hitmap-upr` command")
    if config.eval is None:
        raise ValueError("the eval: section is required for the `hitmap-upr` command")
    analysis = config.upr_analysis
    hitmap = analysis.hitmap
    decision = analysis.decision
    if not decision.models:
        raise ValueError(
            "the upr_analysis.decision: section carries no model: the hit map "
            "reads the same pinned decision model as the CS-13 feature, so it "
            "needs one"
        )
    setup = _prepare_evaluation(config, config.eval.split)
    texts = [record.text for record in setup.records]
    labels = list(setup.labels)
    gold = setup.gold
    buckets = setup.buckets
    # The held-out labels: the bucket's "unseen" ones (the fit side never saw
    # them), which is where the CS-18 question-2 split is read.
    held_out = [label for label in labels if buckets.get(label) == "unseen"]
    openers = tuple(config.preprocess.strip_openers)
    bootstrap_seed = hitmap.bootstrap.seed if hitmap.bootstrap.seed is not None else config.seed

    # -- Version A: the lexical map (pure, no model call) --------------------------
    lexical_block: dict[str, Any] | None = None
    text_side_block: dict[str, Any] | None = None
    if hitmap.include_lexical_map:
        map_cells = build_lexical_map(texts, labels)
        word_classes = label_word_classes(labels)
        lexical_block = aggregate_lexical_map(
            map_cells,
            labels=labels,
            gold=gold,
            buckets=buckets,
            word_classes=word_classes,
            held_out_labels=held_out,
            catch_all_labels=tuple(hitmap.catch_all_labels),
        )
        lexical_block["exactness"] = exactness_check(map_cells, texts, labels)
        lexical_block["label_buckets"] = label_bucket_map(buckets, labels)
    if hitmap.include_text_side_map:
        text_side_block = build_text_side_map(texts, labels, gold)

    # -- Version B: the ablation map (the decision model's own ablation) ----------
    units = tuple(text_units(text, unit=hitmap.unit, openers=openers) for text in texts)
    text_set = ablatable_summary(units)
    variants = build_variants(texts, units)
    sample = UprDecisionSample(
        texts=tuple(texts),
        labels=tuple(labels),
        gold=gold,
        rows=tuple(range(len(texts))),
    )
    arms: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
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
        # The intact texts: the same grid as the CS-9b run, so these calls are
        # that run's cache hits.
        full_matrix, full_cost = score_upr_grid(
            sample,
            decider,
            arm,
            max_questions_per_call=decision.max_questions_per_call,
        )
        blind_matrix, blind_cost = score_upr_grid(
            sample,
            decider,
            arm,
            max_questions_per_call=decision.max_questions_per_call,
            mismatched=True,
        )
        # The ablated texts: one variant per removable unit.
        ablated = np.zeros((len(variants), len(labels)), dtype=np.float64)
        blind_ablated = np.zeros((len(variants), len(labels)), dtype=np.float64)
        n_calls = 0
        blind_calls = 0
        truncated: list[bool] = []
        blind_truncated: list[bool] = []
        if hitmap.include_ablation:
            cells = build_ablation_cells(variants, labels)
            probabilities, truncated, n_calls = score_cells(
                decider,
                cells,
                arm,
                max_questions_per_call=decision.max_questions_per_call,
            )
            ablated = np.asarray(probabilities, dtype=np.float64).reshape(
                len(variants), len(labels)
            )
            blind_cells = build_ablation_cells(variants, labels, mismatched=True)
            blind_probabilities, blind_truncated, blind_calls = score_cells(
                decider,
                blind_cells,
                arm,
                max_questions_per_call=decision.max_questions_per_call,
            )
            blind_ablated = np.asarray(blind_probabilities, dtype=np.float64).reshape(
                len(variants), len(labels)
            )
        measurement = measure_ablation(
            texts=texts,
            labels=labels,
            gold=gold,
            units=units,
            variants=variants,
            full_scores=full_matrix,
            ablated_scores=ablated,
            blind_full_scores=blind_matrix,
            blind_ablated_scores=blind_ablated,
            resamples=hitmap.bootstrap.resamples,
            seed=bootstrap_seed,
        )
        if not hitmap.include_ablation:
            measurement = {
                "applicable": False,
                "note": (
                    "the ablation map was not computed in this run "
                    "(upr_analysis.hitmap.include_ablation is false): the "
                    "reads below are the ones the surrogate's cross-check "
                    "needs (the carrier clauses), not a measurement"
                ),
            }
        info = decider.info
        block = {
            "model": {
                "name": model_config.name,
                "model_id": info.model_id,
                "digest": info.digest,
                "backend": info.backend,
                "host": info.host,
                "prompt_version": decision.prompt_version,
                "extra": info.extra,
            },
            "arm": {
                "state": arm.state,
                "question_type": arm.question_type,
                "instructions": arm.instructions,
                "positive_option": arm.positive_option,
                "max_input_tokens": arm.max_input_tokens,
            },
            "cost": {
                "intact_texts": {
                    "n_cells": full_cost["n_cells"],
                    "n_calls_planned": full_cost["n_calls_planned"],
                    "n_truncated": full_cost["truncation"]["n_truncated"],
                    "note": (
                        "the intact texts' calls: the same grid and the same "
                        "prompt as the CS-9b run, so they are that run's cache "
                        "hits"
                    ),
                },
                "ablations": {
                    "n_variants": len(variants),
                    "n_calls_planned": n_calls,
                    "n_truncated": int(sum(truncated)),
                },
                "label_blind_ablations": {
                    "n_variants": len(variants),
                    "n_calls_planned": blind_calls,
                    "n_truncated": int(sum(blind_truncated)),
                },
                "cache": {
                    "hits": getattr(decider, "cache_hits", 0),
                    "misses": getattr(decider, "cache_misses", 0),
                    "stored": getattr(decider, "n_stored", 0),
                },
                "usage": dict(getattr(decider, "usage", {})),
                "blind_intact_cost": blind_cost,
            },
            "measurement": measurement,
        }
        if hitmap.include_surrogate:
            block["surrogate"] = _hitmap_surrogate_block(
                setup,
                decider,
                arm,
                decision=decision,
                units=units,
                labels=labels,
                gold=gold,
                hitmap=hitmap,
                sample=sample,
            )
        arms[model_config.name] = block
        rows.append(
            _hitmap_upr_row(
                block,
                run_id=run_id,
                split=setup.split,
                unit=hitmap.unit,
                text_set=text_set,
                lexical_block=lexical_block,
                text_side_block=text_side_block,
                seconds=round(time.perf_counter() - started, 3),
            )
        )

    results_dir = config.paths.results / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    payload = {
        "run_id": run_id,
        "command": "hitmap-upr",
        "split": setup.split,
        "source": str(config.paths.data_raw),
        "source_sha256": setup.source_sha256,
        "text_set": {
            "scope": "the whole evaluated split (pre-registered: the text set "
            "is never chosen after looking at the results)",
            "split": setup.split,
            "n_texts": len(texts),
            "n_labels": len(labels),
            "n_gold_cells": int(gold.sum()),
            "unit": hitmap.unit,
            **text_set,
            "held_out_labels": held_out,
            "label_buckets": label_bucket_map(buckets, labels),
        },
        "lexical_map": lexical_block,
        "text_side_map": text_side_block,
        "ablation_map": {"arms": arms},
        "config": {
            "unit": hitmap.unit,
            "include_lexical_map": hitmap.include_lexical_map,
            "include_label_blind": hitmap.include_label_blind,
            "catch_all_labels": list(hitmap.catch_all_labels),
            "bootstrap": {
                "resamples": hitmap.bootstrap.resamples,
                "seed": bootstrap_seed,
            },
            "openers": list(openers),
            "models": [model.model_dump() for model in decision.models],
            "reference_arm": {
                "predictor": setup.predictor.describe(),
                "note": (
                    "the pipeline's own scoring of the same texts, so the map's "
                    "reference arm is the pipeline's numbers and not a "
                    "re-implementation of them"
                ),
            },
            "notes": [
                "a pack hit map is a map of the WORD-OVERLAP feature, not of the embedding model.",
                "no per-word attribution is read out of the boosted-tree "
                "ensemble: a tree ensemble has no per-word attribution.",
                "the decision model is frozen (id and digest pinned) and "
                "carries no UPR label: it is the same tev1 the CS-13 feature "
                "used, so the map and the feature are read against one model.",
                "attribution is not causation unless you perturb: the causal "
                "read comes from the ablation (Version B), never from the "
                "feature's own decomposition (Version A).",
            ],
        },
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (results_dir / "hitmap.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _append_command_rows(
        config,
        "hitmap-upr",
        columns=HITMAP_UPR_COLUMNS,
        key_columns=HITMAP_UPR_KEY_COLUMNS,
        rows=rows,
    )

    print(f"UPR hit map (CS-18): {run_id}")
    print(
        f"  unit: {hitmap.unit}; {len(texts)} texts, {text_set['n_ablatable']} ablatable, "
        f"{text_set['n_not_ablatable']} not ablatable, {len(variants)} variants"
    )
    if lexical_block is not None:
        held_out_block = lexical_block["held_out_labels"]
        print(
            f"  lexical map: the held-out labels' generic share "
            f"{_fmt(held_out_block['generic_share'], 4)}, the top word's share "
            f"{_fmt(lexical_block['concentration']['top1_word_share'], 4)}"
        )
    for name, block in arms.items():
        reads = block["measurement"].get("reads")
        if reads is None:
            print(f"  [{name}] the ablation map was not computed in this run")
            continue
        linked = reads["label_linked_vs_other"]
        formulaic = reads["formulaic_vs_topical"]
        falsification = block["measurement"]["falsification"]
        print(
            f"  [{name}] the label-linked drop {_fmt(linked.get('left_mean'), 4)} against "
            f"{_fmt(linked.get('right_mean'), 4)} (the difference "
            f"{_fmt(linked.get('difference'), 4)}); the formulaic drop "
            f"{_fmt(formulaic.get('left_mean'), 4)} against the topical "
            f"{_fmt(formulaic.get('right_mean'), 4)}"
        )
        print(
            "      falsification: the label-linked drop larger "
            f"{'yes' if falsification['label_linked_drop_greater']['holds'] else 'no'}, "
            "the formulaic drop smaller "
            f"{'yes' if falsification['formulaic_drop_smaller']['holds'] else 'no'}, "
            "the attribution label-driven "
            f"{'yes' if falsification['label_driven']['holds'] else 'no'}"
        )
    print(f"Report written to {results_dir / 'hitmap.json'}")
    return 0


def _hitmap_surrogate_block(
    setup: EvalSetup,
    decider: Any,
    arm: DecisionArm,
    *,
    decision: Any,
    units: Sequence[Sequence[Any]],
    labels: Sequence[str],
    gold: np.ndarray,
    hitmap: Any,
    sample: UprDecisionSample,
) -> dict[str, Any]:
    """The **surrogate attribution** (the map's mode C) and its validation.

    A ridge surrogate of the decision model's own probability: the text's words
    in, the model's confidence out, fitted on the **train** split (the CS-9b
    run's fit side — every call a cache hit) and read on the val split. Its
    per-topic coefficients are the map's per-topic word attribution.

    Three validations ride along: the surrogate's **out-of-sample quality**
    (does it track the model at all?), the **topic-faithfulness** check (are
    the top words the topic's own name words, and does the label-blind
    counterpart's attribution follow the *shifted* topic instead?), and the
    **agreement with the causal map** (do the surrogate's top words fall inside
    the clause the ablation identified as the carrier?).
    """
    train_texts = [record.text for record in setup.train_records]
    fit_sample = UprDecisionSample(
        texts=tuple(train_texts),
        labels=tuple(labels),
        gold=build_label_matrix(setup.train_records, list(labels)),
        rows=tuple(range(len(train_texts))),
    )
    train_matrix, train_cost = score_upr_grid(
        fit_sample,
        decider,
        arm,
        max_questions_per_call=decision.max_questions_per_call,
    )
    train_blind, _ = score_upr_grid(
        fit_sample,
        decider,
        arm,
        max_questions_per_call=decision.max_questions_per_call,
        mismatched=True,
    )
    val_matrix, _ = score_upr_grid(
        sample,
        decider,
        arm,
        max_questions_per_call=decision.max_questions_per_call,
    )
    val_blind, _ = score_upr_grid(
        sample,
        decider,
        arm,
        max_questions_per_call=decision.max_questions_per_call,
        mismatched=True,
    )

    vocabulary = build_vocabulary(train_texts, min_document_frequency=hitmap.surrogate_min_df)
    train_design = build_design_matrix(train_texts, vocabulary)
    val_design = build_design_matrix(list(sample.texts), vocabulary)
    fit = fit_ridge_surrogate(train_design, train_matrix, ridge_lambda=hitmap.surrogate_lambda)
    blind_fit = fit_ridge_surrogate(train_design, train_blind, ridge_lambda=hitmap.surrogate_lambda)
    quality = surrogate_quality(surrogate_predictions(fit, val_design), val_matrix)
    blind_quality = surrogate_quality(surrogate_predictions(blind_fit, val_design), val_blind)
    attribution = topic_attribution(
        fit, vocabulary, list(labels), top_words=hitmap.surrogate_top_words
    )
    blind_attribution = topic_attribution(
        blind_fit, vocabulary, list(labels), top_words=hitmap.surrogate_top_words
    )

    # the topic-faithfulness check: the real attribution's top words should be
    # the topic's own name words, and the blind counterpart's should follow the
    # *shifted* topic instead
    n_labels = len(labels)
    real_own: list[float] = []
    blind_own: list[float] = []
    blind_control: list[float] = []
    for index, label in enumerate(labels):
        share = name_word_share(
            attribution[index], label, list(labels), top_words=hitmap.surrogate_top_words
        )
        if share is not None:
            real_own.append(share)
        shifted = labels[(index + 1) % n_labels]
        share = name_word_share(
            blind_attribution[index],
            shifted,
            list(labels),
            top_words=hitmap.surrogate_top_words,
        )
        if share is not None:
            blind_own.append(share)
        share = name_word_share(
            blind_attribution[index],
            label,
            list(labels),
            top_words=hitmap.surrogate_top_words,
        )
        if share is not None:
            blind_control.append(share)

    # the cross-check with the causal map: the carrier clause per (text, topic)
    carriers: dict[tuple[int, int], str] = {}
    for row, row_units in enumerate(units):
        if len(row_units) < 2:
            continue
        for column in np.flatnonzero(gold[row]):
            linked = label_linked_units(row_units, labels[int(column)])
            if linked:
                carriers[(row, int(column))] = row_units[linked[0]].text
    agreement = agreement_with_carrier(
        attribution,
        carriers,
        top_words=hitmap.surrogate_top_words,
        text_words=dict(enumerate(sample.texts)),
    )

    return {
        "method": (
            "a ridge surrogate of the decision model's own probability: the "
            "text's content words in, the model's confidence out. It is an "
            "ATTRIBUTION (which words predict the model's score), never a "
            "causal claim -- the causal half of the map is the ablation, and "
            "the two are cross-checked against each other here."
        ),
        "fit": {
            "side": "the train split (no val text and no val label enters the fit)",
            "n_texts": len(train_texts),
            "n_topics": n_labels,
            "n_features": len(vocabulary),
            "ridge_lambda": hitmap.surrogate_lambda,
            "min_document_frequency": hitmap.surrogate_min_df,
            "feature": "the word's presence in the text (not its count)",
            "note": (
                "the vocabulary is the train corpus's words: a word that only "
                "appears in val never becomes a feature"
            ),
        },
        "cost": {
            "train_calls_planned": train_cost["n_calls_planned"],
            "note": (
                "the fit side's calls are the CS-9b run's fit-side cache hits, "
                "so the surrogate costs no new model call"
            ),
        },
        "quality": quality,
        "per_topic": attribution,
        "topic_faithfulness": {
            "the real attribution's own-name share": round(float(np.mean(real_own)), 6)
            if real_own
            else None,
            "the blind counterpart's own-name share": round(float(np.mean(blind_own)), 6)
            if blind_own
            else None,
            "the blind counterpart's share of the asked topic": round(
                float(np.mean(blind_control)), 6
            )
            if blind_control
            else None,
            "note": (
                "the real attribution's top words should be the topic's own "
                "name words; the label-blind counterpart is asked about the "
                "*shifted* topic, so its top words should be that topic's words "
                "and NOT the asked one -- the attribution follows the pairing"
            ),
        },
        "agreement_with_the_causal_map": agreement,
        "label_blind_counterpart": {
            "quality": blind_quality,
            "per_topic": blind_attribution,
        },
    }


def _hitmap_upr_row(
    block: dict[str, Any],
    *,
    run_id: str,
    split: str,
    unit: str,
    text_set: dict[str, Any],
    lexical_block: dict[str, Any] | None,
    text_side_block: dict[str, Any] | None,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the hit map (the run's arm)."""
    reads = block["measurement"].get("reads") or {}
    blind = block["measurement"].get("label_blind_counterpart") or {}
    falsification = block["measurement"].get("falsification") or {}
    linked = reads.get("label_linked_vs_other") or {}
    formulaic = reads.get("formulaic_vs_topical") or {}
    gold = reads.get("gold_vs_background") or {}
    blind_linked = blind.get("label_linked_vs_other") or {}
    blind_formulaic = blind.get("formulaic_vs_topical") or {}
    held_out = (lexical_block or {}).get("held_out_labels") or {}
    concentration = (lexical_block or {}).get("concentration") or {}
    text_side = text_side_block or {}
    surrogate = block.get("surrogate") or {}
    surrogate_quality = surrogate.get("quality") or {}
    faithfulness = surrogate.get("topic_faithfulness") or {}
    agreement = surrogate.get("agreement_with_the_causal_map") or {}

    def _interval(read: dict[str, Any]) -> str | None:
        interval = read.get("interval")
        if not interval:
            return None
        return f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"

    def _flag(name: str) -> str:
        block_ = falsification.get(name) or {}
        if "holds" not in block_:
            return "n/a"  # the flag was not measured in this run
        return "yes" if block_.get("holds") else "no"

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "split": split,
        "unit": unit,
        "model": block["model"]["name"],
        "n_texts": text_set.get("n_texts"),
        "n_ablatable": text_set.get("n_ablatable"),
        "share_not_ablatable": _fmt(text_set.get("share_not_ablatable"), 4),
        "n_variants": block["cost"]["ablations"]["n_variants"],
        "n_calls": block["cost"]["ablations"]["n_calls_planned"],
        "linked_drop": _fmt(linked.get("left_mean"), 4),
        "other_drop": _fmt(linked.get("right_mean"), 4),
        "linked_minus_other": _fmt(linked.get("difference"), 4),
        "linked_ci": _interval(linked),
        "formulaic_drop": _fmt(formulaic.get("left_mean"), 4),
        "topical_drop": _fmt(formulaic.get("right_mean"), 4),
        "formulaic_minus_topical": _fmt(formulaic.get("difference"), 4),
        "gold_drop": _fmt(gold.get("left_mean"), 4),
        "background_drop": _fmt(gold.get("right_mean"), 4),
        "gold_minus_background": _fmt(gold.get("difference"), 4),
        "blind_linked_minus_other": _fmt(blind_linked.get("difference"), 4),
        "blind_formulaic_minus_topical": _fmt(blind_formulaic.get("difference"), 4),
        "linked_ok": _flag("label_linked_drop_greater"),
        "formulaic_ok": _flag("formulaic_drop_smaller"),
        "label_driven": _flag("label_driven"),
        "heldout_generic_share": _fmt(held_out.get("generic_share"), 4),
        "top1_word_share": _fmt(concentration.get("top1_word_share"), 4),
        "text_side_exactness": _fmt((text_side.get("exactness") or {}).get("max_abs_error"), 6),
        "text_side_top1_share": _fmt(
            (text_side.get("concentration") or {}).get("top1_word_share"), 4
        ),
        "surrogate_r": _fmt(surrogate_quality.get("mean_correlation"), 4),
        "surrogate_own_name_share": _fmt(
            faithfulness.get("the real attribution's own-name share"), 4
        ),
        "surrogate_carrier_agreement": _fmt(
            agreement.get("share_inside_among_the_words_in_the_text"), 4
        ),
        "surrogate_chance_level": _fmt(agreement.get("chance_level"), 4),
        "seconds": seconds,
    }


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
