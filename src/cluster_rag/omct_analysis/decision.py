"""Stage 3.5: the decision models and the deliverable's rule.

The decision model scores the same-document grid; the **safe-negative
rule** (the track's named deliverable) is measured against it, with the
label-blind counterpart as the yardstick, and the **shortlist rerank**
re-orders the geometry's candidates by the model's probability.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.omct_analysis.arms import (
    _control_indices,
    _pair_indices,
    measure_arm,
    score_arm,
)
from cluster_rag.omct_analysis.helpers import (
    COSINE_FEATURE_NAME,
    DECISION_ARM,
    DECISION_FIXED_CUT,
    DECISION_RULE_FIT_THRESHOLD,
    DECISION_RULE_FIXED_CUT,
    DECISION_RULE_MARGIN,
    DECISION_RULE_VARIANTS,
    HIT_KS,
    STAGE3_ARM,
    _difference,
    _mean_of,
    _weighted_mean,
)
from cluster_rag.omct_analysis.stage3 import (
    Stage3Context,
    Stage3Grid,
    _assemble_reporting,
    _cell_positions,
    _cells_by_document,
    _feature_diagnostics,
    _length_check,
    _margins_of_document,
    _safe_margin_rule,
    _stage3_rotation,
    build_stage3_grid,
    compare_stage3_arm,
)
from cluster_rag.omct_analysis.view import (
    DatasetView,
    ReportingSpec,
    SideEncoding,
)
from cluster_rag.text_processors.base import Combiner, FeatureExtractor
from cluster_rag.text_processors.calibration import (
    THRESHOLD_CRITERION_YOUDEN,
    choose_threshold,
    harm_at_coverage,
    margin_candidates,
    margin_curve,
    rule_counts,
    threshold_candidates,
)

logger = logging.getLogger(__name__)


def score_decision_arm(
    view: DatasetView,
    *,
    decision: Any,
    arm_name: str = DECISION_ARM,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """The decision model's arm over the whole same-document grid (CS-7b).

    The block is the **existing arm's shape** (the pair AUC pooled and per
    document, the within/cross decomposition, the score distributions, the
    rank percentiles, the hit@k, the top-1 provenance, the length
    correlation), so the decision arm's numbers sit next to the embedding
    arms' in the same artifact and the same table.

    Two things are *not* comparable with the embedding arms':

    - the **level** numbers (a probability against a cosine): the scales are
      different units, exactly as the CS-6e note says for the learned arm's
      probability against the cosine's;
    - nothing else: the rank numbers are measured in the **same** candidate
      lists (the paragraph's document's questions), because the arm scores the
      whole grid.
    """
    paragraph_side = SideEncoding(
        texts=tuple(view.paragraphs),
        chunk_lists=tuple((text,) for text in view.paragraphs),
        vectors=np.zeros((len(view.paragraphs), 1), dtype=np.float32),
        meta={
            "kind": "decision",
            "scorer": "a decision model (Ollama's decision-model type)",
            "n_cells": int(decision.n_cells),
        },
    )
    question_side = SideEncoding(
        texts=tuple(view.questions),
        chunk_lists=tuple((text,) for text in view.questions),
        vectors=np.zeros((len(view.questions), 1), dtype=np.float32),
        meta={"kind": "decision", "n_cells": int(decision.n_cells)},
    )
    block = score_arm(
        view,
        arm_name=arm_name,
        similarities=decision.matrix,
        paragraph_side=paragraph_side,
        question_side=question_side,
        hit_ks=hit_ks,
    )
    block["decision"] = dict(decision.metadata)
    return block


def _decision_rule_masks(
    view: DatasetView, grid: Stage3Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid's masks: the judged cells, the verified positives, the unjudged ones."""
    judged = np.zeros(grid.n_cells, dtype=bool)
    positive = np.zeros(grid.n_cells, dtype=bool)
    for record in view.pairs:
        position = grid.index[
            (view.paragraph_row[record.paragraph_text], view.question_col[record.question_text])
        ]
        judged[position] = True
        positive[position] = record.label == 1
    return judged, positive, ~judged


def shortlist_rerank_block(
    view: DatasetView,
    *,
    reference_similarities: np.ndarray,
    decision_matrix: np.ndarray,
    ks: Sequence[int],
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """The shortlist rerank: retrieve with the cosine, rank with the decision model.

    The **shortlist** is the cosine's top-K candidates of a paragraph (the
    geometry's own shortlist, the same one PLAN's Phase-6 reranker would
    consume). The **rerank** reorders that shortlist by the decision model's
    probability, and the metric is the reviewed set's positives' **hit@1**:
    does the true match end up first among the paragraph's candidates?

    Three numbers frame it:

    - the **cosine's own hit@1** (the shortlist's first candidate, over the
      whole candidate list): what the geometry does alone;
    - the **decision model's own hit@1** (its first candidate over the whole
      grid): what the decision model does alone, i.e. "rerank everything";
    - the **reranked hit@1**: what the hybrid does, at the cost of the
      decision calls the shortlist needs (``K`` per paragraph) instead of
      the whole grid.

    The hit@K is the shortlist's own recall — the **ceiling** the rerank
    cannot beat, since a rerank reorders the shortlist but does not enlarge
    it.
    """
    reference = np.asarray(reference_similarities, dtype=np.float64)
    decision = np.asarray(decision_matrix, dtype=np.float64)
    candidate_columns = view.candidate_columns()

    # the cosine's own ranking of every paragraph's candidates
    cosine_ranks: list[list[int]] = []
    decision_ranks: list[list[int]] = []
    for row, candidates in enumerate(candidate_columns):
        ordered = sorted(candidates, key=lambda col: -float(reference[row, col]))
        cosine_ranks.append(ordered)
        decision_ranks.append(sorted(ordered, key=lambda col: -float(decision[row, col])))

    judged_rows = [view.paragraph_row[record.paragraph_text] for record in view.pairs]
    judged_columns = [view.question_col[record.question_text] for record in view.pairs]
    labels = [record.label for record in view.pairs]
    documents = [record.document for record in view.pairs]
    control_rows = [row for _, row, _ in view.control.cells]
    control_columns = [column for _, _, column in view.control.cells]

    def _first_after_rerank(row: int, column: int, k: int) -> bool:
        """Is the cell the paragraph's reranked top-1 inside the shortlist?"""
        shortlist = cosine_ranks[row][:k]
        if column not in shortlist:
            return False
        reranked = decision_ranks[row]
        best = min(
            (candidate for candidate in shortlist),
            key=lambda candidate: reranked.index(candidate),
        )
        return best == column

    def _in_shortlist(row: int, column: int, k: int) -> bool:
        """Is the cell inside the cosine's top-K shortlist?"""
        return column in cosine_ranks[row][:k]

    def _own_first(row: int, column: int, ranking: list[list[int]]) -> bool:
        """Is the cell the paragraph's own top-1 under the given ranking?"""
        ordered = ranking[row]
        return bool(ordered) and ordered[0] == column

    blocks: dict[str, Any] = {}
    for k in ks:
        positive_hits = [
            _first_after_rerank(row, column, k)
            for row, column, label in zip(judged_rows, judged_columns, labels, strict=True)
            if label == 1
        ]
        negative_hits = [
            _first_after_rerank(row, column, k)
            for row, column, label in zip(judged_rows, judged_columns, labels, strict=True)
            if label == 0
        ]
        ceilings = [
            _in_shortlist(row, column, k)
            for row, column, label in zip(judged_rows, judged_columns, labels, strict=True)
            if label == 1
        ]
        control_hits = [
            _first_after_rerank(row, column, k)
            for row, column in zip(control_rows, control_columns, strict=True)
        ]
        by_document: dict[str, Any] = {}
        for document in sorted(set(documents)):
            hits = [
                _first_after_rerank(row, column, k)
                for row, column, other, label in zip(
                    judged_rows, judged_columns, documents, labels, strict=True
                )
                if other == document and label == 1
            ]
            by_document[document] = {
                "n_positives": len(hits),
                "hit1": float(np.mean(hits)) if hits else None,
            }
        document_aucs = [
            block["hit1"] for block in by_document.values() if block["hit1"] is not None
        ]
        blocks[str(k)] = {
            "k": int(k),
            "n_shortlist_cells": int(sum(min(k, len(columns)) for columns in candidate_columns)),
            "positives_hit1": float(np.mean(positive_hits)) if positive_hits else None,
            "positives_hit1_macro": (float(np.mean(document_aucs)) if document_aucs else None),
            "positives_hitk_ceiling": float(np.mean(ceilings)) if ceilings else None,
            "negatives_hit1": float(np.mean(negative_hits)) if negative_hits else None,
            "control_hit1": float(np.mean(control_hits)) if control_hits else None,
            "by_document": by_document,
        }

    def _own_hit1(ranking: list[list[int]]) -> float:
        """The positives' own hit@1 under a ranking (the rerank's convention)."""
        hits = [
            _own_first(row, column, ranking)
            for row, column, label in zip(judged_rows, judged_columns, labels, strict=True)
            if label == 1
        ]
        return float(np.mean(hits)) if hits else 0.0

    return {
        "definition": {
            "shortlist": (
                "the cosine's top-K candidates of a paragraph (the geometry's own shortlist)"
            ),
            "rerank": ("the same shortlist, reordered by the decision model's probability"),
            "metric": (
                "the reviewed set's positives' hit@1: does the true match end up "
                "first among the paragraph's candidates? The hit@K is the "
                "shortlist's own recall — the ceiling a rerank cannot beat"
            ),
        },
        "references": {
            "cosine_hit1": _own_hit1(cosine_ranks),
            "decision_hit1": _own_hit1(decision_ranks),
            "note": (
                "the cosine's own hit@1 (the geometry alone) and the decision "
                "model's own hit@1 (the decision model alone, i.e. reranking "
                "everything): the rerank sits between them"
            ),
        },
        "by_k": blocks,
        "hit_ks": list(hit_ks),
    }


def decision_rule_block(
    view: DatasetView,
    *,
    grid: Stage3Grid,
    scores: np.ndarray,
    mismatched_scores: np.ndarray | None = None,
    reporting: ReportingSpec,
    fixed_cut: float = DECISION_FIXED_CUT,
) -> dict[str, Any]:
    """The safe-negative rule with the decision model's probability (CS-7b).

    Three forms, all measured the way CS-6e measured the rule (the same
    document-level holdout, the same fit-side choices, the same pooled
    numbers over the whole unjudged grid):

    - the **fixed cut** (the probability below which the model calls the pair
      a non-match): the operating point a calibrated model makes meaningful,
      and the one the deliverable would ship;
    - the **fit-side threshold** (the same cut, but fitted on the fit side and
      applied to the holdout): the old way, beside it;
    - the **margin rule** (the cell at least X below its own paragraph's best
      candidate, X fitted on the fit side): the existing rule's shape with the
      decision model's probability as the ranking signal.

    Beside every form: the **label-blind counterpart** — the same model, the
    same prompts, but every cell's question replaced by another question of
    the same document. It is the control-role floor's counterpart for a model
    that is not fitted on anything: it keeps the model's shape, its
    calibration and its cost, and loses only the pairing.

    The unjudged cells' true positive rate stays **unmeasured** (no human
    labelling in this track): the numbers are the rule's reach and its harm
    on the verified positives, never its precision.
    """
    values = np.asarray(scores, dtype=np.float64)
    blind = None if mismatched_scores is None else np.asarray(mismatched_scores, dtype=np.float64)
    judged_mask, positive_mask, unjudged_mask = _decision_rule_masks(view, grid)
    cells_by_document = _cells_by_document(grid)
    documents = sorted(cells_by_document)

    def _margins(document: str, source: np.ndarray) -> np.ndarray:
        """The margins of one document's cells (the cell's distance below its row's best)."""
        cells = cells_by_document[document]
        return _margins_of_document(grid.rows[cells], grid.columns[cells], source[cells])

    def _counterpart(
        *,
        coverage: float | None,
        harm: float | None,
        matched_harm: float | None,
        note: str,
    ) -> dict[str, Any]:
        """The label-blind counterpart's numbers (the same shape as the floor's)."""
        return {
            "n_draws": 1 if blind is not None else 0,
            "per_draw": (
                [
                    {
                        "coverage": coverage,
                        "harm": harm,
                        "harm_at_matched_coverage": matched_harm,
                    }
                ]
                if blind is not None
                else []
            ),
            "mean": {"coverage": coverage, "harm": harm},
            "harm_at_matched_coverage": {
                "coverage": coverage,
                "harm": matched_harm,
                "n_draws_compared": 1 if blind is not None else 0,
            },
            "note": note,
        }

    per_document: dict[str, Any] = {}
    for holdout in documents:
        fit_side_documents = [document for document in documents if document != holdout]
        cells = cells_by_document[holdout]
        holdout_unjudged = unjudged_mask[cells]
        holdout_positive = positive_mask[cells]
        holdout_values = values[cells]
        fit_cells = np.concatenate([cells_by_document[document] for document in fit_side_documents])
        fit_unjudged = unjudged_mask[fit_cells]
        fit_positive = positive_mask[fit_cells]
        fit_values = values[fit_cells]

        variants: dict[str, Any] = {}

        # -- the fixed cut ------------------------------------------------
        cleared = holdout_values < fixed_cut
        counts = rule_counts(cleared, unjudged=holdout_unjudged, positives=holdout_positive)
        blind_counts = None
        if blind is not None:
            blind_cleared = blind[cells] < fixed_cut
            blind_counts = rule_counts(
                blind_cleared, unjudged=holdout_unjudged, positives=holdout_positive
            )
        variants[DECISION_RULE_FIXED_CUT] = {
            "space": (
                "the decision model's probability, cut at a FIXED "
                f"{fixed_cut:g} (the calibrated model's own operating point)"
            ),
            "fit_side": {
                "cut": fixed_cut,
                "note": (
                    "no fitting: the cut is the model's own probability scale, "
                    "which the calibration block measures (CS-7a: the ECE "
                    "0.089-0.137 pooled for tev1)"
                ),
            },
            "holdout": {"counts": counts},
            "counterpart": _counterpart(
                coverage=None if blind_counts is None else blind_counts["coverage"],
                harm=None if blind_counts is None else blind_counts["harm"],
                matched_harm=None if blind_counts is None else blind_counts["harm"],
                note=(
                    "the label-blind counterpart: the same model with every cell's "
                    "question replaced by another question of the same document, "
                    "so it keeps the model's shape and its calibration and loses "
                    "only the pairing. Its harm is at the same cut."
                ),
            ),
        }

        # -- the fit-side threshold ---------------------------------------
        threshold_block: dict[str, Any] = {"threshold": None, "error": None}
        if reporting.threshold_criterion == THRESHOLD_CRITERION_YOUDEN:
            # the reviewer-verified hard negatives (the reviewed set's own
            # ground truth) as the negative class
            negative_class = "the reviewer-verified hard negatives (subset A, ground truth)"
        else:
            # the unjudged ordinary cells (the rule's target negative class)
            negative_class = "the unjudged ordinary cells (presumed negatives)"
        if fit_values.size:
            positives_mask = positive_mask[fit_cells]
            if reporting.threshold_criterion == THRESHOLD_CRITERION_YOUDEN:
                negatives_mask = judged_mask[fit_cells] & ~positive_mask[fit_cells]
            else:
                negatives_mask = unjudged_mask[fit_cells]
            candidates = threshold_candidates(fit_values, n_quantiles=reporting.threshold_quantiles)
            try:
                choice = choose_threshold(
                    fit_values,
                    positives=positives_mask,
                    negatives=negatives_mask,
                    candidates=candidates,
                )
                threshold = float(choice["threshold"])
                threshold_block = {
                    "threshold": threshold,
                    "criterion_value": choice["value"],
                    "tpr": choice["tpr"],
                    "fpr": choice["fpr"],
                    "n_candidates": len(candidates),
                    "curve": choice["curve"],
                    "error": None,
                }
            except ValueError as error:
                threshold_block = {"threshold": None, "error": str(error)}
        threshold = threshold_block.get("threshold")
        if "negative_class" not in threshold_block:
            threshold_block["negative_class"] = negative_class
        if threshold is not None:
            threshold_cleared = holdout_values < threshold
            counts = rule_counts(
                threshold_cleared, unjudged=holdout_unjudged, positives=holdout_positive
            )
            blind_counts = None
            if blind is not None:
                blind_cleared = blind[cells] < threshold
                blind_counts = rule_counts(
                    blind_cleared, unjudged=holdout_unjudged, positives=holdout_positive
                )
        else:
            counts = rule_counts(
                np.zeros_like(holdout_values, dtype=bool),
                unjudged=holdout_unjudged,
                positives=holdout_positive,
            )
            blind_counts = None
        variants[DECISION_RULE_FIT_THRESHOLD] = {
            "space": (
                "the decision model's probability, cut at a threshold chosen on "
                "the FIT SIDE and applied to the holdout (the old way)"
            ),
            "fit_side": {
                **threshold_block,
                "criterion": reporting.threshold_criterion,
                "negative_class": negative_class,
                "note": (
                    "the threshold is chosen on the fit side (the other two "
                    "documents) by the report's own criterion and applied to the "
                    "held-out one"
                ),
            },
            "holdout": {"counts": counts},
            "counterpart": _counterpart(
                coverage=None if blind_counts is None else blind_counts["coverage"],
                harm=None if blind_counts is None else blind_counts["harm"],
                matched_harm=None if blind_counts is None else blind_counts["harm"],
                note=(
                    "the label-blind counterpart at the same threshold: the same "
                    "model with every cell's question replaced by another question "
                    "of the same document"
                ),
            ),
        }

        # -- the margin rule ----------------------------------------------
        fit_margins = np.concatenate(
            [_margins(document, values) for document in fit_side_documents]
        )
        holdout_margins = _margins(holdout, values)
        margin_grid = (
            margin_candidates(fit_margins[fit_unjudged], n_quantiles=reporting.margin_quantiles)
            if fit_unjudged.any()
            else np.asarray([], dtype=np.float64)
        )
        margin_choice, margin_error = _safe_margin_rule(
            fit_margins,
            unjudged=fit_unjudged,
            positives=fit_positive,
            n_quantiles=reporting.margin_quantiles,
        )
        margin = margin_choice["margin"] if margin_choice else None
        margin_cleared = (
            holdout_margins >= margin
            if margin is not None
            else np.zeros_like(holdout_margins, dtype=bool)
        )
        margin_counts = rule_counts(
            margin_cleared, unjudged=holdout_unjudged, positives=holdout_positive
        )
        margin_curve_values = margin_curve(
            holdout_margins,
            unjudged=holdout_unjudged,
            positives=holdout_positive,
            candidates=margin_grid,
        )
        blind_matched = None
        blind_coverage = None
        blind_harm = None
        if blind is not None:
            blind_fit_margins = np.concatenate(
                [_margins(document, blind) for document in fit_side_documents]
            )
            blind_holdout_margins = _margins(holdout, blind)
            blind_choice, _ = _safe_margin_rule(
                blind_fit_margins,
                unjudged=fit_unjudged,
                positives=fit_positive,
                n_quantiles=reporting.margin_quantiles,
            )
            if blind_choice is not None:
                blind_cleared = blind_holdout_margins >= blind_choice["margin"]
                blind_counts = rule_counts(
                    blind_cleared, unjudged=holdout_unjudged, positives=holdout_positive
                )
                blind_coverage = blind_counts["coverage"]
                blind_harm = blind_counts["harm"]
                blind_curve = margin_curve(
                    blind_holdout_margins,
                    unjudged=holdout_unjudged,
                    positives=holdout_positive,
                    candidates=margin_grid,
                )
                blind_matched = harm_at_coverage(blind_curve, margin_counts["coverage"])
        variants[DECISION_RULE_MARGIN] = {
            "space": (
                "the decision model's margin: the cell's distance below its own "
                "paragraph's best candidate (the existing rule's shape with the "
                "decision model's probability as the ranking signal)"
            ),
            "fit_side": {
                "margin": margin,
                "criterion_value": margin_choice["value"] if margin_choice else None,
                "coverage": margin_choice["coverage"] if margin_choice else None,
                "harm": margin_choice["harm"] if margin_choice else None,
                "n_candidates": margin_choice["n_candidates"] if margin_choice else 0,
                "error": margin_error,
                "note": (
                    "the margin is chosen on the fit side by the rule's own "
                    "criterion — the coverage over the fit side's unjudged cells "
                    "minus the harm on the fit side's verified positives — and "
                    "applied to the held-out one"
                ),
            },
            "holdout": {"counts": margin_counts, "curve": margin_curve_values},
            "counterpart": _counterpart(
                coverage=blind_coverage,
                harm=blind_harm,
                matched_harm=blind_matched,
                note=(
                    "the label-blind counterpart: the same model with every cell's "
                    "question replaced by another question of the same document, "
                    "running the same rule (its own fit-side margin choice). Its "
                    "harm at the matched coverage is read off its own curve."
                ),
            ),
        }

        per_document[holdout] = {
            "fit_side_documents": fit_side_documents,
            "variants": variants,
        }

    # the pooled numbers: the three holdouts partition the grid
    pooled: dict[str, Any] = {}
    for variant in DECISION_RULE_VARIANTS:
        totals = {
            "n_unjudged_cells": 0,
            "n_unjudged_cleared": 0,
            "n_positives": 0,
            "n_positives_discarded": 0,
        }
        counterpart_coverages: list[float] = []
        counterpart_harms: list[float] = []
        counterpart_matched: list[float] = []
        for block in per_document.values():
            counts = block["variants"][variant]["holdout"]["counts"]
            for key in totals:
                totals[key] += int(counts[key])
            counterpart = block["variants"][variant]["counterpart"]
            counterpart_mean = counterpart.get("mean") or {}
            if counterpart_mean.get("coverage") is not None:
                counterpart_coverages.append(float(counterpart_mean["coverage"]))
            if counterpart_mean.get("harm") is not None:
                counterpart_harms.append(float(counterpart_mean["harm"]))
            matched = (counterpart.get("harm_at_matched_coverage") or {}).get("harm")
            if matched is not None:
                counterpart_matched.append(float(matched))
        coverage = (
            totals["n_unjudged_cleared"] / totals["n_unjudged_cells"]
            if totals["n_unjudged_cells"]
            else None
        )
        harm = (
            totals["n_positives_discarded"] / totals["n_positives"]
            if totals["n_positives"]
            else None
        )
        pooled[variant] = {
            **totals,
            "coverage": coverage,
            "harm": harm,
            "counterpart": {
                "mean": {
                    "coverage": (
                        float(np.mean(counterpart_coverages)) if counterpart_coverages else None
                    ),
                    "harm": float(np.mean(counterpart_harms)) if counterpart_harms else None,
                },
                "harm_at_matched_coverage": {
                    "coverage": coverage,
                    "harm": (float(np.mean(counterpart_matched)) if counterpart_matched else None),
                    "n_documents_compared": len(counterpart_matched),
                },
            },
            "note": (
                "the rule's reach over the WHOLE unjudged grid (every "
                "same-document cell without a record), pooled over the three "
                "documents, and the share of the verified positives the rule "
                "would discard — read against the label-blind counterpart"
            ),
        }

    return {
        "definition": {
            "fixed_cut": (
                "the decision model's probability below which the model calls "
                "the pair a non-match: a cell under the cut is a candidate safe "
                "negative"
            ),
            "margin": (
                "the cell's distance below its own paragraph's best candidate, "
                "inside the paragraph's document"
            ),
            "counterpart": (
                "the label-blind counterpart: the same model with every cell's "
                "question replaced by another question of the same document"
            ),
        },
        "per_document": per_document,
        "pooled": pooled,
        "unmeasured": (
            "the unjudged cells' true positive rate is UNMEASURED: no human "
            "labelling in this track. The numbers are the rule's reach and its "
            "harm on the verified positives, never its precision on the unjudged "
            "cells."
        ),
    }


def score_stage3_model(
    view: DatasetView,
    *,
    model_name: str,
    extractor: FeatureExtractor,
    combiner_factory: Callable[[], Combiner],
    reference_similarities: np.ndarray,
    control_weight: float = 0.3,
    permutation_resamples: int = 5,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
    hit_ks: Sequence[int] = HIT_KS,
    feature_embeddings: Mapping[str, Any] | None = None,
    calibration_map_factory: Callable[[], Combiner] | None = None,
    reporting: ReportingSpec | None = None,
) -> dict[str, Any]:
    """Score the learned pair scorer of one model (the Stage-3 arm).

    The protocol is the plan's Stage-3 discipline: **document-level holdout**.
    For every document the combiner is fitted on the other two documents'
    judged pairs (plus their unjudged control cells as down-weighted presumed
    negatives) and predicts the held-out document. The out-of-fold scores of
    every judged pair and of every control cell then form **one** score
    matrix, on which the same comparison blocks the Stage-2 runs use are
    computed (the pair AUC pooled / per document / macro, the within/cross
    decomposition and its paired bootstrap, the control-side counterpart, the
    de-shifted AUC, the floor gap and the hit@k). The reference is the
    full-text cosine (the Stage-0 baseline arm) on the same held-out
    documents.

    Args:
        view: The model-independent dataset view.
        model_name: The short handle of the model (the artifact's row key).
        extractor: The feature pack (it embeds the pair's own chunks through
            the ``Embedder`` port).
        combiner_factory: A callable that builds a fresh combiner (one per
            fit: the rotations and the permutations must not share a model).
        reference_similarities: The full-text cosine matrix (the baseline).
        control_weight: The training weight of the fit side's control cells.
        permutation_resamples: The number of label permutations of the floor.
        seed: The seed of the permutations.
        bootstrap_resamples: The number of bootstrap resamples of the paired
            statistics.
        bootstrap_seed: The seed of the bootstrap.
        hit_ks: The hit@k grid of the rank report.
        feature_embeddings: The provenance of the feature pack's **embedding
            sides** — the run's model (the Stage-3.1 behaviour) or the
            random-vector floor (CS-6d). It is recorded in the artifact, so a
            reader knows which space the arm's embedding features live in.
        calibration_map_factory: A callable that builds the **cosine's
            calibration map** (a logistic regression on the cosine as its
            single feature, fitted on the fit side only). It is used by the
            Stage-3 reporting (CS-6e).
        reporting: The Stage-3 reporting's knobs (CS-6e). ``None`` (the
            default) computes no reporting block, so every earlier run's
            artifact keeps its exact content.

    Returns:
        The model's Stage-3 block (the rotations, the out-of-fold summary, the
        comparison and the permutation floor).
    """
    grid = build_stage3_grid(view)
    all_features = extractor.extract_all(grid.pairs)
    if all_features.shape[0] != grid.n_cells:
        raise ValueError(
            f"the feature pack returned {all_features.shape[0]} rows for {grid.n_cells} cells"
        )
    # the combiner sees the selected groups' columns; the diagnostics see
    # every column the pack computes
    features = all_features[:, np.asarray(extractor.selected_indices, dtype=np.int64)]

    pair_rows, pair_columns, labels = _pair_indices(view)
    control_rows, control_columns, control_documents = _control_indices(view)
    pair_documents = np.asarray([record.document for record in view.pairs])
    pair_cells = np.asarray(
        [
            grid.index[(int(row), int(column))]
            for row, column in zip(pair_rows, pair_columns, strict=True)
        ],
        dtype=np.int64,
    )
    control_cells = np.asarray(
        [
            grid.index[(int(row), int(column))]
            for row, column in zip(control_rows, control_columns, strict=True)
        ],
        dtype=np.int64,
    )

    documents = [stratum["document"] for stratum in view.strata]
    cells_by_document = _cells_by_document(grid)
    context = Stage3Context(
        view=view,
        documents=tuple(documents),
        features=features,
        all_features=all_features,
        grid=grid,
        cells_by_document=cells_by_document,
        cell_positions=_cell_positions(cells_by_document, grid.n_cells),
        pair_cells=pair_cells,
        control_cells=control_cells,
        labels=labels,
        pair_documents=pair_documents,
        control_documents=control_documents,
        combiner_factory=combiner_factory,
        calibration_map_factory=calibration_map_factory or combiner_factory,
        control_weight=control_weight,
        reference_similarities=reference_similarities,
        cosine_index=(
            list(extractor.all_feature_names).index(COSINE_FEATURE_NAME)
            if COSINE_FEATURE_NAME in extractor.all_feature_names
            else 0
        ),
    )
    rng = np.random.default_rng(seed)
    rotations: list[dict[str, Any]] = []
    out_of_fold = np.full((len(view.paragraphs), len(view.questions)), np.nan, dtype=np.float64)
    for holdout in documents:
        rotation = _stage3_rotation(
            context,
            holdout=holdout,
            permutation_resamples=permutation_resamples,
            rng=rng,
            feature_names=extractor.feature_names,
            reporting=reporting,
        )
        rotations.append(rotation)
        # The holdout's grid cells take the scores of the model that never
        # saw that document.
        holdout_grid = rotation.pop("holdout_grid_positions")
        holdout_scores = rotation.pop("holdout_scores")
        out_of_fold[grid.rows[holdout_grid], grid.columns[holdout_grid]] = holdout_scores

    comparison = compare_stage3_arm(
        view,
        arm_name=STAGE3_ARM,
        reference_similarities=reference_similarities,
        arm_similarities=out_of_fold,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        hit_ks=hit_ks,
    )
    chance_values = [
        value
        for rotation in rotations
        for value in rotation["permutation_floor"]["chance_auc"]
        if value is not None
    ]
    control_role_values = [
        value
        for rotation in rotations
        for value in rotation["permutation_floor"]["control_role_auc"]
        if value is not None
    ]
    # The headline is the **within-document** AUC: the pair-count-weighted mean
    # of the documents' own AUCs (the same weighting the metric uses). The
    # floors are measured one document at a time, so they get the same
    # weighting here — otherwise a per-document mean would be compared with a
    # pair-count-weighted one.
    pair_counts = {
        stratum["document"]: stratum["n_positive"] * stratum["n_negative"]
        for stratum in view.strata
    }
    chance_within = _weighted_mean(
        [
            (rotation["holdout"], _mean_of(rotation["permutation_floor"]["chance_auc"]))
            for rotation in rotations
        ],
        pair_counts,
    )
    control_role_within = _weighted_mean(
        [
            (rotation["holdout"], _mean_of(rotation["permutation_floor"]["control_role_auc"]))
            for rotation in rotations
        ],
        pair_counts,
    )
    # The control cells' paragraph lengths (the length check's raw material).
    control_lengths = np.asarray(
        [view.paragraph_char_lengths[int(row)] for row in control_rows], dtype=np.float64
    )
    arm_measured = measure_arm(view, out_of_fold)
    reference_measured = measure_arm(view, reference_similarities)
    diagnostics = _feature_diagnostics(
        view,
        features=all_features,
        feature_names=extractor.all_feature_names,
        grid=grid,
        labels=labels,
        pair_documents=pair_documents,
        control_documents=control_documents,
        control_lengths=control_lengths,
    )
    length_check = _length_check(
        arm=arm_measured,
        reference=reference_measured,
        control_lengths=control_lengths,
    )
    # The Stage-3 reporting (CS-6e): the pooled numbers beside the
    # per-document ones. It is computed only when the reporting is enabled,
    # so every earlier run's artifact keeps its exact content.
    reporting_block: dict[str, Any] = {}
    if reporting is not None and reporting.enabled:
        reporting_block = _assemble_reporting(
            rotations,
            grid=grid,
            pair_cells=pair_cells,
            labels=labels,
            out_of_fold=out_of_fold,
            reporting=reporting,
        )
    stage3_block: dict[str, Any] = {
        "model": model_name,
        "feature_pack": extractor.describe(),
        "feature_embeddings": dict(feature_embeddings or {}),
        "combiner": rotations[0]["combiner"] if rotations else None,
        "out_of_fold_scores": {
            "judged_pairs": [
                float(value)
                for value in out_of_fold[grid.rows[pair_cells], grid.columns[pair_cells]]
            ],
            "control_cells": [
                float(value)
                for value in out_of_fold[grid.rows[control_cells], grid.columns[control_cells]]
            ],
            "note": (
                "the arm's out-of-fold score of every judged pair and every "
                "control cell, in the order of the model's 'judged_pairs' and "
                "'control_cells' blocks. They are the per-pair scores two arms "
                "of different runs can be compared on (a paired per-pair "
                "bootstrap), which the per-rotation AUCs alone cannot give."
            ),
        },
        "training": {
            "control_weight": float(control_weight),
            "fit_side": (
                "the fit side's judged pairs (the verified positives and the "
                "reviewer-verified hard negatives) plus its unjudged control "
                "cells as down-weighted presumed negatives"
            ),
            "leakage_rules": [
                "the fit side never contains the holdout document",
                "no corpus-level question statistic is used (the features are pair-local)",
                "no document-identity feature and no length feature",
                "the scaler of the combiner is fitted on the fit side's rows only",
            ],
        },
        "grid": {
            "n_cells": grid.n_cells,
            "n_matrix_positions": int(out_of_fold.size),
            "n_scored_cells": int(np.count_nonzero(~np.isnan(out_of_fold))),
            "n_unscored_positions": int(np.count_nonzero(np.isnan(out_of_fold))),
            "note": (
                "the out-of-fold matrix holds the same-document grid only: the "
                "cross-document positions are not scored (no rotation holds out "
                "two documents at once), and every statistic the comparison uses "
                "lives inside a document"
            ),
        },
        "rotations": rotations,
        "out_of_fold": {
            "pair_auc": comparison["pair_auc"],
            "auc_decomposition": comparison["auc_decomposition"],
            "control_side": comparison["control_side"],
            "floor_gap": comparison["floor_gap"],
            "deshifted_auc": comparison["deshifted_auc"],
            "hit_at_k": comparison["hit_at_k"],
        },
        "comparison": comparison,
        "feature_diagnostics": diagnostics,
        "length_check": length_check,
        "permutation_floor": {
            "n_resamples": int(permutation_resamples),
            "n_measured": len(chance_values),
            "chance": {
                "mean": float(np.mean(chance_values)) if chance_values else None,
                "min": float(np.min(chance_values)) if chance_values else None,
                "max": float(np.max(chance_values)) if chance_values else None,
            },
            "control_role": {
                "mean": float(np.mean(control_role_values)) if control_role_values else None,
                "min": float(np.min(control_role_values)) if control_role_values else None,
                "max": float(np.max(control_role_values)) if control_role_values else None,
            },
            "within_document": {
                "chance": chance_within,
                "control_role": control_role_within,
                "note": (
                    "the pair-count-weighted mean of the documents' own AUCs — "
                    "the same weighting the within-document AUC uses, so these "
                    "two numbers are comparable with the arm's within-document "
                    "AUC and the baseline's"
                ),
            },
            "note": (
                "the label-permutation floors, each a per-document AUC (one per "
                "rotation), so their mean is comparable with the arm's per-"
                "document macro AUC and not with its pooled AUC (the out-of-fold "
                "matrix mixes three models' score levels). A holdout with a "
                "single class has no AUC and is left out (n_measured)."
            ),
        },
        "rotations_summary": [
            {
                "holdout": rotation["holdout"],
                "arm_pair_auc": rotation["holdout_pair_auc"],
                "reference_pair_auc": rotation["holdout_reference_pair_auc"],
                "arm_minus_reference": rotation["holdout_pair_auc_delta"],
                "chance_floor_mean": _mean_of(rotation["permutation_floor"]["chance_auc"]),
                "control_role_floor_mean": _mean_of(
                    rotation["permutation_floor"]["control_role_auc"]
                ),
                "arm_minus_chance_floor": _difference(
                    rotation["holdout_pair_auc"],
                    _mean_of(rotation["permutation_floor"]["chance_auc"]),
                ),
                "arm_minus_control_role_floor": _difference(
                    rotation["holdout_pair_auc"],
                    _mean_of(rotation["permutation_floor"]["control_role_auc"]),
                ),
                "n_holdout_questions": rotation["question_overlap"]["n_holdout_questions"],
                "n_holdout_questions_also_judged_in_fit": rotation["question_overlap"][
                    "n_holdout_questions_also_judged_in_fit"
                ],
            }
            for rotation in rotations
        ],
    }
    # The reporting's block is only there when the reporting is enabled, so
    # every earlier run's artifact keeps its exact content.
    if reporting_block:
        stage3_block["reporting"] = reporting_block
    return stage3_block
