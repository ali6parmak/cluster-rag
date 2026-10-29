"""Stage 3: the learned feature pack and its reporting.

The pack is a table of numbers per pair plus a learner; this module builds
the grid, fits the pack under the document-level holdout, and reports the
calibration, the thresholds and the safe-negative margin (CS-6a … CS-6e).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.eval.metrics import paired_auc_bootstrap, paired_within_group_auc_bootstrap
from cluster_rag.omct_analysis.arms import (
    ArmScores,
    _hit_at_k,
    measure_arm,
)
from cluster_rag.omct_analysis.helpers import (
    BASELINE_ARM,
    COSINE_FEATURE_NAME,
    HIT_KS,
    RULE_VARIANT_ARM,
    RULE_VARIANT_BOTH,
    RULE_VARIANT_COSINE,
    RULE_VARIANTS,
    _auc_decomposition,
    _distribution,
    _floor_gap,
    _macro,
    _quantile,
    _safe_auc,
    _top_quintile_share,
    length_correlation,
)
from cluster_rag.omct_analysis.view import (
    DatasetView,
    ReportingSpec,
)
from cluster_rag.text_processors.base import Combiner, TextPair
from cluster_rag.text_processors.calibration import (
    THRESHOLD_CRITERION_DESCRIPTIONS,
    THRESHOLD_CRITERION_FLOOR_GAP,
    choose_margin,
    choose_threshold,
    harm_at_coverage,
    margin_candidates,
    margin_curve,
    operating_point,
    reliability_diagram,
    rule_counts,
    rule_curve,
    threshold_candidates,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage3Grid:
    """The same-document grid the learned scorer scores.

    The Stage-3 arm is a **per-pair** model, so it scores the whole grid: the
    rank of a pair inside its paragraph's candidate list needs every
    candidate's score, and the candidate list is the paragraph's document's
    questions — the same scope the Stage-2 arms use. The cells are ordered
    (document, row, column), so the artifact's numbers are reproducible.
    """

    rows: np.ndarray
    columns: np.ndarray
    documents: np.ndarray
    pairs: tuple[TextPair, ...]
    index: dict[tuple[int, int], int]

    @property
    def n_cells(self) -> int:
        """The number of grid cells."""
        return int(self.rows.size)


def build_stage3_grid(view: DatasetView) -> Stage3Grid:
    """The grid's cells (documents in the strata's order, rows and columns ascending)."""
    rows: list[int] = []
    columns: list[int] = []
    documents: list[str] = []
    pairs: list[TextPair] = []
    index: dict[tuple[int, int], int] = {}
    for grid in view.grids:
        for row in grid.paragraph_rows:
            for column in grid.question_cols:
                index[(row, column)] = len(pairs)
                rows.append(row)
                columns.append(column)
                documents.append(grid.document)
                pairs.append(
                    TextPair(
                        paragraph_text=view.paragraphs[row],
                        question_text=view.questions[column],
                    )
                )
    return Stage3Grid(
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        documents=np.asarray(documents),
        pairs=tuple(pairs),
        index=index,
    )


def _stage3_control_block(
    control_scores: np.ndarray,
    control_percentiles: np.ndarray,
    control_documents: np.ndarray,
) -> dict[str, Any]:
    """The control-side counterpart of one arm (its level and its rank position)."""
    by_document: dict[str, Any] = {}
    for document in sorted(set(control_documents.tolist())):
        mask = control_documents == document
        scores = control_scores[mask]
        percentiles = control_percentiles[mask]
        by_document[document] = {
            "n": int(mask.sum()),
            "mean": float(scores.mean()) if scores.size else None,
            "p50": _quantile(scores, 0.5),
            "rank_p50": _quantile(percentiles, 0.5),
            "top_fifth_share": _top_quintile_share(percentiles),
        }
    return {
        "n": int(control_scores.size),
        "mean": float(control_scores.mean()) if control_scores.size else None,
        "p50": _quantile(control_scores, 0.5),
        "rank_p50": _quantile(control_percentiles, 0.5),
        "top_fifth_share": _top_quintile_share(control_percentiles),
        "by_document": by_document,
    }


def _deshifted_auc(
    view: DatasetView,
    labels: np.ndarray,
    pair_scores: np.ndarray,
    pair_documents: np.ndarray,
    control_scores: np.ndarray,
    control_documents: np.ndarray,
) -> dict[str, Any]:
    """The AUC after each document's control level is removed.

    The de-shift is computed **inside one arm's own scale**: the control's
    per-document mean minus the control's pooled mean is subtracted from that
    document's reviewed-set scores. A uniform additive shift cannot change an
    AUC, so this removes the *document-level* part of the score level and
    leaves the ranking inside a document (the track's transferable statistic).
    """
    pooled = float(control_scores.mean()) if control_scores.size else 0.0
    shifts: dict[str, float] = {}
    deshifted = np.empty_like(pair_scores, dtype=np.float64)
    for stratum in view.strata:
        document = stratum["document"]
        control_mask = control_documents == document
        shift = float(control_scores[control_mask].mean()) - pooled if control_mask.any() else 0.0
        shifts[document] = shift
        pair_mask = pair_documents == document
        deshifted[pair_mask] = pair_scores[pair_mask] - shift
    return {
        "overall": _safe_auc(labels, deshifted),
        "macro": _macro(
            {
                stratum["document"]: _safe_auc(
                    labels[pair_documents == stratum["document"]],
                    deshifted[pair_documents == stratum["document"]],
                )
                for stratum in view.strata
            }
        ),
        "by_document": {
            stratum["document"]: _safe_auc(
                labels[pair_documents == stratum["document"]],
                deshifted[pair_documents == stratum["document"]],
            )
            for stratum in view.strata
        },
        "per_document_shift": shifts,
    }


def _floor_gap_block(
    positive_scores: np.ndarray, control_scores: np.ndarray
) -> dict[str, float | None]:
    """The floor gap (the positives' mean minus the control's mean) and its standardized form."""
    gap = _floor_gap(positive_scores, control_scores)
    spread = float(control_scores.std()) if control_scores.size else 0.0
    return {
        "gap": gap,
        "standardized": (gap / spread if gap is not None and spread > 0 else None),
    }


def compare_stage3_arm(
    view: DatasetView,
    *,
    arm_name: str,
    reference_similarities: np.ndarray,
    arm_similarities: np.ndarray,
    resamples: int,
    seed: int,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """Compare the learned arm with the full-text baseline arm (the same blocks).

    The blocks are the Stage-2 ones (the pair AUC pooled / per document /
    macro, the within/cross decomposition, the control-side counterpart, the
    de-shifted AUC, the floor gap and the hit@k), with one difference forced
    by the arms' scales: the learned arm's scores are **probabilities** and
    the baseline's are **cosines**, so the level numbers (the control's mean,
    the floor gap) are reported in each arm's own units and are *not*
    comparable. The scale-free numbers (the AUCs, the rank positions, the
    standardized floor gap) are comparable, and they carry the verdict.
    """
    reference = measure_arm(view, reference_similarities)
    arm = measure_arm(view, arm_similarities)
    labels = arm.labels

    paired = paired_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        n_resamples=resamples,
        seed=seed,
    )
    reference_decomposition = _auc_decomposition(labels, reference.pair_scores, reference.documents)
    arm_decomposition = _auc_decomposition(labels, arm.pair_scores, arm.documents)
    within_reference = reference_decomposition.get("within_group_auc")
    within_arm = arm_decomposition.get("within_group_auc")
    cross_reference = reference_decomposition.get("cross_group_auc")
    cross_arm = arm_decomposition.get("cross_group_auc")
    within_paired = paired_within_group_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        arm.documents.tolist(),
        n_resamples=resamples,
        seed=seed,
    )

    reference_auc_by_document = {
        stratum["document"]: _safe_auc(
            labels[reference.documents == stratum["document"]],
            reference.pair_scores[reference.documents == stratum["document"]],
        )
        for stratum in view.strata
    }
    arm_auc_by_document = {
        stratum["document"]: _safe_auc(
            labels[arm.documents == stratum["document"]],
            arm.pair_scores[arm.documents == stratum["document"]],
        )
        for stratum in view.strata
    }

    control_reference = _stage3_control_block(
        reference.control_scores, reference.control_percentiles, reference.control_documents
    )
    control_arm = _stage3_control_block(
        arm.control_scores, arm.control_percentiles, arm.control_documents
    )
    floor_reference = _floor_gap_block(reference.pair_scores[labels == 1], reference.control_scores)
    floor_arm = _floor_gap_block(arm.pair_scores[labels == 1], arm.control_scores)

    return {
        "arm": arm_name,
        "reference_arm": BASELINE_ARM,
        "paired_auc": paired,
        "pair_auc": {
            "reference": paired["auc_reference"],
            "arm": paired["auc_arm"],
            "delta": paired["delta_auc"],
            "macro": {
                "reference": _macro(reference_auc_by_document),
                "arm": _macro(arm_auc_by_document),
                "delta": (
                    _macro(arm_auc_by_document) - _macro(reference_auc_by_document)
                    if _macro(arm_auc_by_document) is not None
                    and _macro(reference_auc_by_document) is not None
                    else None
                ),
            },
            "by_document": {
                document: {
                    "reference": reference_auc_by_document[document],
                    "arm": arm_auc_by_document[document],
                    "delta": (
                        arm_auc_by_document[document] - reference_auc_by_document[document]
                        if arm_auc_by_document[document] is not None
                        and reference_auc_by_document[document] is not None
                        else None
                    ),
                }
                for document in reference_auc_by_document
            },
        },
        "auc_decomposition": {
            "reference": reference_decomposition,
            "arm": arm_decomposition,
            "within_group_delta": (
                within_arm - within_reference
                if within_arm is not None and within_reference is not None
                else None
            ),
            "cross_group_delta": (
                cross_arm - cross_reference
                if cross_arm is not None and cross_reference is not None
                else None
            ),
            "within_group_paired": within_paired,
            "note": (
                "the pooled AUC mixes the ranking inside a document with the "
                "ranking across documents; the within-document delta is the "
                "part that transfers to a corpus without document structure"
            ),
        },
        "control_side": {
            "reference": control_reference,
            "arm": control_arm,
            "rank_p50_delta": (
                control_arm["rank_p50"] - control_reference["rank_p50"]
                if control_arm["rank_p50"] is not None and control_reference["rank_p50"] is not None
                else None
            ),
            "top_fifth_share_delta": (
                control_arm["top_fifth_share"] - control_reference["top_fifth_share"]
                if control_arm["top_fifth_share"] is not None
                and control_reference["top_fifth_share"] is not None
                else None
            ),
            "note": (
                "the control-side counterpart of the discrimination numbers. The "
                "level numbers (mean, p50) are in each arm's own units — the "
                "learned arm scores probabilities, the baseline scores cosines — "
                "so they are not comparable; the rank numbers (rank_p50, the top-"
                "fifth share) are scale-free and are."
            ),
        },
        "negative_enrichment": {
            "rank_percentile": {
                "label1": _distribution(arm.pair_percentiles[labels == 1]),
                "label0": _distribution(arm.pair_percentiles[labels == 0]),
                "control": _distribution(arm.control_percentiles),
            },
            "top_fifth_share": {
                "label1": _top_quintile_share(arm.pair_percentiles[labels == 1]),
                "label0": _top_quintile_share(arm.pair_percentiles[labels == 0]),
                "control": _top_quintile_share(arm.control_percentiles),
            },
        },
        "floor_gap": {
            "reference": floor_reference,
            "arm": floor_arm,
            "standardized_delta": (
                floor_arm["standardized"] - floor_reference["standardized"]
                if floor_arm["standardized"] is not None
                and floor_reference["standardized"] is not None
                else None
            ),
            "note": (
                "the gap between the verified positives' mean score and the "
                "control's mean score, in each arm's own units; the standardized "
                "form divides it by the control's spread and is comparable"
            ),
        },
        "deshifted_auc": {
            "reference": _deshifted_auc(
                view,
                labels,
                reference.pair_scores,
                reference.documents,
                reference.control_scores,
                reference.control_documents,
            ),
            "arm": _deshifted_auc(
                view,
                labels,
                arm.pair_scores,
                arm.documents,
                arm.control_scores,
                arm.control_documents,
            ),
            "note": (
                "each arm's own per-document control level removed (the control's "
                "per-document mean minus its pooled mean): the ranking inside a "
                "document survives the document's score level"
            ),
        },
        "hit_at_k": {
            "reference": {
                "label1": _hit_at_k(reference.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(reference.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(reference.control_ranks, hit_ks),
            },
            "arm": {
                "label1": _hit_at_k(arm.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(arm.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(arm.control_ranks, hit_ks),
            },
        },
    }


@dataclass(frozen=True)
class Stage3Context:
    """The arrays the Stage-3 rotations share (the out-of-fold harness's state).

    The rotations need the same feature matrix, the same grid and the same
    index arrays; bundling them keeps the rotation's signature readable and
    makes the **reporting**'s helpers (CS-6e) use exactly the same arrays as
    the arm itself.
    """

    view: DatasetView
    documents: tuple[str, ...]
    features: np.ndarray
    all_features: np.ndarray
    grid: Stage3Grid
    cells_by_document: dict[str, np.ndarray]
    cell_positions: np.ndarray
    pair_cells: np.ndarray
    control_cells: np.ndarray
    labels: np.ndarray
    pair_documents: np.ndarray
    control_documents: np.ndarray
    combiner_factory: Callable[[], Combiner]
    calibration_map_factory: Callable[[], Combiner]
    control_weight: float
    reference_similarities: np.ndarray
    cosine_index: int


def _cells_by_document(grid: Stage3Grid) -> dict[str, np.ndarray]:
    """The grid's cell indexes per document (a document's cells, in grid order)."""
    return {
        document: np.flatnonzero(grid.documents == document)
        for document in sorted(set(grid.documents.tolist()))
    }


def _cell_positions(cells_by_document: Mapping[str, np.ndarray], n_cells: int) -> np.ndarray:
    """The position of every grid cell inside its own document's cell list."""
    positions = np.zeros(n_cells, dtype=np.int64)
    for cells in cells_by_document.values():
        positions[cells] = np.arange(cells.size)
    return positions


def _margins_of_document(rows: np.ndarray, columns: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The margin of every cell of one document: its distance below its row's best.

    The margin is measured **inside the paragraph's own candidate list** (the
    document's questions), so a document's score level never enters the rule:
    a margin rule, never an absolute cut (the track's rule, which the
    pairwise file's hard negatives demand — they sit above the verified
    positives' 95th percentile, so no absolute cut separates them).

    Args:
        rows: The paragraph row of every cell.
        columns: The question column of every cell.
        scores: The score of every cell.

    Returns:
        The margin of every cell (0 for the paragraph's best cell).
    """
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    boundaries = np.flatnonzero(np.diff(sorted_rows)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [values.size]])
    maxima = np.maximum.reduceat(values[order], starts)
    margins = np.empty_like(values)
    margins[order] = np.repeat(maxima, ends - starts) - values
    return margins


def _training_rows(
    documents: Sequence[str],
    *,
    pair_cells: np.ndarray,
    control_cells: np.ndarray,
    labels: np.ndarray,
    pair_documents: np.ndarray,
    control_documents: np.ndarray,
    control_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The training rows of a set of documents: their judged pairs + their control cells.

    The control cells are **presumed** negatives (their true positive rate is
    unmeasured), so they are down-weighted: they must not dominate the fit
    next to the reviewer-verified labels.
    """
    pair_mask = np.isin(pair_documents, list(documents))
    control_mask = np.isin(control_documents, list(documents))
    rows = np.concatenate([pair_cells[pair_mask], control_cells[control_mask]])
    targets = np.concatenate(
        [labels[pair_mask], np.zeros(int(control_mask.sum()), dtype=np.int64)], axis=0
    )
    weights = np.concatenate(
        [
            np.ones(int(pair_mask.sum()), dtype=np.float64),
            np.full(int(control_mask.sum()), float(control_weight), dtype=np.float64),
        ],
        axis=0,
    )
    return rows, targets, weights


def _cross_fit_scores(
    context: Stage3Context,
    *,
    train_documents: Sequence[str],
    score_documents: Sequence[str],
    label_vector: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Score every scored document with a model that never saw that document.

    This is the **inner** document-level cross-fit: for every scored document
    the model is fitted on the training documents *minus* the scored one, so
    the fit side's own rows are scored out-of-fold. The fit side's choices
    (the per-document threshold, the margin rule's margin) are therefore made
    on scores that no model produced with the holdout in its training set —
    the plan's leakage rule, one level in.

    Args:
        context: The rotations' shared state.
        train_documents: The documents the models are fitted on.
        score_documents: The documents to score.
        label_vector: The labels of the judged pairs (the permuted ones for a
            floor arm); the default is the run's labels.

    Returns:
        The scores of every scored document's cells (in the grid's order).
    """
    labels = context.labels if label_vector is None else label_vector
    scores: dict[str, np.ndarray] = {}
    for document in score_documents:
        training = [name for name in train_documents if name != document]
        rows, targets, weights = _training_rows(
            training,
            pair_cells=context.pair_cells,
            control_cells=context.control_cells,
            labels=labels,
            pair_documents=context.pair_documents,
            control_documents=context.control_documents,
            control_weight=context.control_weight,
        )
        combiner = context.combiner_factory().fit(context.features[rows], targets, weights)
        scores[document] = combiner.score(context.features[context.cells_by_document[document]])
    return scores


def _safe_margin_rule(
    margins: np.ndarray,
    *,
    unjudged: np.ndarray,
    positives: np.ndarray,
    n_quantiles: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """The margin rule's fit-side choice, or the reason there is none.

    A degenerate fit side (no unjudged cell to reach for, no verified
    positive to protect) leaves the rule unmeasurable; the run records the
    reason instead of failing.
    """
    values = margins[unjudged]
    if not values.size:
        return None, ("the fit side has no unjudged cell: the rule's reach is unmeasurable")
    try:
        candidates = margin_candidates(values, n_quantiles=n_quantiles)
        choice = choose_margin(
            margins, unjudged=unjudged, positives=positives, candidates=candidates
        )
    except ValueError as error:
        return None, str(error)
    return choice, None


def _stage3_reporting(
    context: Stage3Context,
    *,
    holdout: str,
    holdout_scores: np.ndarray,
    inner_scores: Mapping[str, np.ndarray],
    floor_draws: Sequence[Mapping[str, Any]],
    reporting: ReportingSpec,
) -> dict[str, Any]:
    """The Stage-3 reporting of one rotation (CS-6e).

    Three blocks, all measured with the plan's discipline:

    - the **calibration**: the arm's own probability and the full-text cosine
      put on a probability scale by a **fit-side** calibration map (a logistic
      regression on the cosine as its single feature), each with a reliability
      diagram (the predicted probability against the observed positive rate,
      with the bin counts), the expected calibration error and the Brier
      score;
    - the **threshold**: the per-document operating point, chosen on the
      **fit side** (the other two documents) and applied to the held-out one.
      The fit side's own rows are scored by the **inner** cross-fit, so the
      choice never rests on a model that saw the holdout;
    - the **safe-negative margin**: the rule that declares a cell a safe
      negative when it sits at least the chosen margin below its own
      paragraph's best candidate. Its numbers are the rule's **coverage** over
      the whole unjudged grid and its **harm** on the verified positives, next
      to the same numbers for the **control-role floor** (the model that only
      knows the training design's prior).

    The two negative subsets are never pooled: the control cells (subset B,
    presumed) and the reviewer-verified hard negatives (subset A, ground
    truth) are separate masks everywhere.
    """
    grid = context.grid
    labels = context.labels
    pair_cells = context.pair_cells
    control_cells = context.control_cells
    pair_documents = context.pair_documents
    control_documents = context.control_documents
    cells_by_document = context.cells_by_document
    cell_positions = context.cell_positions
    fit_side_documents = [document for document in context.documents if document != holdout]

    # the cell masks over the whole grid
    judged_mask = np.zeros(grid.n_cells, dtype=bool)
    judged_mask[pair_cells] = True
    positive_mask = np.zeros(grid.n_cells, dtype=bool)
    positive_mask[pair_cells[labels == 1]] = True
    unjudged_mask = ~judged_mask

    holdout_cells = cells_by_document[holdout]
    holdout_rows = grid.rows[holdout_cells]
    holdout_columns = grid.columns[holdout_cells]
    holdout_unjudged = unjudged_mask[holdout_cells]
    holdout_positive = positive_mask[holdout_cells]

    def _arm_scores(document: str) -> np.ndarray:
        """The arm's scores of a document's cells (the holdout: the outer model)."""
        return holdout_scores if document == holdout else inner_scores[document]

    def _arm_margins(document: str) -> np.ndarray:
        """The arm's margins of a document's cells."""
        cells = cells_by_document[document]
        return _margins_of_document(grid.rows[cells], grid.columns[cells], _arm_scores(document))

    def _cosine_margins(document: str) -> np.ndarray:
        """The cosine's margins of a document's cells (the geometry)."""
        cells = cells_by_document[document]
        rows = grid.rows[cells]
        columns = grid.columns[cells]
        return _margins_of_document(rows, columns, context.reference_similarities[rows, columns])

    # ------------------------------------------------------------------
    # the calibration
    # ------------------------------------------------------------------
    holdout_pair_mask = pair_documents == holdout
    holdout_pair_cells = pair_cells[holdout_pair_mask]
    holdout_pair_labels = labels[holdout_pair_mask]
    arm_probabilities = holdout_scores[cell_positions[holdout_pair_cells]]

    # The cosine's calibration map: a logistic regression on the cosine as its
    # single feature, fitted on the fit side's rows only (Platt scaling).
    fit_pair_mask = pair_documents != holdout
    fit_control_mask = control_documents != holdout
    cosine_feature = context.all_features[:, context.cosine_index].reshape(-1, 1)
    map_rows = np.concatenate([pair_cells[fit_pair_mask], control_cells[fit_control_mask]])
    map_labels = np.concatenate(
        [labels[fit_pair_mask], np.zeros(int(fit_control_mask.sum()), dtype=np.int64)], axis=0
    )
    map_weights = np.concatenate(
        [
            np.ones(int(fit_pair_mask.sum()), dtype=np.float64),
            np.full(int(fit_control_mask.sum()), float(context.control_weight), dtype=np.float64),
        ],
        axis=0,
    )
    calibration_map = context.calibration_map_factory().fit(
        cosine_feature[map_rows], map_labels, map_weights
    )
    cosine_probabilities = calibration_map.score(cosine_feature[holdout_cells])[
        cell_positions[holdout_pair_cells]
    ]

    calibration = {
        "arm": reliability_diagram(arm_probabilities, holdout_pair_labels, n_bins=reporting.bins),
        "cosine": {
            **reliability_diagram(cosine_probabilities, holdout_pair_labels, n_bins=reporting.bins),
            "map": {
                "name": "logistic",
                "feature": COSINE_FEATURE_NAME,
                "fitted_on": (
                    "the fit side's judged pairs and unjudged control cells (the "
                    "cosine as the single feature, the control cells down-weighted)"
                ),
                "note": (
                    "Platt scaling: the cosine put on a probability scale by a "
                    "map fitted on the fit side only, so the map never sees the "
                    "holdout"
                ),
            },
        },
        "note": (
            "the reliability diagram of the holdout's judged pairs: the arm's own "
            "probability and the cosine's fit-side calibration map. One "
            "calibration per document, so the per-document numbers are "
            "like-for-like; the pooled numbers mix the three documents' "
            "calibrations and are flagged as such."
        ),
    }

    # ------------------------------------------------------------------
    # the per-document threshold
    # ------------------------------------------------------------------
    fit_pair_labels = labels[fit_pair_mask]
    fit_pair_scores = np.concatenate(
        [
            inner_scores[document][
                cell_positions[pair_cells[fit_pair_mask][pair_documents[fit_pair_mask] == document]]
            ]
            for document in fit_side_documents
        ]
    )
    fit_control_scores = np.concatenate(
        [
            inner_scores[document][
                cell_positions[
                    control_cells[fit_control_mask][control_documents[fit_control_mask] == document]
                ]
            ]
            for document in fit_side_documents
        ]
    )
    combined = np.concatenate([fit_pair_scores, fit_control_scores])
    positives_mask = np.concatenate(
        [fit_pair_labels == 1, np.zeros(fit_control_scores.size, dtype=bool)], axis=0
    )
    if reporting.threshold_criterion == THRESHOLD_CRITERION_FLOOR_GAP:
        negatives_mask = np.concatenate(
            [
                np.zeros(fit_pair_scores.size, dtype=bool),
                np.ones(fit_control_scores.size, dtype=bool),
            ],
            axis=0,
        )
        negative_class = "the unjudged control cells (subset B, presumed negatives)"
    else:
        negatives_mask = np.concatenate(
            [fit_pair_labels == 0, np.zeros(fit_control_scores.size, dtype=bool)], axis=0
        )
        negative_class = "the reviewer-verified hard negatives (subset A, ground truth)"
    threshold_grid = threshold_candidates(combined, n_quantiles=reporting.threshold_quantiles)
    threshold_error: str | None = None
    try:
        threshold_choice = choose_threshold(
            combined, positives=positives_mask, negatives=negatives_mask, candidates=threshold_grid
        )
    except ValueError as error:
        # A degenerate fit side (no positive or no negative row) leaves the
        # threshold unmeasurable; the run records the reason instead of failing.
        threshold_choice = None
        threshold_error = str(error)
    threshold = float(threshold_choice["threshold"]) if threshold_choice is not None else None
    holdout_control_cells = control_cells[control_documents == holdout]
    holdout_control_scores = holdout_scores[cell_positions[holdout_control_cells]]
    threshold_block = {
        "criterion": reporting.threshold_criterion,
        "criterion_description": THRESHOLD_CRITERION_DESCRIPTIONS.get(
            reporting.threshold_criterion, ""
        ),
        "negative_class": negative_class,
        "fit_side": {
            "threshold": threshold,
            "criterion_value": threshold_choice["value"] if threshold_choice else None,
            "tpr": threshold_choice["tpr"] if threshold_choice else None,
            "fpr": threshold_choice["fpr"] if threshold_choice else None,
            "n_candidates": len(threshold_grid),
            "n_positive": int(positives_mask.sum()),
            "n_negative": int(negatives_mask.sum()),
            "error": threshold_error,
            "note": (
                "the threshold is chosen on the fit side (the other two "
                "documents) and applied to the held-out one. The fit side's rows "
                "are scored by the inner cross-fit, so no model that saw the "
                "holdout produced them."
            ),
            "curve": threshold_choice["curve"] if threshold_choice else [],
        },
        "holdout": {
            "threshold": threshold,
            "operating_point": (
                operating_point(arm_probabilities, holdout_pair_labels, threshold)
                if threshold is not None
                else None
            ),
            "control_side": {
                "n": int(holdout_control_scores.size),
                "share_above_threshold": (
                    float((holdout_control_scores >= threshold).mean())
                    if threshold is not None and holdout_control_scores.size
                    else None
                ),
                "note": (
                    "the holdout's unjudged control cells (subset B) at the "
                    "threshold: they are presumed negatives, never ground truth"
                ),
            },
            "unjudged_grid": (
                rule_counts(
                    holdout_scores < threshold,
                    unjudged=holdout_unjudged,
                    positives=holdout_positive,
                )
                if threshold is not None
                else None
            ),
        },
    }

    # ------------------------------------------------------------------
    # the safe-negative margin rule
    # ------------------------------------------------------------------
    fit_arm_margins = np.concatenate([_arm_margins(document) for document in fit_side_documents])
    fit_cosine_margins = np.concatenate(
        [_cosine_margins(document) for document in fit_side_documents]
    )
    fit_unjudged = np.concatenate(
        [unjudged_mask[cells_by_document[document]] for document in fit_side_documents]
    )
    fit_positive = np.concatenate(
        [positive_mask[cells_by_document[document]] for document in fit_side_documents]
    )
    arm_grid = (
        margin_candidates(fit_arm_margins[fit_unjudged], n_quantiles=reporting.margin_quantiles)
        if fit_unjudged.any()
        else np.asarray([], dtype=np.float64)
    )
    cosine_grid = (
        margin_candidates(fit_cosine_margins[fit_unjudged], n_quantiles=reporting.margin_quantiles)
        if fit_unjudged.any()
        else np.asarray([], dtype=np.float64)
    )
    arm_choice, arm_error = _safe_margin_rule(
        fit_arm_margins,
        unjudged=fit_unjudged,
        positives=fit_positive,
        n_quantiles=reporting.margin_quantiles,
    )
    cosine_choice, cosine_error = _safe_margin_rule(
        fit_cosine_margins,
        unjudged=fit_unjudged,
        positives=fit_positive,
        n_quantiles=reporting.margin_quantiles,
    )
    arm_margin = arm_choice["margin"] if arm_choice else None
    cosine_margin_value = cosine_choice["margin"] if cosine_choice else None

    holdout_arm_margins = _arm_margins(holdout)
    holdout_cosine_margins = _cosine_margins(holdout)
    arm_cleared = (
        holdout_arm_margins >= arm_margin
        if arm_margin is not None
        else np.zeros_like(holdout_arm_margins, dtype=bool)
    )
    cosine_cleared = (
        holdout_cosine_margins >= cosine_margin_value
        if cosine_margin_value is not None
        else np.zeros_like(holdout_cosine_margins, dtype=bool)
    )
    both_cleared = arm_cleared & cosine_cleared
    control_positions = cell_positions[holdout_control_cells]
    arm_curve = margin_curve(
        holdout_arm_margins,
        unjudged=holdout_unjudged,
        positives=holdout_positive,
        candidates=arm_grid,
    )
    cosine_curve = margin_curve(
        holdout_cosine_margins,
        unjudged=holdout_unjudged,
        positives=holdout_positive,
        candidates=cosine_grid,
    )
    both_curve = rule_curve(
        [cosine_cleared & (holdout_arm_margins >= float(candidate)) for candidate in arm_grid],
        unjudged=holdout_unjudged,
        positives=holdout_positive,
        candidates=arm_grid,
    )

    # The control-role floor's own rule: the same procedure on the model whose
    # judged labels were permuted (the control cells keep their role).
    floor_runs: list[dict[str, Any]] = []
    for draw in floor_draws:
        draw_fit_margins = np.concatenate(
            [
                _margins_of_document(
                    grid.rows[cells_by_document[document]],
                    grid.columns[cells_by_document[document]],
                    draw["inner"][document],
                )
                for document in fit_side_documents
            ]
        )
        draw_holdout_margins = _margins_of_document(holdout_rows, holdout_columns, draw["outer"])
        draw_grid = (
            margin_candidates(
                draw_fit_margins[fit_unjudged], n_quantiles=reporting.margin_quantiles
            )
            if fit_unjudged.any()
            else np.asarray([], dtype=np.float64)
        )
        draw_choice, draw_error = _safe_margin_rule(
            draw_fit_margins,
            unjudged=fit_unjudged,
            positives=fit_positive,
            n_quantiles=reporting.margin_quantiles,
        )
        if draw_choice is None:
            continue
        floor_runs.append(
            {
                "choice": draw_choice,
                "margins": draw_holdout_margins,
                "cleared": draw_holdout_margins >= draw_choice["margin"],
                "curve": margin_curve(
                    draw_holdout_margins,
                    unjudged=holdout_unjudged,
                    positives=holdout_positive,
                    candidates=draw_grid,
                ),
                # the two-part rule's curve on the floor: the same cosine's
                # margin AND the floor's own arm margin, so the floor's harm is
                # read off the floor's *own* curve of the same rule
                "both_curve": rule_curve(
                    [
                        cosine_cleared & (draw_holdout_margins >= float(candidate))
                        for candidate in draw_grid
                    ],
                    unjudged=holdout_unjudged,
                    positives=holdout_positive,
                    candidates=draw_grid,
                ),
            }
        )

    def _floor_block(
        real_coverage: float | None, *, cosine_part: np.ndarray | None = None
    ) -> dict[str, Any]:
        """The control-role floor's numbers for one rule variant."""
        if not floor_runs:
            return {
                "n_draws": 0,
                "note": (
                    "the run drew no control-role floor (the permutation "
                    "resamples are zero), so the rule has no floor counterpart"
                ),
            }
        per_draw: list[dict[str, Any]] = []
        for run in floor_runs:
            cleared = run["cleared"]
            curve = run["curve"]
            if cosine_part is not None:
                cleared = cleared & cosine_part
                curve = run["both_curve"]
            counts = rule_counts(cleared, unjudged=holdout_unjudged, positives=holdout_positive)
            matched = harm_at_coverage(curve, real_coverage) if real_coverage is not None else None
            per_draw.append(
                {
                    "margin": run["choice"]["margin"],
                    "coverage": counts["coverage"],
                    "harm": counts["harm"],
                    "harm_at_matched_coverage": matched,
                }
            )
        coverages = [entry["coverage"] for entry in per_draw if entry["coverage"] is not None]
        harms = [entry["harm"] for entry in per_draw if entry["harm"] is not None]
        matched_values = [
            entry["harm_at_matched_coverage"]
            for entry in per_draw
            if entry["harm_at_matched_coverage"] is not None
        ]
        return {
            "n_draws": len(per_draw),
            "per_draw": per_draw,
            "mean": {
                "coverage": float(np.mean(coverages)) if coverages else None,
                "harm": float(np.mean(harms)) if harms else None,
            },
            "harm_at_matched_coverage": {
                "coverage": real_coverage,
                "harm": float(np.mean(matched_values)) if matched_values else None,
                "n_draws_compared": len(matched_values),
            },
            "note": (
                "the control-role floor: the model fitted with the judged pairs' "
                "labels permuted (the control cells keep their presumed-negative "
                "role). The rule runs on it exactly as on the real arm — its own "
                "fit-side margin choice — so the two numbers say how much of the "
                "rule's reach is the training design's own prior. The floor is "
                "one draw per permutation, and the mean over the draws is "
                "reported."
            ),
        }

    def _variant(
        *,
        space: str,
        fit_choice: Mapping[str, Any] | None,
        error: str | None,
        cleared: np.ndarray,
        curve: list[dict[str, Any]],
        floor: Mapping[str, Any],
        note: str | None = None,
    ) -> dict[str, Any]:
        """One rule variant's block: the fit side's choice and the holdout's numbers."""
        counts = rule_counts(cleared, unjudged=holdout_unjudged, positives=holdout_positive)
        control_cleared = cleared[control_positions]
        return {
            "space": space,
            "fit_side": {
                "margin": fit_choice["margin"] if fit_choice else None,
                "criterion_value": fit_choice["value"] if fit_choice else None,
                "coverage": fit_choice["coverage"] if fit_choice else None,
                "harm": fit_choice["harm"] if fit_choice else None,
                "n_candidates": fit_choice["n_candidates"] if fit_choice else 0,
                "error": error,
                "note": note
                or (
                    "the margin is chosen on the fit side (the other two "
                    "documents) by the rule's own criterion — the coverage over "
                    "the fit side's unjudged cells minus the harm on the fit "
                    "side's verified positives — and applied to the held-out one"
                ),
            },
            "holdout": {
                "counts": counts,
                "control_sample": {
                    "n": int(control_positions.size),
                    "n_cleared": int(control_cleared.sum()),
                    "coverage": (float(control_cleared.mean()) if control_positions.size else None),
                    "note": (
                        "the holdout's unjudged control cells (subset B) the rule "
                        "clears: they are presumed negatives, never ground truth"
                    ),
                },
                "curve": curve,
            },
            "floor_arm": dict(floor),
        }

    arm_counts = rule_counts(arm_cleared, unjudged=holdout_unjudged, positives=holdout_positive)
    both_counts = rule_counts(both_cleared, unjudged=holdout_unjudged, positives=holdout_positive)

    variants = {
        RULE_VARIANT_COSINE: _variant(
            space="the full-text cosine's similarity (the geometry)",
            fit_choice=cosine_choice,
            error=cosine_error,
            cleared=cosine_cleared,
            curve=cosine_curve,
            floor={
                "n_draws": 0,
                "note": (
                    "the cosine's margin is not a fitted model, so this variant "
                    "has no control-role floor: the rule is pure geometry and "
                    "inherits nothing from the training design"
                ),
            },
        ),
        RULE_VARIANT_ARM: _variant(
            space="the learned arm's probability (the ranking)",
            fit_choice=arm_choice,
            error=arm_error,
            cleared=arm_cleared,
            curve=arm_curve,
            floor=_floor_block(arm_counts["coverage"]),
        ),
        RULE_VARIANT_BOTH: _variant(
            space=(
                "both: the cosine's margin (the geometry) and the arm's margin "
                "(the ranking) — CS-6d's 'retrieve with the cosine, rank with "
                "the learned arm'"
            ),
            fit_choice=(
                {
                    "margin": arm_choice["margin"],
                    "value": arm_choice["value"],
                    "coverage": both_counts["coverage"],
                    "harm": both_counts["harm"],
                    "n_candidates": arm_choice["n_candidates"],
                    "cosine_margin": cosine_choice["margin"],
                }
                if arm_choice and cosine_choice
                else None
            ),
            error=arm_error or cosine_error,
            cleared=both_cleared,
            curve=both_curve,
            floor=_floor_block(both_counts["coverage"], cosine_part=cosine_cleared),
            note=(
                "each margin is chosen on the fit side by its own criterion; the "
                "two are not jointly optimized, so the intersection cannot "
                "over-fit the fit side"
            ),
        ),
    }

    return {
        "calibration": calibration,
        "thresholds": threshold_block,
        "pool_inputs": {
            "arm": [float(value) for value in arm_probabilities],
            "cosine": [float(value) for value in cosine_probabilities],
            "labels": [int(value) for value in holdout_pair_labels],
        },
        "safe_negative": {
            "definition": {
                "margin": (
                    "the cell's distance below its own paragraph's best "
                    "candidate, inside the paragraph's document"
                ),
                "rule": (
                    "a cell is a safe negative when its margin is at least the "
                    "chosen margin: a margin rule, never an absolute cut, so a "
                    "document's score level never enters it"
                ),
                "criterion": (
                    "the fit side's coverage over the unjudged cells minus the "
                    "harm on the verified positives"
                ),
            },
            "variants": variants,
            "unmeasured": (
                "the unjudged cells' true positive rate is UNMEASURED: no human "
                "labelling in this track. The rule ships as the margin half "
                "only, and its quality claim is about its reach over the "
                "unjudged grid and its harm on the verified positives — read "
                "against the control-role floor — never about its precision on "
                "the unjudged cells."
            ),
        },
    }


def _stage3_rotation(
    context: Stage3Context,
    *,
    holdout: str,
    permutation_resamples: int,
    rng: np.random.Generator,
    feature_names: Sequence[str] = (),
    reporting: ReportingSpec | None = None,
) -> dict[str, Any]:
    """One document-level holdout: fit on two documents, predict the third.

    The fit side's rows are its **judged** pairs (the verified positives and
    the reviewer-verified hard negatives) plus its **unjudged control cells**
    as presumed negatives, down-weighted by ``control_weight`` (the control's
    true positive rate is unmeasured, so it must not dominate the fit). The
    holdout's rows are never seen by the fit.
    """
    view = context.view
    documents = context.documents
    features = context.features
    grid = context.grid
    pair_cells = context.pair_cells
    control_cells = context.control_cells
    labels = context.labels
    pair_documents = context.pair_documents
    control_documents = context.control_documents
    combiner_factory = context.combiner_factory
    control_weight = context.control_weight
    reference_similarities = context.reference_similarities
    fit_side_documents = [document for document in documents if document != holdout]
    fit_pairs = pair_documents != holdout
    holdout_pairs = pair_documents == holdout
    fit_control = control_documents != holdout
    holdout_control = control_documents == holdout
    holdout_grid = grid.documents == holdout

    fit_features = np.concatenate(
        [features[pair_cells[fit_pairs]], features[control_cells[fit_control]]], axis=0
    )
    fit_labels = np.concatenate(
        [labels[fit_pairs], np.zeros(int(fit_control.sum()), dtype=np.int64)], axis=0
    )
    fit_weights = np.concatenate(
        [
            np.ones(int(fit_pairs.sum()), dtype=np.float64),
            np.full(int(fit_control.sum()), float(control_weight), dtype=np.float64),
        ],
        axis=0,
    )

    combiner = combiner_factory().fit(fit_features, fit_labels, fit_weights)
    holdout_scores = combiner.score(features[holdout_grid])

    # The label-permutation floors: the same fit with the labels destroyed.
    #
    # - **chance floor** (the plan's): the fit side's **whole label vector**
    #   is permuted (the judged pairs' labels and the control cells' labels
    #   together, so the class counts and the weights stay exactly as they
    #   are and only the assignment changes). The model then has no label
    #   information at all, so the AUC must land at chance (0.5).
    # - **control-role floor** (the diagnostic): only the judged pairs'
    #   labels are permuted and the control cells keep their presumed-negative
    #   role. The gap to the chance floor is what the *training design* alone
    #   (the "the ordinary unjudged cells are negative" prior) buys on the
    #   reviewed set — a label-free signal the arm inherits.
    chance_auc: list[float | None] = []
    control_role_auc: list[float | None] = []
    control_labels = np.zeros(int(fit_control.sum()), dtype=np.int64)
    # The control-role floor's **grid** scores (CS-6e): the rule's numbers on
    # the model that only knows the training design's prior. The draw is kept
    # out of the artifact's arrays and reduced to counts.
    floor_draws: list[dict[str, Any]] = []
    reporting_enabled = bool(reporting and reporting.enabled)
    for _ in range(permutation_resamples):
        shuffled = labels[fit_pairs].copy()
        rng.shuffle(shuffled)
        permuted_labels = np.concatenate([shuffled, control_labels], axis=0)
        # the control-role floor: the control cells keep their presumed-negative role
        permuted = combiner_factory().fit(fit_features, permuted_labels, fit_weights)
        permuted_scores = permuted.score(features[pair_cells[holdout_pairs]])
        control_role_auc.append(_safe_auc(labels[holdout_pairs], permuted_scores))
        if reporting_enabled:
            permuted_full = labels.copy()
            permuted_full[fit_pairs] = shuffled
            floor_draws.append(
                {
                    "outer": permuted.score(features[holdout_grid]),
                    "inner": _cross_fit_scores(
                        context,
                        train_documents=fit_side_documents,
                        score_documents=fit_side_documents,
                        label_vector=permuted_full,
                    ),
                }
            )
        # the chance floor: the whole label vector is permuted
        permuted_pool = permuted_labels.copy()
        rng.shuffle(permuted_pool)
        chance = combiner_factory().fit(fit_features, permuted_pool, fit_weights)
        chance_scores = chance.score(features[pair_cells[holdout_pairs]])
        chance_auc.append(_safe_auc(labels[holdout_pairs], chance_scores))

    reference = measure_arm(view, reference_similarities)
    holdout_reference_scores = reference.pair_scores[holdout_pairs]
    holdout_pair_auc = _safe_auc(
        labels[holdout_pairs], combiner.score(features[pair_cells[holdout_pairs]])
    )
    holdout_reference_auc = _safe_auc(labels[holdout_pairs], holdout_reference_scores)
    # The question overlap between the two sides: the plan's leakage rule (i).
    # 186/294 question numbers are judged in more than one document, so a
    # *question-level* feature (the question-type block) can carry the fit
    # side's label for a question the holdout also carries. The overlap is
    # measured here, so the question-type arm's number can be read against it.
    fit_questions = set(grid.columns[pair_cells[fit_pairs]].tolist())
    holdout_columns = grid.columns[pair_cells[holdout_pairs]]
    holdout_questions = set(holdout_columns.tolist())
    shared_questions = fit_questions & holdout_questions
    # The Stage-3 reporting (CS-6e): the calibration, the per-document
    # threshold and the safe-negative margin, all measured on this rotation.
    reporting_block: dict[str, Any] = {}
    if reporting_enabled:
        inner_scores = _cross_fit_scores(
            context,
            train_documents=fit_side_documents,
            score_documents=fit_side_documents,
        )
        reporting_block = _stage3_reporting(
            context,
            holdout=holdout,
            holdout_scores=holdout_scores,
            inner_scores=inner_scores,
            floor_draws=floor_draws,
            reporting=reporting,
        )
    rotation_report = {
        "holdout": holdout,
        "fit_documents": [document for document in documents if document != holdout],
        "n_fit_pairs": int(fit_pairs.sum()),
        "n_fit_positive": int((labels[fit_pairs] == 1).sum()),
        "n_fit_negative": int((labels[fit_pairs] == 0).sum()),
        "n_fit_control": int(fit_control.sum()),
        "control_weight": float(control_weight),
        "fit_weight_mass": {
            "judged": float(fit_weights[: int(fit_pairs.sum())].sum()),
            "control": float(fit_weights[int(fit_pairs.sum()) :].sum()),
        },
        "n_holdout_pairs": int(holdout_pairs.sum()),
        "n_holdout_positive": int((labels[holdout_pairs] == 1).sum()),
        "n_holdout_negative": int((labels[holdout_pairs] == 0).sum()),
        "n_holdout_control": int(holdout_control.sum()),
        "n_holdout_grid_cells": int(holdout_grid.sum()),
        "question_overlap": {
            "n_holdout_questions": len(holdout_questions),
            "n_holdout_questions_also_judged_in_fit": len(shared_questions),
            "n_holdout_pairs_with_a_shared_question": int(
                np.isin(holdout_columns, list(shared_questions)).sum()
            ),
            "note": (
                "how many of the holdout's questions the fit side also judged. "
                "The features are pair-local (no question-level statistic "
                "enters them), but a question-type weight fitted on the fit "
                "side reaches the holdout's copy of the same question, so a "
                "question-type-only arm must be read against this overlap."
            ),
        },
        "holdout_pair_auc": holdout_pair_auc,
        "holdout_reference_pair_auc": holdout_reference_auc,
        "holdout_pair_auc_delta": (
            holdout_pair_auc - holdout_reference_auc
            if holdout_pair_auc is not None and holdout_reference_auc is not None
            else None
        ),
        "holdout_grid_positions": holdout_grid,
        "holdout_scores": holdout_scores,
        "permutation_floor": {
            "n_resamples": int(permutation_resamples),
            "chance_auc": chance_auc,
            "control_role_auc": control_role_auc,
            "note": (
                "the label-permutation floors. The chance floor permutes the fit "
                "side's whole label vector (the class counts and the weights stay "
                "as they are, only the assignment changes), so the model has no "
                "label information and the AUC must land at chance (0.5). The "
                "control-role floor permutes only the judged pairs' labels and "
                "keeps the control cells' presumed-negative role: the gap to the "
                "chance floor is what the training design alone buys on the "
                "reviewed set."
            ),
        },
        "coefficients": _named_coefficients(combiner, feature_names),
        "feature_usage": _named_feature_usage(combiner, feature_names),
        "combiner": combiner.describe(),
    }
    # The reporting's block is only there when the reporting is enabled, so
    # every earlier run's artifact keeps its exact content.
    if reporting_enabled:
        rotation_report["reporting"] = reporting_block
    return rotation_report


def _feature_diagnostics(
    view: DatasetView,
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    grid: Stage3Grid,
    labels: np.ndarray,
    pair_documents: np.ndarray,
    control_documents: np.ndarray,
    control_lengths: np.ndarray,
) -> dict[str, Any]:
    """Every feature's own numbers: its discrimination and its control side.

    The Stage-3.3 change set asks for **every new feature's control-side
    counterpart next to its discrimination number**, so each feature gets:

    - the **discrimination** number: the within-document AUC of the feature
      *used as a score* (the same statistic the headline is measured with),
      the pooled AUC and the per-document macro beside it;
    - the **control-side counterpart**: the ordinary (unjudged) cells' mean
      and median and their rank percentile inside the paragraph's candidate
      list — a feature that lifts the ordinary cells shows up here next to
      one that separates the reviewed set;
    - the **length correlation**: how much the feature's value travels with
      the paragraph's length. A feature that is a length proxy would
      memorize the document instead of the topic (the plan's leakage rule
      (ii)), so the number is measured, not assumed.

    The features are measured on the **whole grid** (the same-document
    cells), so the reviewed pairs and the control cells are scored on one
    scale.
    """
    per_feature: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        values = np.asarray(features[:, index], dtype=np.float64)
        matrix = np.full((len(view.paragraphs), len(view.questions)), np.nan, dtype=np.float64)
        matrix[grid.rows, grid.columns] = values
        measured = measure_arm(view, matrix)
        decomposition = _auc_decomposition(labels, measured.pair_scores, measured.documents)
        by_document = {
            stratum["document"]: _safe_auc(
                labels[pair_documents == stratum["document"]],
                measured.pair_scores[pair_documents == stratum["document"]],
            )
            for stratum in view.strata
        }
        per_feature[name] = {
            "within_auc": decomposition.get("within_group_auc"),
            "pooled_auc": _safe_auc(labels, measured.pair_scores),
            "macro_auc": _macro(by_document),
            "by_document": by_document,
            "control": {
                "mean": float(measured.control_scores.mean())
                if measured.control_scores.size
                else None,
                "p50": _quantile(measured.control_scores, 0.5),
                "rank_p50": _quantile(measured.control_percentiles, 0.5),
                "top_fifth_share": _top_quintile_share(measured.control_percentiles),
            },
            "length_correlation": length_correlation(
                control_lengths, measured.control_scores, control_documents
            ),
        }
    return {
        "per_feature": per_feature,
        "note": (
            "every feature the pack computes, measured on its own: the "
            "within-document AUC (the headline's statistic, the feature used "
            "as a score), the pooled AUC and the per-document macro beside "
            "it; the control-side counterpart (the ordinary unjudged cells' "
            "level and rank position); and the correlation between the "
            "feature and the paragraph's length — the length-proxy check, "
            "since a length feature would memorize the document instead of "
            "the topic."
        ),
    }


def _length_check(
    *,
    arm: ArmScores,
    reference: ArmScores,
    control_lengths: np.ndarray,
) -> dict[str, Any]:
    """The length check: does the arm's score travel with the paragraph's length?

    The Stage-0 caveat says a method's scale shift is partly a **length**
    effect and partly a **document** effect. The baseline arm's correlation
    is the reference (it is already recorded per document); the arm's own
    correlation sits beside it, and the **rank shift** (the arm's control
    rank percentile minus the baseline's) is correlated with the length as
    well — the rank shift is scale-free, so it compares the two arms without
    their different units (a probability against a cosine).
    """
    rank_shift = arm.control_percentiles - reference.control_percentiles
    return {
        "arm": length_correlation(control_lengths, arm.control_scores, arm.control_documents),
        "reference": length_correlation(
            control_lengths, reference.control_scores, reference.control_documents
        ),
        "rank_shift": length_correlation(control_lengths, rank_shift, arm.control_documents),
        "note": (
            "the correlation between a control cell's paragraph length and its "
            "score, for the learned arm and for the full-text cosine (the "
            "reference). The rank-shift block is the same correlation for the "
            "difference of the two arms' rank percentiles — it is scale-free, "
            "so it compares the arms without their different units."
        ),
    }


def _named_coefficients(combiner: Combiner, feature_names: Sequence[str]) -> dict[str, Any]:
    """The fitted coefficients, keyed by the feature name.

    The combiner standardizes its features, so the coefficients are
    comparable with each other: their magnitudes say which features the fit
    side leaned on. A coefficient on a **question-type** feature is a
    question-level prior by construction (the feature does not depend on the
    paragraph), so it is worth seeing.
    """
    getter = getattr(combiner, "coefficients", None)
    if getter is None:
        return {}
    values = getter()
    named: dict[str, float] = {}
    for index, value in values.items():
        position = int(index)
        name = feature_names[position] if position < len(feature_names) else f"feature_{position}"
        named[name] = float(value)
    return {
        "values": named,
        "note": (
            "the fit side's standardized coefficients: they say which features "
            "the fit leaned on. A question-type feature does not depend on the "
            "paragraph, so its coefficient is a question-level prior."
        ),
    }


def _named_feature_usage(combiner: Combiner, feature_names: Sequence[str]) -> dict[str, Any]:
    """Which features the fitted trees split on, keyed by the feature name.

    A tree ensemble has no coefficients: the record of its splits is what
    says which features the model used (the Stage-3.2 importances
    capability). The combiner that has no such record (the logistic
    regression) returns an empty block.

    The block is a **usage** diagnostic, not an accuracy-based importance: it
    says what the trees split on, not what would improve the score if a
    feature were removed. It is computed on the fit side's rows only.
    """
    getter = getattr(combiner, "feature_usage", None)
    if getter is None:
        return {}
    values = getter()
    named: dict[str, dict[str, float]] = {}
    for index, usage in values.items():
        position = int(index)
        name = feature_names[position] if position < len(feature_names) else f"feature_{position}"
        named[name] = {key: float(value) for key, value in usage.items()}
    return {
        "per_feature": named,
        "note": (
            "the fit side's trees' split usage: how often each feature was "
            "split on, the gain those splits bought and the shares of all the "
            "splits and all the gain. It is a usage diagnostic (what the "
            "model splits on), not an accuracy-based importance (what would "
            "improve the score if the feature were removed)."
        ),
    }


def _assemble_reporting(
    rotations: Sequence[dict[str, Any]],
    *,
    grid: Stage3Grid,
    pair_cells: np.ndarray,
    labels: np.ndarray,
    out_of_fold: np.ndarray,
    reporting: ReportingSpec,
) -> dict[str, Any]:
    """The model's Stage-3 reporting: the pooled numbers beside the per-document ones.

    The **pooled** numbers mix the three documents' calibrations (each
    document's cells are scored by a different model), so they are flagged;
    the per-document numbers are the like-for-like ones. Every pooled number
    sits beside its per-document macro, as the plan's reporting discipline
    asks.
    """
    per_document: dict[str, dict[str, Any]] = {
        rotation["holdout"]: rotation["reporting"] for rotation in rotations
    }
    pool_inputs = [rotation["reporting"].pop("pool_inputs") for rotation in rotations]
    arm_probabilities = np.concatenate(
        [np.asarray(entry["arm"], dtype=np.float64) for entry in pool_inputs]
    )
    cosine_probabilities = np.concatenate(
        [np.asarray(entry["cosine"], dtype=np.float64) for entry in pool_inputs]
    )
    pooled_labels = np.concatenate(
        [np.asarray(entry["labels"], dtype=np.int64) for entry in pool_inputs]
    )
    out_of_fold_labels = labels

    def _calibration_block(key: str, pooled_values: np.ndarray) -> dict[str, Any]:
        """The calibration block of one probability source (the arm, the cosine)."""
        by_document = {
            document: per_document[document]["calibration"][key] for document in per_document
        }
        macro_ece = _macro({document: by_document[document].get("ece") for document in by_document})
        block = {
            "pooled": reliability_diagram(pooled_values, pooled_labels, n_bins=reporting.bins),
            "by_document": by_document,
            "macro_ece": macro_ece,
            "note": (
                "the reliability diagram of the predicted probability against the "
                "observed positive rate, with the bin counts. The per-document "
                "diagrams are like-for-like (one calibration per document, one "
                "model); the pooled diagram mixes the three documents' "
                "calibrations, so it is a summary and not a like-for-like "
                "comparison. The macro is the mean of the per-document expected "
                "calibration errors."
            ),
        }
        # the cosine's block carries the provenance of its calibration map
        map_block = next(iter(per_document.values()))["calibration"][key].get("map")
        if map_block is not None:
            block["map"] = map_block
        return block

    # the per-document thresholds and the out-of-fold (pooled) operating point
    out_of_fold_scores = out_of_fold[grid.rows[pair_cells], grid.columns[pair_cells]]
    thresholds_by_document: dict[str, Any] = {}
    for document, block in per_document.items():
        threshold_block = block["thresholds"]
        thresholds_by_document[document] = {
            "fit_side": {
                key: value for key, value in threshold_block["fit_side"].items() if key != "curve"
            },
            "holdout": threshold_block["holdout"],
        }
    pooled_operating = {
        document: operating_point(
            out_of_fold_scores, out_of_fold_labels, float(block["fit_side"]["threshold"])
        )
        for document, block in thresholds_by_document.items()
        if block["fit_side"].get("threshold") is not None
    }
    chosen_thresholds = [
        float(block["fit_side"]["threshold"])
        for block in thresholds_by_document.values()
        if block["fit_side"].get("threshold") is not None
    ]
    median_threshold = float(np.median(chosen_thresholds)) if chosen_thresholds else None
    thresholds = {
        "criterion": reporting.threshold_criterion,
        "criterion_description": THRESHOLD_CRITERION_DESCRIPTIONS.get(
            reporting.threshold_criterion, ""
        ),
        "negative_class": next(iter(per_document.values()))["thresholds"]["negative_class"],
        "per_document": thresholds_by_document,
        "out_of_fold": {
            "operating_point_per_document_threshold": pooled_operating,
            "median_threshold": median_threshold,
            "operating_point_at_median": (
                operating_point(out_of_fold_scores, out_of_fold_labels, median_threshold)
                if median_threshold is not None
                else None
            ),
            "note": (
                "the pooled operating point of the out-of-fold matrix — the "
                "whole grid's scores, one cut. The out-of-fold matrix mixes the "
                "three documents' calibrations (each document's cells come from "
                "a different model), so a single global cut is NOT like-for-like: "
                "the per-document thresholds are. The three cuts' pooled "
                "operating points are reported side by side so the cost of one "
                "global cut is visible."
            ),
        },
    }

    # the safe-negative rule: the per-document blocks and the pooled counts
    variants: dict[str, Any] = {}
    for variant in RULE_VARIANTS:
        per_document_counts = {
            document: block["safe_negative"]["variants"][variant]
            for document, block in per_document.items()
        }
        totals = {
            "n_unjudged_cells": 0,
            "n_unjudged_cleared": 0,
            "n_positives": 0,
            "n_positives_discarded": 0,
        }
        for block in per_document_counts.values():
            counts = block["holdout"]["counts"]
            for key in totals:
                totals[key] += int(counts[key])
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
        floor_blocks = [block["floor_arm"] for block in per_document_counts.values()]
        floor_draws = [entry for block in floor_blocks for entry in block.get("per_draw", [])]
        floor_coverages = [
            entry["coverage"] for entry in floor_draws if entry.get("coverage") is not None
        ]
        floor_harms = [entry["harm"] for entry in floor_draws if entry.get("harm") is not None]
        # The floor's harm at the real arm's reach, **per document**: each
        # document's own reach, the floor's own curve of the same rule, then the
        # mean over the documents. The count of documents where the real arm
        # discards fewer verified positives is the verdict's number.
        matched_per_document: list[float] = []
        n_documents_better = 0
        for block in per_document_counts.values():
            matched = (block["floor_arm"].get("harm_at_matched_coverage") or {}).get("harm")
            if matched is None:
                continue
            matched_per_document.append(float(matched))
            document_harm = (block["holdout"]["counts"] or {}).get("harm")
            if document_harm is not None and float(document_harm) < float(matched):
                n_documents_better += 1
        matched = [
            entry["harm_at_matched_coverage"]
            for entry in floor_draws
            if entry.get("harm_at_matched_coverage") is not None
        ]
        variants[variant] = {
            "space": next(iter(per_document_counts.values()))["space"],
            "pooled": {
                **totals,
                "coverage": coverage,
                "harm": harm,
                "note": (
                    "the rule's reach over the WHOLE unjudged grid (every "
                    "same-document cell without a record), pooled over the three "
                    "documents, and the share of the verified positives the rule "
                    "would discard"
                ),
            },
            "per_document": {
                document: {
                    "fit_side": block["fit_side"],
                    "holdout": {
                        "counts": block["holdout"]["counts"],
                        "control_sample": block["holdout"]["control_sample"],
                    },
                    "floor_arm": block["floor_arm"],
                }
                for document, block in per_document_counts.items()
            },
            "floor_arm": {
                "n_draws": len(floor_draws),
                "mean": {
                    "coverage": (float(np.mean(floor_coverages)) if floor_coverages else None),
                    "harm": float(np.mean(floor_harms)) if floor_harms else None,
                },
                "harm_at_matched_coverage": {
                    "coverage": coverage,
                    "harm": (
                        float(np.mean(matched_per_document))
                        if matched_per_document
                        else (float(np.mean(matched)) if matched else None)
                    ),
                    "n_documents_compared": len(matched_per_document),
                    "n_documents_better": n_documents_better,
                    "n_draws_compared": len(matched),
                    "note": (
                        "the floor's harm at the real arm's reach, per document: "
                        "each document's own reach, read off the floor's own curve "
                        "of the same rule, then averaged over the three "
                        "documents. The count of documents where the real arm "
                        "discards fewer verified positives than the floor at the "
                        "same reach is the verdict's number."
                    ),
                },
                "note": (
                    "the control-role floor's rule: the model fitted with the "
                    "judged pairs' labels permuted (the control cells keep their "
                    "presumed-negative role), running the same rule-selection "
                    "procedure on the fit side. The matched-coverage number is "
                    "the floor's harm at the real arm's reach — the rule's own "
                    "numbers against the floor, in the rule's own units."
                ),
            },
        }

    return {
        "calibration": {
            "arm": _calibration_block("arm", arm_probabilities),
            "cosine": _calibration_block("cosine", cosine_probabilities),
            "note": (
                "two probability sources: the learned arm's own probability and "
                "the full-text cosine put on a probability scale by a fit-side "
                "calibration map (a logistic regression on the cosine as its "
                "single feature)."
            ),
        },
        "thresholds": thresholds,
        "safe_negative": {
            "definition": next(iter(per_document.values()))["safe_negative"]["definition"],
            "variants": variants,
            "unmeasured": next(iter(per_document.values()))["safe_negative"]["unmeasured"],
        },
    }
