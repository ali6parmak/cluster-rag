"""The dictionary arm (the track's CS-49): the question map as a pair scorer.

The map's measurement side, in the same shape as the track's other arms
(``omct_analysis/arms.py`` and ``decision.py``): the arm is scored on the same
1,520 pairs and the same 4,000 control cells as every run, and compared with
the run's own full-text cosine — paired, with the control side beside the
discrimination number.

Two things are special to this arm and stated in the artifact:

- it reads **no embedding** (the map is the corpus's own statistics), so its
  "sides" are placeholders and the arm is model-independent;
- its judged cells are scored **out-of-fold** (the map of the fold that did not
  contain the pair), while the control cells are scored by the **full**
  reviewed-set map (they never entered any fit) — the vintage difference is
  recorded with its per-fold spread.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.omct_analysis.arms import compare_arms, measure_arm, score_arm
from cluster_rag.omct_analysis.helpers import _auc_decomposition, _macro, _safe_auc
from cluster_rag.omct_analysis.view import DatasetView, SideEncoding
from cluster_rag.text_processors.lexical.question_map_score import PRIMARY_SCORING

logger = logging.getLogger(__name__)

# The arm's name in the artifact and the table.
DICTIONARY_ARM = "dictionary"

# The registered bar of the label-blind test (CS-49's condition 2): the arm's
# within-document AUC minus the blind map's must be at least this.
BLIND_GAP_BAR = 0.05


def _dictionary_side(texts: Sequence[str], note: str) -> SideEncoding:
    """A placeholder side for the arm's metadata block (no embedding exists)."""
    return SideEncoding(
        texts=tuple(texts),
        chunk_lists=tuple((index,) for index in range(len(texts))),
        vectors=np.zeros((len(texts), 0), dtype=np.float32),
        meta={"unit": "the dictionary's own words (no embedding)", "note": note},
    )


def _arm_stats(view: DatasetView, measured: Any) -> dict[str, Any]:
    """The AUC block of one arm's measured scores (within, pooled, per document)."""
    labels = measured.labels
    decomposition = _auc_decomposition(labels, measured.pair_scores, measured.documents)
    by_document: dict[str, float | None] = {}
    for stratum in view.strata:
        document = stratum["document"]
        mask = measured.documents == document
        by_document[document] = _safe_auc(labels[mask], measured.pair_scores[mask])
    return {
        "within_auc": decomposition.get("within_group_auc"),
        "cross_auc": decomposition.get("cross_group_auc"),
        "pooled_auc": _safe_auc(labels, measured.pair_scores),
        "macro_auc": _macro(by_document),
        "by_document": by_document,
        "n_pairs": int(labels.size),
        "n_within_group_pairs": decomposition.get("n_within_group_pairs"),
        "within_group_pair_share": decomposition.get("within_group_pair_share"),
    }


def _non_floor_stats(view: DatasetView, measured: Any, keep: np.ndarray) -> dict[str, Any]:
    """One arm's AUC over the pairs whose fold map has a table (the floor excluded)."""
    block: dict[str, Any] = {
        "n_pairs": int(keep.sum()),
        "within_auc": None,
        "pooled_auc": None,
        "by_document": {},
        "note": (
            "the arm's reading over the pairs that have a table in their fold's map "
            "(the floor carriers excluded); the headline number is over all pairs"
        ),
    }
    if keep.any():
        block["within_auc"] = _auc_decomposition(
            measured.labels[keep], measured.pair_scores[keep], measured.documents[keep]
        ).get("within_group_auc")
        block["pooled_auc"] = _safe_auc(measured.labels[keep], measured.pair_scores[keep])
        for stratum in view.strata:
            document = stratum["document"]
            mask = keep & (measured.documents == document)
            block["by_document"][document] = (
                _safe_auc(measured.labels[mask], measured.pair_scores[mask]) if mask.any() else None
            )
    return block


def measure_dictionary_arms(
    view: DatasetView,
    grids: Mapping[str, Any],
    blind_grids: Mapping[str, Any] | None,
    *,
    primary: str = PRIMARY_SCORING,
) -> dict[str, Any]:
    """The four arms' AUCs, the floor's cost and the label-blind counterpart.

    Args:
        view: The dataset view (the pairs, the control, the documents).
        grids: The real cross-fit grids (the ``cross_fit_grids`` output).
        blind_grids: The label-blind counterpart's grids (the same output with
            the questions permuted); ``None`` skips the counterpart (the block
            then says the blind arm was not computed).
        primary: The pre-registered primary arm.

    Returns:
        The readout block: the per-scoring AUCs, the floor share, the blind
        counterpart, the blind gap (pooled and per document) and the secondary
        reading over the pairs that have a table.
    """
    floor = np.asarray(grids["floor"], dtype=bool)
    keep = ~floor
    real: dict[str, Any] = {}
    non_floor: dict[str, Any] = {}
    blind: dict[str, Any] | None = {} if blind_grids is not None else None
    for scoring, matrix in grids["scorings"].items():
        stats = _arm_stats(view, measure_arm(view, matrix))
        stats["n_floor_pairs"] = int(floor.sum())
        stats["floor_share"] = float(floor.mean()) if floor.size else None
        real[scoring] = stats
        non_floor[scoring] = _non_floor_stats(view, measure_arm(view, matrix), keep)
        if blind is not None and blind_grids is not None:
            blind[scoring] = _arm_stats(view, measure_arm(view, blind_grids["scorings"][scoring]))

    primary_real = real[primary]
    blind_gap: dict[str, Any] | None = None
    if blind is not None:
        primary_blind = blind[primary]
        blind_gap = {
            "primary": (
                primary_real["within_auc"] - primary_blind["within_auc"]
                if primary_real["within_auc"] is not None
                and primary_blind["within_auc"] is not None
                else None
            ),
            "bar": BLIND_GAP_BAR,
            "by_document": {},
            "note": (
                "the arm's within-document AUC minus the blind map's: the part of "
                "the arm that reads label information rather than the questionnaire's "
                "shape (the blind arm permutes the questions across the pairs, the "
                "counts preserved)"
            ),
        }
        for document in primary_real["by_document"]:
            left = primary_real["by_document"].get(document)
            right = primary_blind["by_document"].get(document)
            blind_gap["by_document"][document] = (
                left - right if left is not None and right is not None else None
            )

    # The secondary reading, **per arm**: the pairs whose fold map has a table
    # for their question (the map can speak about them at all). The primary
    # number itself is over all the reviewed pairs; this one says what the arm
    # does when the map is not silent.
    blind_gap_by_arm: dict[str, float | None] = {}
    if blind is not None:
        for scoring in real:
            left = real[scoring]["within_auc"]
            right = blind[scoring]["within_auc"]
            blind_gap_by_arm[scoring] = (
                left - right if left is not None and right is not None else None
            )

    return {
        "primary": primary,
        "scorings": real,
        "blind": blind,
        "blind_gap": blind_gap,
        "blind_gap_by_arm": blind_gap_by_arm,
        "floor": {
            "n_floor_pairs": int(floor.sum()),
            "floor_share": float(floor.mean()) if floor.size else None,
        },
        "secondary_non_floor": non_floor[primary],
        "secondary_non_floor_by_arm": non_floor,
        "notes": [
            "the four arms are CS-30's, ported; the primary arm is fixed a priori "
            "by the UPR prior (sum_positive), never selected on this run's numbers.",
            "the judged cells are scored out-of-fold (4/5 of the pairs); the control "
            "cells are scored by the full reviewed-set map (they never entered any "
            "fit). The per-fold control means are recorded beside the full-map one.",
            "the floor carriers (the pairs whose fold map has no table for their "
            "question) sit at the row's floor and can never be ranked first; the "
            "primary number is over all pairs, the secondary reading excludes them.",
        ],
    }


def build_dictionary_model(
    view: DatasetView,
    *,
    grids: Mapping[str, Any],
    blind_grids: Mapping[str, Any] | None,
    reference_similarities: np.ndarray,
    reference_name: str,
    folds: int,
    seed: int,
    resamples: int,
    bootstrap_seed: int,
    primary: str = PRIMARY_SCORING,
    probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The artifact's ``kind: dictionary`` model entry (the arm + its comparison).

    Args:
        view: The dataset view.
        grids: The real cross-fit grids.
        blind_grids: The label-blind grids.
        reference_similarities: The run's full-text cosine matrix.
        reference_name: The reference model's name (the run's embedder).
        folds: The cross-fit's fold count.
        seed: The cross-fit's seed.
        resamples: The paired bootstrap's resamples.
        bootstrap_seed: The paired bootstrap's seed.
        primary: The pre-registered primary arm.
        probe: The control-vintage block (the per-fold control means).

    Returns:
        The model entry: the arm's measured block, its comparison against the
        cosine and the readout.
    """
    matrix = grids["scorings"][primary]
    readout = measure_dictionary_arms(view, grids, blind_grids, primary=primary)
    if probe is not None:
        readout["control_vintage"] = dict(probe)
    reference_measured = measure_arm(view, reference_similarities)
    reference_stats = _arm_stats(view, reference_measured)
    readout["reference"] = {
        "model": reference_name,
        "within_auc": reference_stats["within_auc"],
        "pooled_auc": reference_stats["pooled_auc"],
        "macro_auc": reference_stats["macro_auc"],
        "by_document": reference_stats["by_document"],
    }

    arm_block = score_arm(
        view,
        arm_name=DICTIONARY_ARM,
        similarities=matrix,
        paragraph_side=_dictionary_side(
            view.paragraphs,
            "the dictionary arm reads the paragraph's words; no embedding is encoded",
        ),
        question_side=_dictionary_side(
            view.questions,
            "the dictionary arm reads the question's table; no embedding is encoded",
        ),
    )
    arm_block["scoring"] = primary
    arm_block["unit"] = "the question map's word weights (no embedding)"
    comparison = compare_arms(
        view,
        arm_name=DICTIONARY_ARM,
        reference_similarities=reference_similarities,
        arm_similarities=matrix,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    # The four arms' paired comparisons against the cosine, so the spread is
    # **measured** and not hand-derived. They are observations of the spread
    # (the pre-registered test is the primary arm's), never a second headline.
    spread: dict[str, Any] = {}
    for scoring, scoring_matrix in grids["scorings"].items():
        spread[scoring] = (
            comparison
            if scoring == primary
            else compare_arms(
                view,
                arm_name=scoring,
                reference_similarities=reference_similarities,
                arm_similarities=scoring_matrix,
                resamples=resamples,
                seed=bootstrap_seed,
            )
        )
    readout["spread_comparisons"] = spread
    readout["spread_note"] = (
        "the four arms' paired comparisons against the same cosine: the primary "
        "arm's is the pre-registered test; the other three are the spread, "
        "measured (not selected on this run's numbers) — an observation the "
        "reviewer may turn into a pre-registered test of its own."
    )
    return {
        "name": DICTIONARY_ARM,
        "kind": "dictionary",
        "model": {
            "name": DICTIONARY_ARM,
            "model_id": "the question map (CS-48)",
            "backend": "n/a",
            "unit": "the reviewed pairs' own word-to-question association",
            "primary_scoring": primary,
            "reference_model": reference_name,
            "folds": int(folds),
            "seed": int(seed),
        },
        "arms": {DICTIONARY_ARM: arm_block},
        "comparisons": {DICTIONARY_ARM: comparison},
        "readout": readout,
    }
