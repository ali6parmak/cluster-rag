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
from cluster_rag.text_processors.features.question_map import UNIT_PHRASES
from cluster_rag.text_processors.lexical.question_map_score import (
    PRIMARY_SCORING,
    UNION_SCORING,
)

logger = logging.getLogger(__name__)

# The arm's name in the artifact and the table.
DICTIONARY_ARM = "dictionary"

# The registered bar of the label-blind test (CS-49's condition 2): the arm's
# within-document AUC minus the blind map's must be at least this.
BLIND_GAP_BAR = 0.05

# The registered bar of the unit test (CS-53's condition 1): the union arm's
# within-document AUC minus the word arm's must be at least this for "phrases
# work"; below it the increment is a null (the registered prediction).
UNION_BAR = 0.05

# The band the null prediction is stated against (CS-53): the union is
# predicted to land within this of the word arm — the two halves read the same
# evidence (a bigram occurs only where both its words do).
UNION_NULL_BAND = 0.02


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


def class_arms_block(
    view: DatasetView,
    class_grids: Mapping[str, Mapping[str, Any]],
    *,
    primary: str,
    full_matrix: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The class-restricted arms' readings, each against the **full** arm (CS-51).

    The increment test: a restricted arm is scored behind the same folds and
    compared with the full primary arm on the same pairs, so the paired
    within-document difference is what the hard filter changes (the cosine
    reference is not involved).

    Args:
        view: The dataset view.
        class_grids: The restricted grids, keyed by the restriction name (each
            one's ``scorings[primary]`` is the matrix).
        primary: The base arm the restrictions are applied to.
        full_matrix: The full primary arm's grid.
        resamples: The paired bootstrap's resamples.
        seed: The paired bootstrap's seed.

    Returns:
        The readout block: per restriction, its AUC, the paired difference
        against the full arm, its floor and its non-floor reading.
    """
    block: dict[str, Any] = {}
    for name, grids in class_grids.items():
        matrix = grids["scorings"][primary]
        measured = measure_arm(view, matrix)
        stats = _arm_stats(view, measured)
        floor = np.asarray(grids["floor"], dtype=bool)
        comparison = compare_arms(
            view,
            arm_name=f"{primary}_{name}",
            reference_similarities=full_matrix,
            arm_similarities=matrix,
            resamples=resamples,
            seed=seed,
        )
        paired = comparison["auc_decomposition"]["within_group_paired"]
        block[name] = {
            "within_auc": stats["within_auc"],
            "delta_vs_full": paired["delta_auc"],
            "ci_low": paired["ci_low"],
            "ci_high": paired["ci_high"],
            "p_delta_positive": paired["p_delta_positive"],
            "by_document": stats["by_document"],
            "floor_pairs": int(floor.sum()),
            "floor_share": float(floor.mean()) if floor.size else None,
            "non_floor": _non_floor_stats(view, measured, ~floor),
            "reference_arm": primary,
        }
    return block


def union_arm_block(
    view: DatasetView,
    union: Mapping[str, Any],
    *,
    reference_similarities: np.ndarray,
    reference_name: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The union arm's reading (CS-53): the word z arm + the phrase z arm.

    The registered test is the **increment over the word arm** on the same
    pairs (a paired within-document difference, the pairs the resampling
    unit), with the comparison against the run's cosine beside it. The block
    also carries both halves' own readings, their count arms (the
    measure-free control, CS-49's instrument) and the floor.

    Args:
        view: The dataset view.
        union: The :func:`union_grids` output (the union matrix, its two
            halves, both halves' floor flags and the count arms when the
            caller computed them).
        reference_similarities: The run's full-text cosine.
        reference_name: The reference model's name.
        resamples: The paired bootstrap's resamples.
        seed: The paired bootstrap's seed.

    Returns:
        The readout block: the union's AUCs, the increment over the word arm
        (and its bar), the cosine comparison, the halves, the counts and the
        floor.
    """
    matrix = np.asarray(union["union"], dtype=np.float64)
    word = np.asarray(union["word"], dtype=np.float64)
    phrase = np.asarray(union["phrase"], dtype=np.float64)
    word_floor = np.asarray(union["word_floor"], dtype=bool)
    phrase_floor = np.asarray(union["phrase_floor"], dtype=bool)
    floor = word_floor | phrase_floor

    measured = measure_arm(view, matrix)
    stats = _arm_stats(view, measured)
    word_stats = _arm_stats(view, measure_arm(view, word))
    phrase_stats = _arm_stats(view, measure_arm(view, phrase))

    def _count(matrix_or_none: Any) -> float | None:
        if matrix_or_none is None:
            return None
        counted = _arm_stats(view, measure_arm(view, np.asarray(matrix_or_none, dtype=np.float64)))
        return counted["within_auc"]

    comparison_word = compare_arms(
        view,
        arm_name=f"{UNION_SCORING}_vs_word",
        reference_similarities=word,
        arm_similarities=matrix,
        resamples=resamples,
        seed=seed,
    )
    comparison_cosine = compare_arms(
        view,
        arm_name=UNION_SCORING,
        reference_similarities=reference_similarities,
        arm_similarities=matrix,
        resamples=resamples,
        seed=seed,
    )
    paired = comparison_word["auc_decomposition"]["within_group_paired"]
    paired_cosine = comparison_cosine["auc_decomposition"]["within_group_paired"]
    by_document_vs_word: dict[str, float | None] = {}
    for document, value in stats["by_document"].items():
        reference_value = word_stats["by_document"].get(document)
        by_document_vs_word[document] = (
            value - reference_value if value is not None and reference_value is not None else None
        )
    return {
        "arm": UNION_SCORING,
        "scoring": union.get("scoring"),
        "within_auc": stats["within_auc"],
        "pooled_auc": stats["pooled_auc"],
        "by_document": stats["by_document"],
        "vs_word": {
            "arm": f"word {union.get('scoring')}",
            "within_auc_word": word_stats["within_auc"],
            "delta": paired["delta_auc"],
            "ci_low": paired["ci_low"],
            "ci_high": paired["ci_high"],
            "p_delta_positive": paired["p_delta_positive"],
            "bar": UNION_BAR,
            "null_band": UNION_NULL_BAND,
            "by_document": by_document_vs_word,
        },
        "vs_cosine": {
            "model": reference_name,
            "within_auc_cosine": paired_cosine["auc_reference"],
            "delta": paired_cosine["delta_auc"],
            "ci_low": paired_cosine["ci_low"],
            "ci_high": paired_cosine["ci_high"],
            "p_delta_positive": paired_cosine["p_delta_positive"],
            "pooled_delta": comparison_cosine["paired_auc"]["delta_auc"],
            "deshifted_auc": comparison_cosine["deshifted_auc"]["overall"],
        },
        "word": {
            "within_auc": word_stats["within_auc"],
            "non_floor_auc": _non_floor_stats(view, measure_arm(view, word), ~word_floor)[
                "within_auc"
            ],
            "count_within_auc": _count(union.get("word_count")),
        },
        "phrase": {
            "within_auc": phrase_stats["within_auc"],
            "non_floor_auc": _non_floor_stats(view, measure_arm(view, phrase), ~phrase_floor)[
                "within_auc"
            ],
            "count_within_auc": _count(union.get("phrase_count")),
        },
        "floor": {
            "n_floor_pairs": int(floor.sum()),
            "floor_share": float(floor.mean()) if floor.size else None,
            "word_floor_pairs": int(word_floor.sum()),
            "phrase_floor_pairs": int(phrase_floor.sum()),
            "same_floor_set": bool(np.array_equal(word_floor, phrase_floor)),
        },
        "secondary_non_floor": _non_floor_stats(view, measured, ~floor),
        "notes": [
            "the union is the word map's z-scored score plus the phrase map's own, "
            "both out-of-fold behind the same folds (CS-39's composition, ported).",
            "the registered test is the union's paired within-document difference "
            "against the word arm on the same pairs; the cosine comparison is the "
            "reference reading, not the headline.",
            "the count arms are the measure-free controls (CS-49's instrument): each "
            "half's own counts, not the sum of the two (a sum double-counts).",
            "the floor carries a pair whose question has no table in its fold's map, "
            "in EITHER half; the halves' flag sets are reported beside each other.",
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
    class_grids: Mapping[str, Mapping[str, Any]] | None = None,
    class_readout: Mapping[str, Any] | None = None,
    pos_grids: Mapping[str, Mapping[str, Any]] | None = None,
    pos_readout: Mapping[str, Any] | None = None,
    union: Mapping[str, Any] | None = None,
    unit_name: str | None = None,
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
        class_grids: The class-restricted grids (CS-51), keyed by the
            restriction name; ``None`` runs none.
        class_readout: The map's class readout (CS-51), carried beside the arms.
        pos_grids: The POS-restricted grids (CS-52), keyed by the restriction
            name; ``None`` runs none.
        pos_readout: The map's grammatical readout (CS-52), carried beside it.
        union: The union arm's grids (CS-53, the :func:`union_grids` output);
            ``None`` runs none.
        unit_name: The map's unit (CS-53), recorded on the entry so a phrase
            run's artifact is never read as a word run's.

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
    arm_block["unit"] = (
        "the question map's phrase weights (no embedding)"
        if unit_name == UNIT_PHRASES
        else "the question map's word weights (no embedding)"
    )
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
    if class_grids:
        readout["class_arms"] = class_arms_block(
            view,
            class_grids,
            primary=primary,
            full_matrix=matrix,
            resamples=resamples,
            seed=bootstrap_seed,
        )
    if pos_grids:
        readout["pos_arms"] = class_arms_block(
            view,
            pos_grids,
            primary=primary,
            full_matrix=matrix,
            resamples=resamples,
            seed=bootstrap_seed,
        )
    readout["spread_note"] = (
        "the four arms' paired comparisons against the same cosine: the primary "
        "arm's is the pre-registered test; the other three are the spread, "
        "measured (not selected on this run's numbers) — an observation the "
        "reviewer may turn into a pre-registered test of its own."
    )
    if union is not None:
        readout["union"] = union_arm_block(
            view,
            union,
            reference_similarities=reference_similarities,
            reference_name=reference_name,
            resamples=resamples,
            seed=bootstrap_seed,
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
            "map_units": unit_name,
        },
        "arms": {DICTIONARY_ARM: arm_block},
        "comparisons": {DICTIONARY_ARM: comparison},
        "readout": readout,
        "class_readout": None if class_readout is None else dict(class_readout),
        "pos_readout": None if pos_readout is None else dict(pos_readout),
    }
