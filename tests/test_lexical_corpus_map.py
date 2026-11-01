"""Unit tests for the corpus map as a scorer (the track's CS-30).

The hand-computed cases: the four scoring functions on a tiny map whose
log-odds are worked out by hand, the empty table's floor, the cross-fit's
determinism, and the map's train-only boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor, predictor_requirements
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import (
    FLOOR_GAP,
    SCORING_FUNCTIONS,
    cross_fit_scores,
    score_cells,
)

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in

# The hand-computed log-odds of the tiny map (4 texts, 2 per label):
#   a word in 2 of 4 texts, both of them with the label: log(5) + log(5)
#   a word in 1 of 4 texts, with the label:      log(3) + log(5/3)
LOG_ODDS_DOUBLE = math.log(5.0) + math.log(5.0)
LOG_ODDS_SINGLE = math.log(3.0) + math.log(5.0 / 3.0)


def _records() -> tuple[Record, ...]:
    return (
        Record(id="t1", text="alpha bravo", language="en", labels=["A"]),
        Record(id="t2", text="alpha charlie", language="en", labels=["A"]),
        Record(id="t3", text="delta echo", language="en", labels=["B"]),
        Record(id="t4", text="delta foxtrot", language="en", labels=["B"]),
    )


def _map():
    """The tiny corpus map: A's table {alpha, bravo, charlie}, B's {delta, echo, foxtrot}."""
    records = _records()
    return build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        LABELS,
    )


def _context() -> PredictorContext:
    """Train-only context: A x2, B x2, Z x0."""
    return PredictorContext(labels=LABELS, train_records=_records())


def test_the_four_scoring_functions_are_hand_computable() -> None:
    """The text 'alpha bravo' against the tiny map, arm by arm."""
    corpus_map = _map()
    scores = score_cells(["alpha bravo"], corpus_map, scoring="sum_logodds")
    assert scores.shape == (1, 3)
    assert scores[0, 0] == pytest.approx(LOG_ODDS_DOUBLE + LOG_ODDS_SINGLE)
    # a word that is not in B's table contributes 0
    assert scores[0, 1] == 0.0
    # the label with no train example has an empty table: the row's floor
    assert scores[0, 2] == pytest.approx(min(scores[0, 0], scores[0, 1]) - FLOOR_GAP)

    positive = score_cells(["alpha bravo"], corpus_map, scoring="sum_positive")
    assert positive[0, 0] == pytest.approx(LOG_ODDS_DOUBLE + LOG_ODDS_SINGLE)

    best = score_cells(["alpha bravo"], corpus_map, scoring="max_logodds")
    assert best[0, 0] == pytest.approx(LOG_ODDS_DOUBLE)
    # no word in B's table: "no evidence" reads 0.0
    assert best[0, 1] == 0.0

    count = score_cells(["alpha bravo"], corpus_map, scoring="support_count")
    assert count[0, 0] == 2.0
    assert count[0, 1] == 0.0


def test_a_repeated_word_counts_once() -> None:
    """The map's unit is the word's presence, so a repetition adds nothing."""
    corpus_map = _map()
    once = score_cells(["alpha"], corpus_map, scoring="support_count")
    thrice = score_cells(["alpha alpha alpha"], corpus_map, scoring="support_count")
    assert once[0, 0] == thrice[0, 0] == 1.0


def test_the_empty_table_is_the_floor_and_the_others_are_not() -> None:
    """A label with no train example can never outrank a label that has one."""
    corpus_map = _map()
    scores = score_cells(["alpha bravo"], corpus_map, scoring="sum_logodds")
    assert scores[0, 2] < scores[0, 0]
    assert scores[0, 2] < scores[0, 1]
    # the floor is strictly below every seen label, whatever the arm's scale
    negative = score_cells(["echo foxtrot"], corpus_map, scoring="sum_logodds")
    assert negative[0, 2] < negative[0, 0] < negative[0, 1]
    assert np.isfinite(scores).all()


def test_an_unknown_scoring_function_fails() -> None:
    """A typo in the arm's name fails loudly, before any run."""
    with pytest.raises(ValueError, match="Unknown scoring function"):
        score_cells(["alpha"], _map(), scoring="sum_everything")


def test_the_cross_fit_is_deterministic_and_keeps_the_corpus_order() -> None:
    """The same corpus and seed give the same out-of-fold scores, row by row."""
    records = _records()
    texts = [record.text for record in records]
    label_sets = [record.labels for record in records]
    first = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42)
    second = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42)
    for arm in SCORING_FUNCTIONS:
        assert first[arm].shape == (4, 3)
        assert np.array_equal(first[arm], second[arm])
    # the label with no example is never above a label that has one
    scores = first["sum_logodds"]
    assert (scores[:, 2:3] <= scores[:, :2]).all()
    # and a label whose only example sits in the held-out fold is unfittable in
    # that fold too, so it shares the floor there (the out-of-fold cost of the
    # rare labels, the same for every arm)
    assert (scores[:, 0] == -FLOOR_GAP).any()


def test_the_cross_fit_needs_two_folds_and_matching_lengths() -> None:
    """A degenerate request fails before anything is computed."""
    records = _records()
    texts = [record.text for record in records]
    label_sets = [record.labels for record in records]
    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_fit_scores(texts, label_sets, LABELS, folds=1)
    with pytest.raises(ValueError, match="label sets"):
        cross_fit_scores(texts, label_sets[:-1], LABELS, folds=2)


def test_the_predictor_needs_no_embeddings() -> None:
    """It is embedding-free: the evaluate command wires no artifact for it."""
    requires, kinds, needs_labels = predictor_requirements("corpus_map", {})
    assert requires is False and kinds == () and needs_labels is False


def test_the_map_is_built_from_the_train_split_only() -> None:
    """A word the train split never saw is not in the map and changes nothing."""
    predictor = build_predictor("corpus_map", {}, _context(), seed=42)
    plain = predictor.score(["alpha bravo"])
    with_oov = predictor.score(["alpha bravo zulu"])
    assert np.array_equal(plain, with_oov)


def test_the_fit_side_is_reported_and_the_selection_is_out_of_fold() -> None:
    """The fit side's numbers are in the lineage; the selection is the out-of-fold one."""
    predictor = build_predictor("corpus_map", {}, _context(), seed=42)
    fit_side = predictor.describe()["fit_side"]
    assert fit_side["selected_by"] == "out_of_fold_micro_f1"
    assert set(fit_side["out_of_fold_micro_f1"]) == set(SCORING_FUNCTIONS)
    assert set(fit_side["in_fold_micro_f1"]) == set(SCORING_FUNCTIONS)
    # the selection is the out-of-fold maximum
    selected = max(
        SCORING_FUNCTIONS,
        key=lambda arm: fit_side["out_of_fold_micro_f1"][arm],
    )
    assert fit_side["selected"] == selected
    assert fit_side["cross_fit_folds"] == 5


def test_the_map_shape_is_reported() -> None:
    """The lineage carries the map's own shape (the CS-23 facts)."""
    predictor = build_predictor("corpus_map", {}, _context(), seed=42)
    corpus = predictor.describe()["map"]
    assert corpus["split"] == "train"
    assert corpus["n_texts"] == 4
    assert corpus["n_labels"] == 3
    assert corpus["labels_with_examples"] == 2
    assert corpus["labels_without_examples"] == ["Z"]
    assert corpus["n_words"] > 0 and corpus["n_pairs"] > 0


def test_the_predictor_is_deterministic() -> None:
    """The same train split and seed give the same scores, bit for bit."""
    first = build_predictor("corpus_map", {}, _context(), seed=42)
    second = build_predictor("corpus_map", {}, _context(), seed=42)
    assert np.array_equal(first.score(["alpha bravo"]), second.score(["alpha bravo"]))


def test_unsupported_and_invalid_parameters_fail_before_the_fit() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("corpus_map", {"temperature": 1.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="scoring"):
        build_predictor("corpus_map", {"scoring": "sum_everything"}, _context(), seed=0)
    with pytest.raises(ValueError, match="fit_side_k"):
        build_predictor("corpus_map", {"fit_side_k": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="folds"):
        build_predictor("corpus_map", {"folds": 1}, _context(), seed=0)
