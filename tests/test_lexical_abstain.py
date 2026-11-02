"""Unit tests for the corpus map's **abstention** (the track's CS-37).

The abstention is the pack's channel: a label with **no table at all** carries
``NaN`` ("no evidence") instead of the floor ("impossible"), so a learner can
tell the two apart. The classifier never abstains — the harness rejects
non-finite scores — so the floor stays exactly as it is there.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import (
    FLOOR_GAP,
    cross_fit_scores,
    score_cells,
)

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in


def _records() -> tuple[Record, ...]:
    return (
        Record(id="t1", text="alpha bravo", language="en", labels=["A"]),
        Record(id="t2", text="alpha charlie", language="en", labels=["A"]),
        Record(id="t3", text="delta echo", language="en", labels=["B"]),
        Record(id="t4", text="delta foxtrot", language="en", labels=["B"]),
    )


def _map():
    records = _records()
    return build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        LABELS,
    )


def test_the_abstention_carries_no_value_where_there_is_no_table() -> None:
    """Z has no table: its cells carry NaN, not a number."""
    scores = score_cells(["alpha bravo"], _map(), scoring="support_count", abstain=True)
    assert np.isnan(scores[0, 2])
    assert scores[0, 0] == 2.0  # A: 'alpha' and 'bravo'
    assert scores[0, 1] == 0.0  # B: the text says nothing about it


def test_the_abstention_changes_nothing_but_the_empty_tables() -> None:
    """The seen labels' scores are identical with and without the abstention."""
    corpus_map = _map()
    plain = score_cells(["alpha bravo", "delta echo"], corpus_map, scoring="sum_logodds")
    abstained = score_cells(
        ["alpha bravo", "delta echo"], corpus_map, scoring="sum_logodds", abstain=True
    )
    assert np.array_equal(plain[:, :2], abstained[:, :2])
    # the floor (the default) is finite; the abstention is not
    assert np.isfinite(plain).all()
    assert np.isnan(abstained[:, 2]).all()


def test_the_floor_is_still_the_floor_without_the_abstention() -> None:
    """The classifier's own reading is untouched: the empty table carries the floor."""
    scores = score_cells(["alpha bravo"], _map(), scoring="support_count")
    assert scores[0, 2] == pytest.approx(min(scores[0, 0], scores[0, 1]) - FLOOR_GAP)


def test_the_abstention_rides_into_the_cross_fit() -> None:
    """A label with no example in a fold's texts abstains in that fold."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    crossed = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42, abstain=True)
    scores = crossed["sum_logodds"]
    # Z has no example at all: it abstains in every fold
    assert np.isnan(scores[:, 2]).all()
    # and a fold whose training texts hold no example of A (or of B) abstains
    # for that label too — the fold's map really does not know it
    assert np.isnan(scores[:, :2]).any(axis=1).all()
    finite = np.isfinite(scores)
    assert finite[~np.isnan(scores)].all()
    # without the abstention the same fold's cells carry the floor
    plain = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42)
    assert np.isfinite(plain["sum_logodds"]).all()


def test_the_predictor_never_abstains() -> None:
    """The harness rejects non-finite scores: the classifier's scores stay finite."""
    context = PredictorContext(labels=LABELS, train_records=_records())
    predictor = build_predictor("corpus_map", {}, context, seed=42)
    scores = predictor.score(["alpha bravo"])
    assert np.isfinite(scores).all()
    assert scores[0, 2] == pytest.approx(min(scores[0, 0], scores[0, 1]) - FLOOR_GAP)
