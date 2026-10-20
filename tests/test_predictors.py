"""Unit tests for the trivial predictors and the predictor registry (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext

LABELS = ("A", "B", "C", "Z")  # Z is the unseen holdout-label stand-in


def _record(rid: str, labels: list[str]) -> Record:
    return Record(id=rid, text=f"text {rid}", language="en", labels=labels)


def _context() -> PredictorContext:
    """Train-only context: A x2, B x1, C x1, Z x0."""
    return PredictorContext(
        labels=LABELS,
        train_records=(
            _record("r1", ["A", "B"]),
            _record("r2", ["A"]),
            _record("r3", ["C"]),
        ),
    )


def test_most_frequent_scores_are_train_counts_only() -> None:
    """Scored by train frequency, constant across records, 0 for unseen labels."""
    predictor = build_predictor("most_frequent", {}, _context(), seed=0)
    scores = predictor.score(["any text", "another"])
    assert scores.shape == (2, 4) and np.issubdtype(scores.dtype, np.floating)
    assert scores[0].tolist() == [2.0, 1.0, 1.0, 0.0]
    assert scores[1].tolist() == scores[0].tolist()  # ignores the query text entirely
    assert predictor.describe() == {
        "name": "most_frequent",
        "train_records": 3,
        "n_known_labels": 3,
    }


def test_random_predictor_is_seeded_and_reproducible() -> None:
    """Same seed -> identical scores bit-for-bit; different seed differs."""
    same_seed_a = build_predictor("random", {}, _context(), seed=7)
    same_seed_b = build_predictor("random", {}, _context(), seed=7)
    other_seed = build_predictor("random", {}, _context(), seed=8)
    matrix_a = same_seed_a.score(["x", "y"])
    assert np.array_equal(matrix_a, same_seed_b.score(["x", "y"]))
    assert not np.array_equal(matrix_a, other_seed.score(["x", "y"]))
    assert matrix_a.shape == (2, 4)
    assert (matrix_a >= 0).all() and (matrix_a < 1).all()
    assert other_seed.describe() == {"name": "random", "seed": 8}


def test_build_predictor_unknown_name_lists_known() -> None:
    """An unknown registry name fails with the supported names in the message."""
    with pytest.raises(ValueError, match="most_frequent"):
        build_predictor("knn", {}, _context(), seed=0)


def test_build_predictor_rejects_unsupported_params() -> None:
    """Unsupported parameter keys fail at construction, before any run."""
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("most_frequent", {"temperature": 1.0}, _context(), seed=0)


def test_build_predictor_passes_params_through() -> None:
    """Params reach the predictor (empty dict is valid for the trivial ones)."""
    predictor = build_predictor("random", {}, _context(), seed=3)
    assert predictor.describe()["seed"] == 3
    assert (
        build_predictor("most_frequent", {}, _context(), seed=3).describe()["name"]
        == "most_frequent"
    )
