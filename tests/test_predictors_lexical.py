"""Unit tests for the lexical floor predictor (the track's CS-29).

The hand-computed cases: a label's own word lifts its score, a label with no
train example scores 0.0 and never enters a top-k, a word the **train** split
never saw changes nothing (the vectorizer is fitted on train only), and the
same train split always gives the same scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor, predictor_requirements
from cluster_rag.predictors.base import PredictorContext

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in


def _record(rid: str, text: str, labels: list[str]) -> Record:
    return Record(id=rid, text=text, language="en", labels=labels)


def _context() -> PredictorContext:
    """Train-only context: three A texts, three B texts, no Z at all."""
    return PredictorContext(
        labels=LABELS,
        train_records=(
            _record("r1", "alpha alpha alpha one", ["A"]),
            _record("r2", "alpha alpha two two", ["A"]),
            _record("r3", "alpha three three three", ["A"]),
            _record("r4", "beta beta beta one", ["B"]),
            _record("r5", "beta beta two two", ["B"]),
            _record("r6", "beta three three three", ["B"]),
        ),
    )


def test_a_labels_own_word_lifts_its_score() -> None:
    """The text's own word pushes its label up and the other label down."""
    predictor = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    scores = predictor.score(["alpha alpha alpha", "beta beta beta"])
    assert scores.shape == (2, 3)
    # column order is the label space's (A, B, Z)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]
    # the scores are probabilities
    assert (scores >= 0).all() and (scores <= 1).all()


def test_a_label_without_a_train_example_scores_zero_and_never_ranks() -> None:
    """The unseen holdout label: no train example -> 0.0 -> never in a top-2."""
    predictor = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    scores = predictor.score(["alpha alpha alpha", "beta beta beta"])
    assert scores[:, 2].tolist() == [0.0, 0.0]
    # a fitted label's probability is strictly above 0, so Z is strictly last
    assert (scores[:, :2] > 0).all()


def test_the_vocabulary_is_fitted_on_train_only() -> None:
    """A word the train split never saw is ignored (it is not a feature)."""
    predictor = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    plain = predictor.score(["alpha alpha alpha"])
    with_oov = predictor.score(["alpha alpha alpha zeta"])
    assert np.array_equal(plain, with_oov)


def test_the_fit_is_deterministic() -> None:
    """The same train split gives the same scores, bit for bit."""
    first = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    second = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    assert np.array_equal(first.score(["alpha beta"]), second.score(["alpha beta"]))


def test_the_predictor_needs_no_embeddings() -> None:
    """It is embedding-free: the evaluate command wires no artifact for it."""
    requires, kinds, needs_labels = predictor_requirements("lexical_tfidf", {})
    assert requires is False and kinds == () and needs_labels is False


def test_describe_reports_the_fits_facts() -> None:
    """The lineage: the params, the vocabulary's size, the fitted labels."""
    predictor = build_predictor("lexical_tfidf", {}, _context(), seed=0)
    described = predictor.describe()
    assert described["name"] == "lexical_tfidf"
    assert described["n_train_examples"] == 6
    assert described["n_labels"] == 3
    assert described["n_labels_fitted"] == 2
    assert described["n_labels_without_examples"] == 1
    assert described["n_labels_degenerate"] == 0
    assert described["n_features"] > 0
    assert described["params"]["ngram_max"] == 2
    assert described["params"]["sublinear_tf"] is True


def test_unsupported_and_invalid_parameters_fail_before_the_fit() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("lexical_tfidf", {"temperature": 1.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="ngram_max"):
        build_predictor("lexical_tfidf", {"ngram_max": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="min_df"):
        build_predictor("lexical_tfidf", {"min_df": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="C"):
        build_predictor("lexical_tfidf", {"C": 0.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="max_iter"):
        build_predictor("lexical_tfidf", {"max_iter": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="stop_words"):
        build_predictor("lexical_tfidf", {"stop_words": "french"}, _context(), seed=0)
    with pytest.raises(ValueError, match="class_weight"):
        build_predictor("lexical_tfidf", {"class_weight": "custom"}, _context(), seed=0)


def test_a_label_every_train_text_carries_is_not_fitted() -> None:
    """A degenerate label (all train texts carry it) scores 0.0 and is counted."""
    context = PredictorContext(
        labels=LABELS,
        train_records=(
            _record("r1", "alpha alpha alpha one", ["A", "Z"]),
            _record("r2", "alpha alpha two two", ["A", "Z"]),
            _record("r3", "alpha three three three", ["A", "Z"]),
            _record("r4", "beta beta beta one", ["B", "Z"]),
            _record("r5", "beta beta two two", ["B", "Z"]),
            _record("r6", "beta three three three", ["B", "Z"]),
        ),
    )
    predictor = build_predictor("lexical_tfidf", {}, context, seed=0)
    assert predictor.describe()["n_labels_degenerate"] == 1
    assert predictor.describe()["n_labels_fitted"] == 2
    assert predictor.score(["alpha alpha alpha"])[0, 2] == 0.0
