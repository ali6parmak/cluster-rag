"""Unit tests for the **lexical kNN** predictor (the track's CS-41).

The words alone as a **retriever**: the TF-IDF nearest neighbours over the
train split, their labels voted with the main pipeline's own kNN machinery.
The tests are about the contract: the votes are the machinery's own (the sum
of the neighbours' clipped similarities), the vocabulary is fitted on the
train split only, the unseen label never gets a vote, the label-blind
counterpart destroys the association, and the scores are deterministic.
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


def test_the_nearest_neighbours_vote_for_their_own_label() -> None:
    """The text's own words make its own label the neighbours' label."""
    predictor = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    scores = predictor.score(["alpha alpha alpha", "beta beta beta"])
    assert scores.shape == (2, 3)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_the_votes_are_the_knn_machinerys_own() -> None:
    """The score is the sum of the k neighbours' clipped similarities.

    The expectation is computed by an independent path (the same vectorizer,
    the similarities by hand), so the test pins the *mechanism*, not a number
    copied from the implementation.
    """
    context = _context()
    predictor = build_predictor("lexical_knn", {"k": 2}, context, seed=0)
    texts = ["alpha alpha beta"]
    scores = predictor.score(texts)

    # the same vectorizer, fitted on the train split only
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2")
    train = [record.text for record in context.train_records]
    matrix = vectorizer.fit_transform(train)
    query = vectorizer.transform(texts)
    similarities = (query @ matrix.T).toarray()[0]
    order = np.argsort(-similarities, kind="stable")[:2]
    expected_a = float(similarities[order][0] + similarities[order][1])
    # both neighbours carry A (the query is closest to the alpha texts)
    assert [context.train_records[int(index)].labels for index in order] == [["A"], ["A"]]
    assert scores[0, 0] == pytest.approx(expected_a, abs=1e-6)
    assert scores[0, 1] == pytest.approx(0.0)


def test_a_label_without_a_train_example_never_gets_a_vote() -> None:
    """The unseen holdout label: no train example -> no vote -> score 0."""
    from cluster_rag.eval.multilabel import apply_decision_rule

    predictor = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    scores = predictor.score(["alpha alpha alpha", "beta beta beta"])
    assert scores[:, 2].tolist() == [0.0, 0.0]
    # the neighbours' own label is the only one that gets a vote
    assert scores[0, 0] > 0 and scores[0, 1] == 0.0
    assert scores[1, 1] > 0 and scores[1, 0] == 0.0
    # and the unseen label never enters a top-2 (the harness's own tie-break
    # puts the label space's earlier columns first)
    predicted = apply_decision_rule(scores, rule="top_k", k=2)
    assert predicted[:, 2].sum() == 0


def test_the_vocabulary_is_fitted_on_train_only() -> None:
    """A word the train split never saw cannot make a neighbour."""
    predictor = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    scores = predictor.score(["unseenword unseenword"])
    # no known feature: no neighbour with a positive similarity -> no votes
    assert scores[0].tolist() == [0.0, 0.0, 0.0]


def test_the_label_blind_counterpart_destroys_the_association() -> None:
    """The control: the retrieval is untouched, the label association is not."""
    real = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    blind = build_predictor("lexical_knn", {"k": 3, "label_blind": True}, _context(), seed=0)
    texts = ["alpha alpha alpha", "beta beta beta"]
    real_scores = real.score(texts)
    blind_scores = blind.score(texts)
    # the retrieval is the same (the same neighbours), the votes are not
    assert not np.allclose(real_scores, blind_scores)
    assert blind.describe()["label_blind"] is True
    assert real.describe()["label_blind"] is False


def test_the_fit_is_deterministic() -> None:
    """The same train split gives the same scores, bit for bit."""
    first = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    again = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    texts = ["alpha alpha alpha", "beta two three"]
    assert np.array_equal(first.score(texts), again.score(texts))


def test_the_predictor_needs_no_embeddings() -> None:
    """The lexical kNN is embedding-free: no embedding run, no vectors."""
    requires, kinds, needs_labels = predictor_requirements("lexical_knn", {"k": 3})
    assert requires is False
    assert kinds == ()
    assert needs_labels is False


def test_describe_reports_the_facts() -> None:
    """The lineage the run artifact carries."""
    predictor = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    description = predictor.describe()
    assert description["name"] == "lexical_knn"
    assert description["k"] == 3
    assert description["power"] == 1.0
    assert description["n_train_examples"] == 6
    assert description["n_labels"] == 3
    assert description["n_features"] > 0
    assert "fitted on the train split only" in description["vectorizer"]
    assert "knn_vote_scores" in description["votes"]


def test_unsupported_and_invalid_parameters_fail_before_the_fit() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    with pytest.raises(ValueError, match="does not support parameter"):
        build_predictor("lexical_knn", {"k": 3, "temperature": 1.0}, _context(), seed=0)
    with pytest.raises(ValueError, match="requires params\\['k'\\]"):
        build_predictor("lexical_knn", {}, _context(), seed=0)
    with pytest.raises(ValueError, match="params\\['k'\\]"):
        build_predictor("lexical_knn", {"k": 0}, _context(), seed=0)
    with pytest.raises(ValueError, match="params\\['power'\\]"):
        build_predictor("lexical_knn", {"k": 3, "power": "one"}, _context(), seed=0)
    with pytest.raises(ValueError, match="params\\['label_blind'\\]"):
        build_predictor("lexical_knn", {"k": 3, "label_blind": "yes"}, _context(), seed=0)
    with pytest.raises(ValueError, match="ngram_max"):
        build_predictor("lexical_knn", {"k": 3, "ngram_max": 0}, _context(), seed=0)


def test_the_tfidf_knobs_are_the_lexical_floors_own() -> None:
    """The two lexical predictors share the TF-IDF's knobs (CS-29's)."""
    from cluster_rag.predictors.lexical_tfidf import TFIDF_PARAMS

    predictor = build_predictor("lexical_knn", {"k": 3}, _context(), seed=0)
    assert predictor.describe()["tfidf_params"] == dict(TFIDF_PARAMS)
