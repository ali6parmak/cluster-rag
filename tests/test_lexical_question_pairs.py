"""Unit tests for the question-pair relation (the track's CS-57).

The hand-computed cases: the vocabulary Jaccard, the co-occurrence from the
shared paragraphs, the relation readout (the co-occurring pairs' mean against
the others'), the closest-pairs readout and the vocabulary-less class.
"""

from __future__ import annotations

import pytest

from cluster_rag.text_processors.lexical.question_pairs import (
    co_occurrence_matrix,
    jaccard_matrix,
    question_pairs_readout,
)

Q1 = "torture in prison"
Q2 = "torture in prison cells"
Q3 = "water rights"


def _data():
    """Three questions over two paragraphs: Q1 and Q2 share the first."""
    questions = (Q1, Q2, Q3)
    pair_questions = [Q1, Q2, Q3]
    pair_paragraphs = ["the first paragraph", "the first paragraph", "the second paragraph"]
    return questions, pair_questions, pair_paragraphs


def test_the_jaccard_is_hand_computed() -> None:
    """J(Q1, Q2) = 2 shared of 3 = 2/3; nothing shared elsewhere."""
    questions, _, _ = _data()
    from cluster_rag.text_processors.lexical.question_pairs import _vocabularies

    vocabularies = _vocabularies(questions)
    assert vocabularies[Q1] == frozenset({"torture", "prison"})
    assert vocabularies[Q2] == frozenset({"torture", "prison", "cells"})
    matrix = jaccard_matrix(questions, vocabularies)
    assert matrix[0, 1] == pytest.approx(2.0 / 3.0)
    assert matrix[0, 2] == 0.0
    assert matrix[1, 2] == 0.0
    assert matrix[0, 0] == 0.0  # the diagonal is not a pair


def test_the_co_occurrence_is_the_shared_paragraph() -> None:
    """Q1 and Q2 share the first paragraph; Q3 shares nothing."""
    questions, pair_questions, pair_paragraphs = _data()
    matrix = co_occurrence_matrix(questions, pair_questions, pair_paragraphs)
    assert matrix[0, 1] is True or matrix[0, 1]
    assert matrix[1, 0] is True or matrix[1, 0]
    assert not matrix[0, 2]
    assert not matrix[2, 0]
    # the matrix is symmetric
    assert (matrix == matrix.T).all()


def test_the_relation_readout_is_hand_computed() -> None:
    """The co-occurring pair is the only lexical one: delta = 2/3."""
    questions, pair_questions, pair_paragraphs = _data()
    block = question_pairs_readout(
        questions,
        pair_questions,
        pair_paragraphs,
        resamples=50,
        seed=42,
    )
    assert block["n_questions"] == 3
    assert block["n_question_pairs"] == 3
    assert block["relation"]["co_occurring_mean"] == pytest.approx(2.0 / 3.0)
    assert block["relation"]["other_mean"] == 0.0
    assert block["relation"]["delta"] == pytest.approx(2.0 / 3.0)
    assert block["relation"]["ci_low"] <= block["relation"]["delta"]
    assert block["relation"]["ci_high"] >= block["relation"]["delta"]
    # the closest pairs: Q1-Q2 first (2/3), then the two zeros
    assert block["closest"]["pairs"][0]["jaccard"] == pytest.approx(2.0 / 3.0)
    assert block["closest"]["pairs"][0]["co_occur"] is True
    assert block["closest"]["co_occurrence_share"] == pytest.approx(1.0 / 3.0)
    assert block["co_occurrence"]["density"] == pytest.approx(1.0 / 3.0)
    # deterministic: the same seed, the same interval
    again = question_pairs_readout(
        questions, pair_questions, pair_paragraphs, resamples=50, seed=42
    )
    assert again["relation"]["ci_low"] == block["relation"]["ci_low"]


def test_the_vocabulary_less_class_is_the_shared_vocabulary() -> None:
    """Q1 has no word of its own (both of its words are also in Q2)."""
    questions, pair_questions, pair_paragraphs = _data()
    block = question_pairs_readout(
        questions, pair_questions, pair_paragraphs, resamples=10, seed=42
    )
    assert block["vocabulary_less"]["n_questions"] == 1
    assert block["vocabulary_less"]["questions"] == [Q1]
    assert block["vocabulary_less"]["share"] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="same length"):
        question_pairs_readout(questions, pair_questions, pair_paragraphs[:-1])
    with pytest.raises(ValueError, match="at least 2 questions"):
        question_pairs_readout((Q1,), [Q1], ["p"])
