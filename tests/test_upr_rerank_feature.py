"""Tests for the reranker's score as a **pack feature** (the track's CS-16).

Contracts under test:

- the **pre-registered fill value** of the candidate-restricted feature: the
  cells outside the candidate set take the row's **minimum scored value**,
  so the fill never lifts a non-candidate above a candidate and never
  manufactures a separation the reranker did not produce;
- the pack's **rerank** column: it is filled from the matrix the caller
  passes in (the pack never calls the model), and the group is refused
  without one.
"""

from __future__ import annotations

import pytest

from cluster_rag.text_processors.features.upr_features import (
    RERANK_FEATURE_NAME,
    UPR_FEATURE_GROUP_PREFIXES,
)
from cluster_rag.text_processors.rerank.upr_rerank import fill_restricted_scores


def test_the_fill_is_the_rows_minimum() -> None:
    """The cells outside the candidate set take the row's floor."""
    matrix = fill_restricted_scores([(0, 2)], [[0.5, 0.7]], 4)
    assert matrix[0, 0] == pytest.approx(0.5)
    assert matrix[0, 2] == pytest.approx(0.7)
    # the floor: the row's minimum scored value
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[0, 3] == pytest.approx(0.5)


def test_the_fill_never_lifts_a_non_candidate() -> None:
    """No cell outside the set is above the set's own minimum."""
    matrix = fill_restricted_scores([(1,)], [[0.2]], 3)
    assert matrix[0, 0] == pytest.approx(0.2)
    assert matrix[0, 2] == pytest.approx(0.2)


def test_the_fill_keeps_a_zero_scored_candidate() -> None:
    """A candidate whose score is 0.0 is not mistaken for a filled cell."""
    matrix = fill_restricted_scores([(0, 1)], [[0.0, 0.7]], 3)
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[0, 1] == pytest.approx(0.7)
    assert matrix[0, 2] == pytest.approx(0.0)


def test_the_fill_rejects_mismatched_candidates_and_scores() -> None:
    """The candidate columns and the scores must agree."""
    with pytest.raises(ValueError, match="same length"):
        fill_restricted_scores([(0,)], [[0.5], [0.6]], 3)
    with pytest.raises(ValueError, match="disagree"):
        fill_restricted_scores([(0, 1)], [[0.5]], 3)


def test_the_rerank_group_is_a_pack_group() -> None:
    """The pack knows the group (and its column's prefix)."""
    assert "rerank" in UPR_FEATURE_GROUP_PREFIXES
    assert RERANK_FEATURE_NAME.startswith("rerank_")
