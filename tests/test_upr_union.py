"""Tests for the UPR union candidate generator (the track's CS-14).

Contracts under test:

- the **z-normalization**: every row's scores are z-scored within the text's
  own label list (a row with no spread is scored at zero);
- the **z-max**: the union is the larger of the two z-scored matrices, and
  the two matrices must share one shape;
- the **block**: the union's own numbers, the two references' numbers and the
  **label-blind control's** numbers beside them.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.text_processors.decision.upr_union import (
    row_z_scores,
    upr_union_block,
    z_max_union,
)


def test_the_row_z_scores_normalize_within_the_row() -> None:
    """Every row is z-scored over its own labels."""
    z = row_z_scores(np.asarray([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]))
    assert z[0].mean() == pytest.approx(0.0)
    assert z[0, 2] > z[0, 0]
    # a row with no spread is scored at zero
    assert np.allclose(z[1], 0.0)


def test_the_z_max_takes_the_larger_of_the_two() -> None:
    """The union's score is the larger of the two z-scores."""
    left = np.asarray([[1.0, 2.0]])
    right = np.asarray([[0.5, 4.0]])
    union = z_max_union(left, right)
    # both rows are z-scored to [-1, 1]; the union keeps the larger of the two
    assert union[0, 0] == pytest.approx(-1.0)
    assert union[0, 1] == pytest.approx(1.0)


def test_the_z_max_rejects_mismatched_shapes() -> None:
    """The two matrices must share one shape."""
    with pytest.raises(ValueError, match="share one shape"):
        z_max_union(np.zeros((2, 3)), np.zeros((2, 4)))


def test_the_block_carries_the_references_and_the_control() -> None:
    """The union's numbers, the two references' and the blind control's."""
    champion = np.asarray([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]])
    decision = np.asarray([[0.2, 0.7, 0.1], [0.6, 0.1, 0.2]])
    blind = np.asarray([[0.1, 0.2, 0.7], [0.2, 0.6, 0.1]])
    gold = np.asarray([[True, False, False], [False, True, False]])
    block = upr_union_block(
        champion,
        decision,
        gold,
        blind_decision=blind,
        ks=(1, 3),
        shortlist_ms=(1, 3),
    )
    assert block["union"]["shortlist_recall"]["recall@1"] == pytest.approx(0.5)
    assert block["champion"]["shortlist_recall"]["recall@1"] == pytest.approx(1.0)
    # the decision model's own top-1 is never the gold cell in this fixture
    assert block["decision_model"]["shortlist_recall"]["recall@1"] == pytest.approx(0.0)
    assert "label_blind_control" in block
    assert block["n_gold_cells"] == 2
    assert "not built" in block["note"]


def test_the_block_without_the_control() -> None:
    """The control is optional."""
    scores = np.asarray([[0.9, 0.1]])
    gold = np.asarray([[True, False]])
    block = upr_union_block(scores, scores, gold, ks=(1,), shortlist_ms=(1,))
    assert "label_blind_control" not in block
