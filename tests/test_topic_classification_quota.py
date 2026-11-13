"""Hand-computed tests for the zero-shot quota insertion (TC-11)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.quota import quota_insertion


def test_quota_inserts_at_the_bottom_of_the_shortlist() -> None:
    base = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    candidates = np.array([[5, 4, 3]], dtype=np.int32)
    out = quota_insertion(base, candidates, quota=1, depth=3)
    # head [0,1,2] -> keep [0,1], insert 5, displace 2; tail [3,4,5] minus used
    assert out[0].tolist() == [0, 1, 5, 2, 3, 4]


def test_two_slots_displace_the_last_two() -> None:
    base = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    candidates = np.array([[4, 5, 3]], dtype=np.int32)
    out = quota_insertion(base, candidates, quota=2, depth=3)
    assert out[0].tolist() == [0, 4, 5, 1, 2, 3]


def test_candidate_already_in_the_shortlist_is_skipped() -> None:
    base = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    candidates = np.array([[1, 4, 5]], dtype=np.int32)
    out = quota_insertion(base, candidates, quota=2, depth=3)
    # 1 is already in the head, so 4 and 5 fill the two slots; 1 and 2 are displaced
    assert out[0].tolist() == [0, 4, 5, 1, 2, 3]


def test_quota_zero_is_a_copy() -> None:
    base = np.array([[0, 1, 2, 3]], dtype=np.int32)
    out = quota_insertion(base, np.array([[3, 2]], dtype=np.int32), quota=0, depth=2)
    assert out.tolist() == base.tolist()
    assert out is not base


def test_more_candidates_than_available_slots_is_fine() -> None:
    base = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)
    candidates = np.array([[4, 5]], dtype=np.int32)
    out = quota_insertion(base, candidates, quota=3, depth=3)
    # only two candidates exist, so only two are inserted
    assert out[0].tolist() == [0, 4, 5, 1, 2, 3]


def test_every_row_stays_a_permutation() -> None:
    rng = np.random.default_rng(0)
    base = np.stack([rng.permutation(10) for _ in range(5)]).astype(np.int32)
    candidates = np.stack([rng.permutation(10)[:4] for _ in range(5)]).astype(np.int32)
    out = quota_insertion(base, candidates, quota=3, depth=6)
    for row in range(5):
        assert sorted(out[row].tolist()) == list(range(10))


def test_validation() -> None:
    base = np.array([[0, 1, 2]], dtype=np.int32)
    candidates = np.array([[2]], dtype=np.int32)
    with pytest.raises(ValueError):
        quota_insertion(np.zeros(3, dtype=np.int32), candidates, quota=1)
    with pytest.raises(ValueError):
        quota_insertion(base, np.array([[0, 1, 2, 3]], dtype=np.int32), quota=1)
    with pytest.raises(ValueError):
        quota_insertion(base, candidates, quota=1, depth=0)
    with pytest.raises(ValueError):
        quota_insertion(base, candidates, quota=-1, depth=2)
    with pytest.raises(ValueError):
        quota_insertion(base, candidates, quota=3, depth=2)
