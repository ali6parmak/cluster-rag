"""Tests for the evaluation bucket mapping (eval/buckets.py)."""

from __future__ import annotations

import pytest

from cluster_rag.eval.buckets import BUCKET_NAMES, assign_label_buckets, labels_by_bucket


def test_assign_label_buckets_boundaries_and_precedence() -> None:
    """head/head_min boundary, torso/torso_min boundary, and unseen precedence."""
    counts = {"a": 100, "b": 99, "c": 40, "d": 39, "e": 100}
    train = {"a": 70, "b": 70, "c": 28, "d": 0, "e": 0}
    buckets = assign_label_buckets(counts, train, head_min=100, torso_min=40)
    assert buckets == {"a": "head", "b": "torso", "c": "torso", "d": "unseen", "e": "unseen"}


def test_assign_label_buckets_absent_from_train_means_unseen() -> None:
    """A label missing from train_counts (count 0) is unseen, whatever its size."""
    buckets = assign_label_buckets({"rare": 5, "mid": 60}, {"mid": 9}, head_min=100, torso_min=40)
    assert buckets == {"rare": "unseen", "mid": "torso"}


def test_assign_label_buckets_unseen_overrides_size() -> None:
    """A dataset-wide head label absent from train is 'unseen', never 'head'."""
    buckets = assign_label_buckets({"big": 500}, {"big": 0}, head_min=100, torso_min=40)
    assert buckets["big"] == "unseen"


def test_assign_label_buckets_rejects_bad_thresholds_and_counts() -> None:
    """Non-positive or ordered-backwards thresholds and inconsistent counts fail."""
    with pytest.raises(ValueError, match="positive"):
        assign_label_buckets({"a": 10}, {"a": 5}, head_min=0, torso_min=40)
    with pytest.raises(ValueError, match="strictly greater"):
        assign_label_buckets({"a": 10}, {"a": 5}, head_min=40, torso_min=40)
    with pytest.raises(ValueError, match="more train examples"):
        assign_label_buckets({"a": 5}, {"a": 10}, head_min=100, torso_min=40)


def test_labels_by_bucket_groups_and_orders() -> None:
    """Groups in BUCKET_NAMES order, alphabetical labels within each bucket."""
    buckets = {"zeta": "tail", "alpha": "head", "mid": "unseen", "beta": "tail"}
    grouped = labels_by_bucket(buckets)
    assert list(grouped) == list(BUCKET_NAMES)
    assert grouped == {
        "head": ["alpha"],
        "torso": [],
        "tail": ["beta", "zeta"],
        "unseen": ["mid"],
    }


def test_labels_by_bucket_ignores_unknown_bucket_names() -> None:
    """A mapping with a bogus bucket name does not crash the grouping helper."""
    grouped = labels_by_bucket({"a": "head", "b": "galaxy"})
    assert grouped["head"] == ["a"] and all(grouped[name] == [] for name in ("torso", "tail"))
