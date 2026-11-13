"""Hand-computed tests for the EURLEX-4K ranking metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.topic_classification.metrics import (
    evaluate_ranking,
    label_recall_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    per_document_recall_at_k,
    r_precision,
    ranked_labels,
    recall_at_k,
)


def test_ranked_labels_is_stable_on_ties() -> None:
    ranked = ranked_labels(np.array([[0.5, 0.5, 0.1]]))
    assert ranked.tolist() == [[0, 1, 2]]


def test_recall_and_r_precision_on_one_document() -> None:
    # top-1 = label 0 (gold), top-2 = labels 0 and 1 (1 is not gold)
    scores = np.array([[0.9, 0.8, 0.1, 0.0]])
    ranked = ranked_labels(scores)
    gold = [{0, 2}]
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(1.0)
    assert r_precision(ranked, gold) == pytest.approx(0.5)


def test_ndcg_at_two() -> None:
    scores = np.array([[0.9, 0.8, 0.1, 0.0]])
    ranked = ranked_labels(scores)
    gold = [{0, 2}]
    # DCG = 1/log2(2) + 0/log2(3) = 1.0
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    assert ndcg_at_k(ranked, gold, 2) == pytest.approx(1.0 / 1.6309297535714578)


def test_two_documents_are_averaged() -> None:
    scores = np.array(
        [
            [0.9, 0.8, 0.1],  # doc 0: gold {0}; top-1 hits
            [0.1, 0.9, 0.8],  # doc 1: gold {2}; top-1 misses, top-2 hits
        ]
    )
    ranked = ranked_labels(scores)
    gold = [{0}, {2}]
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 2) == pytest.approx(1.0)
    assert r_precision(ranked, gold) == pytest.approx(0.5)


def test_label_recall_by_bucket() -> None:
    scores = np.array([[0.9, 0.8, 0.1]])
    ranked = ranked_labels(scores)
    gold = [{0, 2}]
    buckets = {0: "head", 2: "tail"}
    result = label_recall_at_k(ranked, gold, buckets, 1)
    assert result["head"] == {"hits": 1, "labels": 1, "recall": 1.0}
    assert result["tail"] == {"hits": 0, "labels": 1, "recall": 0.0}


def test_evaluate_ranking_shape() -> None:
    scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.9]])
    result = evaluate_ranking(scores, [{0, 2}, {2, 3}], {0: "head", 2: "tail", 3: "head"})
    assert set(result) == {"documents", "labels", "primary", "secondary", "diagnostics"}
    assert result["primary"]["recall@50"] == 1.0
    assert result["primary"]["r_precision"] == 0.75
    assert result["diagnostics"]["coverage@50"] == 1.0


def test_per_document_recall_matches_the_macro_average() -> None:
    scores = np.array([[0.9, 0.8, 0.1], [0.1, 0.9, 0.8]])
    ranked = ranked_labels(scores)
    gold = [{0}, {2}]
    values = per_document_recall_at_k(ranked, gold, 1)
    assert values.tolist() == [1.0, 0.0]
    assert values.mean() == pytest.approx(recall_at_k(ranked, gold, 1))


def test_per_document_recall_validates_its_inputs() -> None:
    ranked = ranked_labels(np.array([[0.9, 0.1]]))
    with pytest.raises(ValueError):
        per_document_recall_at_k(ranked, [{0}], 0)
    with pytest.raises(ValueError):
        per_document_recall_at_k(ranked, [set()], 1)


def test_paired_bootstrap_ci_is_deterministic_and_centred() -> None:
    differences = np.array([0.02, 0.03, 0.01, 0.04, 0.0, 0.02, 0.03, 0.01])
    result = paired_bootstrap_ci(differences, n_resamples=500, seed=7)
    again = paired_bootstrap_ci(differences, n_resamples=500, seed=7)
    assert result == again
    assert result["mean"] == pytest.approx(0.02)
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["n_documents"] == 8
    assert result["n_resamples"] == 500


def test_paired_bootstrap_zero_difference_interval_contains_zero() -> None:
    result = paired_bootstrap_ci(np.zeros(20), n_resamples=200, seed=1)
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_paired_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([]))
