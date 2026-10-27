"""Tests for the UPR rerank probe (the track's CS-15).

Contracts under test:

- the **candidate set**: every text's top-k columns, in the incumbent's own
  order;
- the **label side**: the catalog's description, the label's own name, and
  the **label-blind counterpart** (the deterministic cyclic shift);
- the **rank fusion**: the candidates are re-ordered by the reranker and the
  cells outside the set keep the incumbent's order below them;
- the **ceiling**: the gold cells inside the set are the true positives a
  perfect reordering could reach, and the micro-F1 follows the top-k rule's
  accounting;
- the **paired comparison**: the resampling unit is the text, and the
  difference is the arm's minus the incumbent's.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.text_processors.rerank.upr_rerank import (
    LABEL_DOCUMENT_DESCRIPTION,
    LABEL_DOCUMENT_NAME,
    ORDER_LABEL_FIRST,
    ORDER_TEXT_FIRST,
    candidate_ceiling,
    candidate_columns,
    fuse_candidates,
    label_documents,
    measure_reranked_arm,
    rerank_matrix,
    shifted_documents,
)
from cluster_rag.upr_rerank import build_candidate_requests, paired_arm_comparison

DECISION = {"rule": "top_k", "k": 2}


def test_the_candidate_columns_are_the_incumbents_own_top_k() -> None:
    """The candidate set is the incumbent's own top-k, in its order."""
    scores = np.asarray([[0.1, 0.9, 0.5, 0.2], [0.4, 0.3, 0.2, 0.1]])
    columns = candidate_columns(scores, 2)
    assert columns == [(1, 2), (0, 1)]


def test_the_candidate_columns_clip_to_the_label_count() -> None:
    """A k larger than the label space is clipped, never an error."""
    columns = candidate_columns(np.asarray([[0.1, 0.9]]), 10)
    assert columns == [(1, 0)]


def test_the_label_documents_are_the_description_or_the_name() -> None:
    """The label side is the catalog's description, with the name as the variant."""
    labels = ["Detention", "General"]
    descriptions = {"Detention": "the arbitrary detention of a person"}
    assert label_documents(labels, descriptions, LABEL_DOCUMENT_DESCRIPTION) == [
        "the arbitrary detention of a person",
        "General",  # no description: the name is the fallback
    ]
    assert label_documents(labels, descriptions, LABEL_DOCUMENT_NAME) == labels


def test_the_label_documents_reject_an_unknown_variant() -> None:
    """The variant is a config value, not a free string."""
    with pytest.raises(ValueError, match="unknown label document variant"):
        label_documents(["a"], None, "paragraph")


def test_the_shifted_documents_are_the_cyclic_shift() -> None:
    """The label-blind counterpart: the same list, shifted by one."""
    assert shifted_documents(["a", "b", "c"]) == ["b", "c", "a"]
    assert shifted_documents([]) == []


def test_the_rerank_matrix_marks_the_candidates() -> None:
    """The cells outside the candidate set are NaN (the fusion's marker)."""
    matrix = rerank_matrix([(0, 2)], [[0.5, 0.7]], 3)
    assert matrix[0, 0] == pytest.approx(0.5)
    assert matrix[0, 2] == pytest.approx(0.7)
    assert np.isnan(matrix[0, 1])


def test_the_fusion_reorders_the_candidates_and_keeps_the_rest_below() -> None:
    """The candidates take the top ranks; the rest keep the incumbent's order."""
    incumbent = np.asarray([[0.9, 0.8, 0.7, 0.6]])
    fused = fuse_candidates(incumbent, [(0, 1)], [[0.1, 0.5]])
    # the reranker prefers the second candidate: it now ranks first
    assert fused[0, 1] > fused[0, 0]
    # the cells outside the set keep the incumbent's order, below the set
    assert fused[0, 2] > fused[0, 3]
    assert fused[0, 0] > fused[0, 2]


def test_the_ceiling_counts_the_gold_cells_inside_the_set() -> None:
    """The ceiling: the gold cells inside the set are the reachable true positives."""
    gold = np.asarray([[True, False, False], [False, True, False]])
    columns = [(0, 1), (1, 2)]
    block = candidate_ceiling(columns, gold, n_predicted=4)
    assert block["n_gold_cells"] == 2
    assert block["n_gold_cells_in_set"] == 2
    assert block["cell_level_share"] == pytest.approx(1.0)
    # TP = 2, FP = 4 - 2 = 2, FN = 2 - 2 = 0 -> F1 = 4 / (4 + 2 + 0)
    assert block["micro_f1_ceiling"] == pytest.approx(4 / 6)


def test_the_ceiling_of_a_set_without_the_gold_is_zero() -> None:
    """A set that misses the gold cell cannot reach it."""
    gold = np.asarray([[False, True]])
    block = candidate_ceiling([(0,)], gold, n_predicted=2)
    assert block["n_gold_cells_in_set"] == 0
    assert block["micro_f1_ceiling"] == pytest.approx(0.0)


def test_the_ceiling_reports_the_buckets() -> None:
    """The per-bucket coverage: the unseen bucket's ceiling is stated."""
    gold = np.asarray([[True, False], [False, True]])
    block = candidate_ceiling(
        [(0,), (1,)],
        gold,
        n_predicted=2,
        bucket_masks={"seen": np.asarray([True, False]), "unseen": np.asarray([False, True])},
    )
    assert block["per_bucket"]["seen"]["n_gold_cells_in_set"] == 1
    assert block["per_bucket"]["unseen"]["cell_level_share"] == pytest.approx(1.0)


def test_the_measurement_reads_the_fused_order() -> None:
    """The re-ordered arm's measurement: the harness's own metrics."""
    gold = np.asarray([[True, False, False]])
    # the fused matrix holds the negative final rank: the best cell is 0
    fused = np.asarray([[0.0, -1.0, -2.0]])
    measurement = measure_reranked_arm(fused, gold, decision=DECISION, ks=(1, 2), shortlist_ms=(1,))
    # the top-2 rule predicts two labels: one hit, one false positive
    assert measurement["decision"]["micro_f1"] == pytest.approx(2 / 3)
    assert measurement["ranking"]["recall@1"] == pytest.approx(1.0)
    # the fused matrix holds ranks, not probabilities
    assert measurement["calibration"]["applicable"] is False


def test_the_pair_requests_carry_the_requested_order() -> None:
    """The order variant decides which side is the query."""
    texts = ["the text"]
    documents = [["the label"]]
    text_first = build_candidate_requests(
        texts, documents, kind="cross_encoder", order=ORDER_TEXT_FIRST
    )
    label_first = build_candidate_requests(
        texts, documents, kind="cross_encoder", order=ORDER_LABEL_FIRST
    )
    assert text_first[0].query == "the text"
    assert text_first[0].documents == ("the label",)
    assert label_first[0].query == "the label"
    assert label_first[0].documents == ("the text",)


def test_the_listwise_requests_take_the_text_as_the_query() -> None:
    """The listwise family: one query, the whole candidate list."""
    requests = build_candidate_requests(
        ["the text"], [["a", "b"]], kind="listwise", order=ORDER_TEXT_FIRST
    )
    assert len(requests) == 1
    assert requests[0].query == "the text"
    assert requests[0].documents == ("a", "b")


def test_the_listwise_family_has_no_reverse_order() -> None:
    """Its API admits one query and many documents, so the reverse does not exist."""
    with pytest.raises(ValueError, match="one query and many documents"):
        build_candidate_requests(["the text"], [["a"]], kind="listwise", order=ORDER_LABEL_FIRST)


def test_the_paired_comparison_is_paired_over_the_texts() -> None:
    """The difference is the arm's minus the incumbent's, on the same texts."""
    gold = np.asarray([[True, False, False, False], [False, True, False, False]])
    incumbent = np.asarray([[0.9, 0.05, 0.04, 0.0], [0.2, 0.8, 0.05, 0.0]])
    fused = np.asarray([[0.01, 0.9, 0.04, 0.0], [0.2, 0.8, 0.05, 0.0]])
    outcome = paired_arm_comparison(
        fused,
        incumbent,
        gold,
        ["a", "b", "c", "d"],
        {"seen": ["a", "b", "c", "d"]},
        decision=DECISION,
        ks=(1,),
        resamples=50,
        seed=42,
    )
    pooled = outcome["pooled"]["micro_f1"]
    # the first text's top-1 flips to the wrong label: the arm is worse
    assert pooled["delta"] < 0
    assert pooled["left"] < pooled["right"]
    assert "seen" in outcome["per_bucket"]
