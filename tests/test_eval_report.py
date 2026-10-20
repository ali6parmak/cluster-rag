"""Hand-computed tests for the per-bucket report assembly (eval/report.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cluster_rag.eval.report import build_label_report

LABELS = ["a", "b", "c", "d"]
BUCKETS = {"a": "head", "d": "head", "b": "unseen", "c": "tail"}

# Row 0 gold {a, d} (both head), row 1 gold {b} (unseen); c is never gold.
SCORES = np.array(
    [
        [0.9, 0.1, 0.8, 0.3],
        [0.2, 0.9, 0.7, 0.6],
    ]
)
GOLD = np.array(
    [
        [True, False, False, True],
        [False, True, False, False],
    ]
)
PREDICTED = np.array(
    [
        [True, False, True, False],  # top-2 of row 0 plus one extra (c) for F1/coverage
        [False, True, False, False],
    ]
)


def _report() -> dict:
    return build_label_report(
        SCORES, GOLD, LABELS, BUCKETS, PREDICTED, ks=(1, 2, 10), shortlist_ms=(2, 20)
    )


def test_report_overall_hand_computed() -> None:
    """The unrestricted section computes every metric family on the 2x4 case."""
    overall = _report()["overall"]
    assert overall["bucket"] == "overall"
    assert overall["n_labels"] == 4 and overall["n_labels_with_gold"] == 3
    assert overall["n_records"] == 2 and overall["n_rows_with_gold"] == 2
    assert overall["n_gold_cells"] == 3
    assert overall["f1"]["n_predicted_cells"] == 3
    assert overall["f1"]["n_labels_in_macro"] == 3 and overall["f1"]["n_labels_without_gold"] == 1

    ranking = overall["ranking"]
    assert ranking["precision@1"] == pytest.approx(1.0)
    assert ranking["recall@1"] == pytest.approx(0.75)
    assert ranking["precision@2"] == pytest.approx(0.5)
    assert ranking["recall@2"] == pytest.approx(0.75)
    # row 0: gold at ranks 1 and 3 of 4 -> DCG = 1 + 1/log2(4), IDCG = 1 + 1/log2(3)
    row0_ndcg10 = 1.5 / (1.0 + 1.0 / np.log2(3.0))
    assert ranking["ndcg@10"] == pytest.approx((row0_ndcg10 + 1.0) / 2.0)

    f1 = overall["f1"]
    assert f1["micro_precision"] == pytest.approx(2 / 3)
    assert f1["micro_recall"] == pytest.approx(2 / 3)
    assert f1["micro_f1"] == pytest.approx(2 / 3)
    assert f1["macro_f1"] == pytest.approx(2 / 3)  # per-label F1s: 1, 1, n/a, 0
    assert f1["sample_f1"] == pytest.approx(0.75)  # rows: 0.5 and 1.0

    assert overall["coverage"]["coverage"] == pytest.approx(2 / 3)
    assert overall["coverage"]["n_labels_suggested"] == 3

    shortlist = overall["shortlist"]
    assert shortlist["recall@2"] == pytest.approx(0.75)  # row0 1/2, row1 1/1
    assert shortlist["recall@20"] == pytest.approx(1.0)  # m clips to 4 columns


def test_report_per_bucket_restricted_views() -> None:
    """Each bucket section masks the columns; missing buckets have no section."""
    report = _report()
    bucket_names = set(report["per_bucket"])
    assert bucket_names == {"head", "tail", "unseen"}  # torso has no labels here

    head = report["per_bucket"]["head"]
    assert head["n_labels"] == 2 and head["n_rows_with_gold"] == 1
    assert head["n_gold_cells"] == 2
    assert head["ranking"]["precision@1"] == pytest.approx(1.0)
    assert head["ranking"]["recall@1"] == pytest.approx(0.5)
    assert head["ranking"]["precision@2"] == pytest.approx(1.0)  # k clips to the view size
    assert head["f1"]["micro_f1"] == pytest.approx(2 / 3)
    assert head["f1"]["macro_f1"] == pytest.approx(0.5)  # per-label F1s: 1 (a), 0 (d)
    assert head["coverage"]["coverage"] == pytest.approx(0.5)
    assert head["shortlist"]["recall@2"] == pytest.approx(0.5)  # gold d ranks 3rd globally

    unseen = report["per_bucket"]["unseen"]
    assert unseen["n_labels"] == 1 and unseen["n_gold_cells"] == 1
    assert unseen["ranking"]["precision@1"] == pytest.approx(1.0)
    assert unseen["ranking"]["recall@1"] == pytest.approx(1.0)
    assert unseen["f1"]["macro_f1"] == pytest.approx(1.0)
    assert unseen["coverage"]["coverage"] == pytest.approx(1.0)
    assert unseen["shortlist"]["recall@2"] == pytest.approx(1.0)

    tail = report["per_bucket"]["tail"]
    assert tail["n_rows_without_gold"] == 2 and tail["n_rows_with_gold"] == 0
    assert tail["ranking"]["precision@1"] == 0.0
    assert tail["f1"]["micro_f1"] == 0.0 and tail["f1"]["macro_f1"] == 0.0
    assert tail["f1"]["n_labels_in_macro"] == 0
    assert tail["coverage"]["coverage"] == 0.0
    assert tail["coverage"]["n_labels_suggested"] == 1  # c is suggested on row 0 anyway
    assert tail["shortlist"]["recall@2"] == 0.0 and tail["n_gold_cells"] == 0


def test_report_is_json_serializable() -> None:
    """The flat-integer/float/nested-dict payload dumps without conversion."""
    payload = json.dumps(_report(), indent=2)
    assert '"n_records"' in payload


def test_report_overall_matches_unmasked_direct_calls() -> None:
    """overall is exactly the single-mask code path (mask = None)."""
    from cluster_rag.eval.multilabel import (
        decision_f1_metrics,
        label_coverage,
        ranking_metrics,
        shortlist_recall,
    )

    report = _report()["overall"]
    direct_ranking = ranking_metrics(SCORES, GOLD, ks=(1, 2, 10))
    direct_f1 = decision_f1_metrics(PREDICTED, GOLD)
    for key, value in direct_ranking.items():
        if key.startswith("n_"):
            continue
        assert report["ranking"][key] == pytest.approx(value)
    for key in ("micro_f1", "macro_f1", "sample_f1"):
        assert report["f1"][key] == pytest.approx(direct_f1[key])
    assert report["shortlist"]["recall@20"] == pytest.approx(
        shortlist_recall(SCORES, GOLD, ms=(20,))["recall@20"]
    )
    assert report["coverage"]["coverage"] == pytest.approx(
        label_coverage(PREDICTED, GOLD)["coverage"]
    )


def test_report_validates_labels_and_bucket_names() -> None:
    """Mismatched label lists and malformed bucket mappings fail loudly."""
    with pytest.raises(ValueError, match="must match"):
        build_label_report(
            SCORES, GOLD, ["a", "b", "c"], BUCKETS, PREDICTED, ks=(1,), shortlist_ms=(2,)
        )
    with pytest.raises(ValueError, match="unique"):
        build_label_report(
            SCORES, GOLD, ["a", "a", "c", "d"], BUCKETS, PREDICTED, ks=(1,), shortlist_ms=(2,)
        )
    incomplete = {label: bucket for label, bucket in BUCKETS.items() if label != "d"}
    with pytest.raises(ValueError, match="missing from label_buckets"):
        build_label_report(SCORES, GOLD, LABELS, incomplete, PREDICTED, ks=(1,), shortlist_ms=(2,))
    bogus = {**BUCKETS, "d": "galaxy"}
    with pytest.raises(ValueError, match="unknown bucket name"):
        build_label_report(SCORES, GOLD, LABELS, bogus, PREDICTED, ks=(1,), shortlist_ms=(2,))
