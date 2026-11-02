"""Unit tests for the corpus map's **stability** (the track's CS-36).

The hand-computed cases: the stability of a pair on a corpus of one text (every
share is 1.0), the presence share against an oracle that replays the same
bootstrap draws, the sign statistic (a pair whose sign flips is not a link),
the filter (only the surviving pairs contribute), the rule that a label whose
every pair is dropped carries the floor, and the cross-fit's per-fold filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import (
    FLOOR_GAP,
    cross_fit_scores,
    score_cells,
)
from cluster_rag.text_processors.lexical.stability import (
    HEAD_K,
    estimate_stability,
    stability_block,
)

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in


def _records() -> tuple[Record, ...]:
    return (
        Record(id="t1", text="alpha bravo", language="en", labels=["A"]),
        Record(id="t2", text="alpha charlie", language="en", labels=["A"]),
        Record(id="t3", text="delta echo", language="en", labels=["B"]),
        Record(id="t4", text="delta foxtrot", language="en", labels=["B"]),
    )


def _map(texts=None, label_sets=None):
    """The tiny corpus map: A's table {alpha, bravo, charlie}, B's {delta, echo, foxtrot}."""
    if texts is None:
        records = _records()
        texts = [record.text for record in records]
        label_sets = [record.labels for record in records]
    return build_corpus_map(texts, label_sets, LABELS)


def test_a_corpus_whose_every_text_carries_the_label_is_fully_stable() -> None:
    """Every resample holds the label's texts: every share is 1.0."""
    texts = ["alpha beta", "alpha beta", "alpha beta"]
    label_sets = [["A"], ["A"], ["A"]]
    base = _map(texts, label_sets)
    report = estimate_stability(texts, label_sets, base, resamples=8, seed=42)
    assert [pair.word for pair in report.pairs] == ["alpha", "beta"]
    for pair in report.pairs:
        assert pair.present_share == 1.0
        assert pair.sign_stability == 1.0
        assert pair.head_stability == 1.0
        assert pair.survives(1.0)


def test_the_presence_share_is_the_share_of_resamples_that_drew_the_pair() -> None:
    """The oracle replays the same bootstrap draws and counts them itself."""
    texts = ["alpha bravo", "alpha charlie", "delta echo"]
    label_sets = [["A"], ["A"], ["B"]]
    base = _map(texts, label_sets)
    resamples, seed = 64, 7
    report = estimate_stability(texts, label_sets, base, resamples=resamples, seed=seed)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(texts), size=len(texts)) for _ in range(resamples)]
    # 'alpha' sits in the first two texts, both of them with the label A
    expected = sum(1 for draw in draws if 0 in draw or 1 in draw)
    pair = next(pair for pair in report.pairs if pair.label == "A" and pair.word == "alpha")
    assert pair.present_share == pytest.approx(expected / resamples)


def test_a_pair_whose_sign_flips_is_not_a_stable_link() -> None:
    """'alpha' goes with A in the full map and against it in a resample of t1's."""
    texts = ["alpha beta", "alpha beta gamma", "alpha beta gamma delta"]
    label_sets = [["A"], ["B"], ["B"]]
    base = _map(texts, label_sets)
    report = estimate_stability(texts, label_sets, base, resamples=64, seed=3)
    pair = next(pair for pair in report.pairs if pair.label == "A" and pair.word == "alpha")
    assert pair.log_odds < 0.0  # the full map: the word goes against the label
    assert pair.present_share > pair.sign_stability  # the sign does not always hold
    assert pair.sign_stability <= pair.present_share


def test_the_estimate_is_deterministic_and_the_seed_moves_it() -> None:
    """The same seed gives the same report; another seed gives another one."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    base = _map(texts, label_sets)
    first = estimate_stability(texts, label_sets, base, resamples=16, seed=42)
    second = estimate_stability(texts, label_sets, base, resamples=16, seed=42)
    other = estimate_stability(texts, label_sets, base, resamples=16, seed=43)
    assert [pair.sign_stability for pair in first.pairs] == [
        pair.sign_stability for pair in second.pairs
    ]
    assert [pair.sign_stability for pair in first.pairs] != [
        pair.sign_stability for pair in other.pairs
    ]


def test_the_filter_keeps_only_the_surviving_pairs() -> None:
    """The survivors are the pairs whose sign reaches the threshold."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    base = _map(texts, label_sets)
    report = estimate_stability(texts, label_sets, base, resamples=32, seed=42)
    survivors = report.survivors(0.0)
    assert set(survivors) == {"A", "B"}
    assert set(survivors["A"]) == {"alpha", "bravo", "charlie"}
    # every label of the map is in the mapping, the empty ones included
    assert report.label_survivors(0.0)["Z"] == frozenset()
    assert report.surviving_pairs(0.0) == len(report.pairs)


def test_a_label_whose_every_pair_is_dropped_carries_the_floor() -> None:
    """No evidence left means the label is ranked last, exactly like Z."""
    corpus_map = _map()
    scores = score_cells(["alpha bravo"], corpus_map, scoring="support_count")
    # A keeps 'alpha', B loses every pair: B carries the row's floor, like Z
    filtered = score_cells(
        ["alpha bravo"],
        corpus_map,
        scoring="support_count",
        keep={"A": frozenset({"alpha"}), "B": frozenset(), "Z": frozenset()},
    )
    assert filtered[0, 0] == 1.0
    # the floor is the row's smallest SEEN score minus one (the filtered map's
    # own seen labels: A alone)
    assert filtered[0, 1] == pytest.approx(filtered[0, 0] - FLOOR_GAP)
    assert filtered[0, 2] == pytest.approx(filtered[0, 1])
    assert filtered[0, 1] < filtered[0, 0]
    assert np.isfinite(filtered).all()
    # and the unfiltered map gives B a score of its own (0.0), not the floor
    assert scores[0, 1] == 0.0


def test_the_unfiltered_path_is_unchanged_by_the_filter_machinery() -> None:
    """A filter that keeps everything scores exactly like no filter at all."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    base = _map(texts, label_sets)
    report = estimate_stability(texts, label_sets, base, resamples=8, seed=42)
    plain = score_cells(["alpha bravo", "delta echo"], base, scoring="sum_logodds")
    kept = score_cells(
        ["alpha bravo", "delta echo"],
        base,
        scoring="sum_logodds",
        keep=report.label_survivors(0.0),
    )
    assert np.array_equal(plain, kept)
    crossed_plain = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42)
    crossed_kept = cross_fit_scores(
        texts,
        label_sets,
        LABELS,
        folds=2,
        seed=42,
        stability_min=0.0,
        stability_resamples=8,
    )
    for arm, scores in crossed_plain.items():
        assert np.array_equal(scores, crossed_kept[arm])


def test_the_cross_fit_estimates_the_filter_on_each_fold() -> None:
    """The fold's own map decides the fold's filter (the fit side's guard)."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    crossed = cross_fit_scores(
        texts,
        label_sets,
        LABELS,
        folds=2,
        seed=42,
        stability_min=0.0,
        stability_resamples=8,
    )
    plain = cross_fit_scores(texts, label_sets, LABELS, folds=2, seed=42)
    assert crossed["sum_logodds"].shape == (4, 3)
    assert np.array_equal(crossed["sum_logodds"], plain["sum_logodds"])


def test_the_block_carries_the_readout_and_the_survival_curve() -> None:
    """The artifact's block: the knobs, the curve, the per-label and per-pair rows."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    base = _map(texts, label_sets)
    report = estimate_stability(texts, label_sets, base, resamples=8, seed=42, head_k=2)
    block = stability_block(report)
    assert block["available"] is True
    assert block["knobs"]["resamples"] == 8
    assert block["knobs"]["seed"] == 42
    assert block["knobs"]["head_k"] == 2
    curve = block["summary"]["survival_curve"]
    assert len(curve) == 21
    assert curve[0]["share"] == 1.0
    assert curve[-1]["share"] <= curve[0]["share"]
    assert {row["label"] for row in block["per_pair"]} == {"A", "B"}
    # the per-label readout covers the whole label space (Z has no table)
    assert {row["label"] for row in block["per_label"]} == {"A", "B", "Z"}
    assert block["survival_by_support"][0]["n_with"] == "1"


def test_the_head_size_is_the_head_size() -> None:
    """K = 10 is the registered head size (CS-35's top-10 convention)."""
    assert HEAD_K == 10


def test_a_degenerate_request_fails_before_anything_is_computed() -> None:
    """No resample, no corpus, mismatched lengths: all fail loudly."""
    texts = [record.text for record in _records()]
    label_sets = [record.labels for record in _records()]
    base = _map()
    with pytest.raises(ValueError, match="at least 1 resample"):
        estimate_stability(texts, label_sets, base, resamples=0)
    with pytest.raises(ValueError, match="non-empty corpus"):
        estimate_stability([], [], base, resamples=2)
    with pytest.raises(ValueError, match="label sets"):
        estimate_stability(texts, label_sets[:-1], base, resamples=2)


def test_the_predictor_applies_and_reports_the_filter() -> None:
    """The predictor's lineage carries the filter's own numbers."""
    context = PredictorContext(labels=LABELS, train_records=_records())
    predictor = build_predictor(
        "corpus_map",
        {"stability_min": 0.0, "stability_resamples": 8},
        context,
        seed=42,
    )
    stability = predictor.describe()["stability"]
    assert stability["enabled"] is True
    assert stability["threshold"] == 0.0
    assert stability["resamples"] == 8
    assert stability["n_surviving_pairs"] == stability["n_pairs"]
    # a filter that keeps everything scores like the full map
    plain = build_predictor("corpus_map", {}, context, seed=42)
    assert np.array_equal(predictor.score(["alpha bravo"]), plain.score(["alpha bravo"]))


def test_the_label_blind_counterpart_is_a_different_map() -> None:
    """The blind map carries no word↔label association: its scores differ."""
    context = PredictorContext(labels=LABELS, train_records=_records())
    plain = build_predictor("corpus_map", {}, context, seed=42)
    blind = build_predictor("corpus_map", {"label_blind": True}, context, seed=42)
    assert blind.describe()["label_blind"] is True
    assert not np.array_equal(blind.score(["alpha bravo"]), plain.score(["alpha bravo"]))


def test_the_fit_side_carries_the_filter_s_provenance() -> None:
    """The fit side says which filter it read (the cross-fit's own)."""
    context = PredictorContext(labels=LABELS, train_records=_records())
    predictor = build_predictor(
        "corpus_map",
        {"stability_min": 0.75, "stability_resamples": 8},
        context,
        seed=42,
    )
    fit_side = predictor.describe()["fit_side"]
    assert fit_side["stability_min"] == 0.75
    assert fit_side["stability_resamples"] == 8
    assert fit_side["selected_by"] == "out_of_fold_micro_f1"


def test_invalid_parameters_fail_before_the_fit() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    context = PredictorContext(labels=LABELS, train_records=_records())
    with pytest.raises(ValueError, match="stability_min"):
        build_predictor("corpus_map", {"stability_min": 2.0}, context, seed=42)
    with pytest.raises(ValueError, match="stability_min"):
        build_predictor("corpus_map", {"stability_min": "half"}, context, seed=42)
    with pytest.raises(ValueError, match="stability_resamples"):
        build_predictor("corpus_map", {"stability_resamples": 0}, context, seed=42)
    with pytest.raises(ValueError, match="label_blind"):
        build_predictor("corpus_map", {"label_blind": "yes"}, context, seed=42)


def test_the_floor_gap_is_the_one_the_scorer_uses() -> None:
    """The floor's gap is the scorer's own constant (no second definition)."""
    assert FLOOR_GAP == 1.0
