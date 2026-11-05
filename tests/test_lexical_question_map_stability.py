"""Unit tests for the question map's stability (the track's CS-54).

The hand-computed cases: the statistic on a corpus that cannot change under a
resample (every share is 1.0), the survival curve / survivors / by-support
readout on a hand-authored report, the filter's effect on the scorer's grid
(a dropped link contributes nothing; an emptied table carries the floor), and
the per-fold filter (a judged pair is scored by its own fold's filter).
"""

from __future__ import annotations

import pytest

from cluster_rag.text_processors.features.question_map import build_question_map
from cluster_rag.text_processors.lexical.question_map_score import (
    cross_fit_grids,
    score_grid,
)
from cluster_rag.text_processors.lexical.question_map_stability import (
    QuestionLinkStability,
    QuestionMapStability,
    estimate_question_map_stability,
    head_retention,
    kept_share,
    stability_block,
    stability_filter,
)

QUESTION = "Were the cases documented?"
EMPTY = "A question with no positive?"


def test_the_identical_corpus_survives_every_resample() -> None:
    """Every pair identical and positive -> every resample is the same corpus."""
    paragraphs = ["alpha bravo"] * 4
    questions = [QUESTION] * 4
    labels = [1, 1, 1, 1]
    question_map = build_question_map(paragraphs, questions, labels)
    report = estimate_question_map_stability(
        paragraphs, questions, labels, question_map, resamples=5, seed=42
    )

    assert report.resamples == 5
    assert report.n_pairs == 4
    assert len(report.links) == 2  # alpha and bravo
    for link in report.links:
        assert link.present_share == 1.0
        assert link.sign_stability == 1.0
        assert link.head_stability == 1.0
    assert kept_share(report, 0.75) == 1.0
    assert kept_share(report, 0.99) == 1.0


def test_the_survival_readout_is_hand_computed() -> None:
    """A hand-authored report: the curve, the survivors, the by-support rows."""
    links = (
        QuestionLinkStability(QUESTION, "alpha", 2, 2, 3.0, 1.0, 1.0, 1.0, 0.9),
        QuestionLinkStability(QUESTION, "bravo", 1, 1, 2.0, 0.5, 0.9, 0.8, 0.5),
        QuestionLinkStability(QUESTION, "charlie", 1, 1, 1.0, 0.2, 0.6, 0.5, 0.2),
    )
    report = QuestionMapStability(
        resamples=10,
        seed=42,
        head_k=10,
        n_pairs=4,
        questions=(QUESTION, EMPTY),
        links=links,
    )

    # survivors at 0.75: alpha (1.0) and bravo (0.8); charlie (0.5) is out
    survivors = stability_filter(report, 0.75)
    assert survivors[QUESTION] == frozenset({"alpha", "bravo"})
    assert survivors[EMPTY] == frozenset()
    assert report.surviving_links(0.90) == 1
    assert kept_share(report, 0.75) == pytest.approx(2 / 3)
    # the survival curve is threshold-free: 0.00..1.00 in steps of 0.05
    curve = report.survival_curve()
    assert len(curve) == 21
    assert curve[0] == {"threshold": 0.0, "share": 1.0}
    assert next(row for row in curve if row["threshold"] == 0.75)["share"] == pytest.approx(2 / 3)
    assert next(row for row in curve if row["threshold"] == 0.90)["share"] == pytest.approx(1 / 3)
    # by support: the two single-pair links and the two-pair link
    by_support = {row["n_with"]: row for row in report.by_support()}
    assert set(by_support) == {1, 2}
    assert by_support[1]["links"] == 2
    assert by_support[1]["share_surviving_075"] == pytest.approx(0.5)  # bravo only
    assert by_support[2]["share_surviving_075"] == pytest.approx(1.0)
    # the head retention: bravo's head_stability is exactly the 0.5 threshold
    heads = head_retention(report)
    assert heads["head_links"] == 3
    assert heads["kept"] == 2  # alpha (0.9) and bravo (0.5)
    block = stability_block(report)
    assert block["n_links"] == 3
    assert block["resamples"] == 10
    # the measured single-pair row: bravo (0.8) and charlie (0.5)
    assert block["single_pair"]["n_with"] == 1
    assert block["single_pair"]["mean_sign_stability"] == pytest.approx(0.65)
    # the theory it is read against (the 1 - e^-1 cliff)
    assert block["prediction"]["single_pair_survival"] == pytest.approx(1.0 - 2.718281828459045**-1)


def test_a_filtered_link_contributes_nothing_and_an_emptied_question_floors() -> None:
    """The filter is the scorer's own `keep`: a dropped link is not counted."""
    paragraphs = ["alpha bravo", "delta echo", "alpha bravo"]
    questions = [QUESTION, QUESTION, EMPTY]
    labels = [1, 0, 0]
    question_map = build_question_map(paragraphs, questions, labels)
    grid_paragraphs = ("alpha bravo", "delta echo")
    grid_questions = (QUESTION, EMPTY)

    full = score_grid(grid_paragraphs, grid_questions, question_map, scoring="sum_zscore_positive")
    alpha_row = next(row for row in question_map.table_rows(QUESTION) if row.word == "alpha")
    kept = score_grid(
        grid_paragraphs,
        grid_questions,
        question_map,
        scoring="sum_zscore_positive",
        keep={QUESTION: frozenset({"alpha"})},
    )
    assert kept[0, 0] == pytest.approx(max(alpha_row.z_score, 0.0))
    assert kept[0, 0] < full[0, 0]  # bravo's evidence is gone
    # every link dropped: the question has no table left -> the row's floor
    floored = score_grid(
        grid_paragraphs,
        grid_questions,
        question_map,
        scoring="sum_zscore_positive",
        keep={},
    )
    assert floored[0, 0] < 0.0


def test_the_fold_filter_is_the_folds_own() -> None:
    """A judged pair is scored by the filter of the fold that did not hold it."""
    paragraphs = ["alpha bravo", "alpha charlie", "alpha delta", "alpha echo"]
    questions = [QUESTION] * 4
    labels = [1, 1, 0, 0]
    folds = [0, 0, 1, 1]  # fold 1 = the positives, so fold 1's map scores the negatives
    every = {QUESTION: frozenset({"alpha", "bravo", "charlie"})}

    full = cross_fit_grids(paragraphs, questions, labels, folds)
    unfiltered = full["scorings"]["sum_zscore_positive"]
    # the negatives carry alpha, so the fold-1 map scores them positively
    assert unfiltered[2, 0] > 0.0

    # keep alpha everywhere but in fold 1's filter: the negatives lose it
    filtered = cross_fit_grids(
        paragraphs,
        questions,
        labels,
        folds,
        keep=every,
        keep_by_fold={0: every, 1: {QUESTION: frozenset({"bravo"})}},
    )
    assert filtered["scorings"]["sum_zscore_positive"][2, 0] == 0.0

    # drop everything in fold 1: the pairs it scores carry the floor
    emptied = cross_fit_grids(
        paragraphs,
        questions,
        labels,
        folds,
        keep=every,
        keep_by_fold={0: every, 1: {}},
    )
    # the positives (fold 0) are scored by a map built from the negatives only,
    # which has no positive pair for the question -> those two already floor;
    # emptying fold 1's filter floors the two negatives as well
    unfiltered_floors = full["floor"].tolist()
    assert unfiltered_floors == [True, True, False, False]
    assert emptied["floor"].tolist() == [True, True, True, True]
    assert emptied["scorings"]["sum_zscore_positive"][2, 0] < 0.0
