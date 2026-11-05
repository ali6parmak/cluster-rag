"""Unit tests for the question map as a pair scorer (the track's CS-49).

The hand-computed cases: the four arms' scores on a tiny grid, the empty
table's floor, the cross-fit (a pair is scored by its own fold's map, never by
the full one), the floor flags and the label-blind permutation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cluster_rag.text_processors.features.question_map import build_question_map
from cluster_rag.text_processors.lexical.question_map_score import (
    FLOOR_GAP,
    PRIMARY_SCORING,
    blind_question_permutation,
    cross_fit_grids,
    score_grid,
)

QUESTION = "Were the cases documented?"
EMPTY = "A question with no positive?"

# The hand-computed log-odds (2 pairs for QUESTION: one positive, one
# negative; 1 positive of 2 pairs):
#   alpha: in the positive, in no negative -> log(1.5/0.5) + log(1.5/0.5)
LOG_ODDS_ALPHA = math.log(3.0) + math.log(3.0)


def _tiny():
    """Two questions: QUESTION (one positive, one negative) and EMPTY (none)."""
    paragraphs = ["alpha bravo", "delta echo", "alpha bravo"]
    questions = [QUESTION, QUESTION, EMPTY]
    labels = [1, 0, 0]
    question_map = build_question_map(paragraphs, questions, labels)
    grid_paragraphs = ("alpha bravo", "delta echo")
    grid_questions = (QUESTION, EMPTY)
    return question_map, grid_paragraphs, grid_questions


def test_the_four_arms_are_hand_computable() -> None:
    """The grid's scores, arm by arm, and the empty table's floor."""
    question_map, paragraphs, questions = _tiny()

    positive = score_grid(paragraphs, questions, question_map, scoring="sum_positive")
    assert positive.shape == (2, 2)
    # row 0 ("alpha bravo"): both words are in QUESTION's table with the same
    # hand-computed log-odds; the EMPTY column carries the row's floor
    assert positive[0, 0] == pytest.approx(2.0 * LOG_ODDS_ALPHA)
    assert positive[0, 1] == pytest.approx(2.0 * LOG_ODDS_ALPHA - FLOOR_GAP)
    # row 1 ("delta echo"): no word of it is in QUESTION's table -> 0
    assert positive[1, 0] == 0.0
    assert positive[1, 1] == pytest.approx(-FLOOR_GAP)

    log_odds = score_grid(paragraphs, questions, question_map, scoring="sum_logodds")
    assert log_odds[0, 0] == pytest.approx(2.0 * LOG_ODDS_ALPHA)
    assert log_odds[1, 0] == 0.0

    best = score_grid(paragraphs, questions, question_map, scoring="max_logodds")
    assert best[0, 0] == pytest.approx(LOG_ODDS_ALPHA)
    assert best[1, 0] == 0.0

    count = score_grid(paragraphs, questions, question_map, scoring="support_count")
    assert count[0, 0] == pytest.approx(2.0)
    assert count[1, 0] == 0.0
    with pytest.raises(ValueError, match="Unknown scoring function"):
        score_grid(paragraphs, questions, question_map, scoring="sum_zscore")


def test_the_empty_table_sits_below_every_seen_column() -> None:
    """The floor rule: a question with no table can never be ranked first."""
    question_map, paragraphs, questions = _tiny()
    grid = score_grid(paragraphs, questions, question_map, scoring="sum_positive")
    assert np.all(grid[:, 1] < grid[:, 0])
    assert grid[0, 1] == pytest.approx(grid[0, 0] - FLOOR_GAP)


def test_the_cross_fit_scores_each_pair_from_another_fold() -> None:
    """A judged pair is read by the map of the fold that did not contain it."""
    paragraphs = ["alpha bravo", "alpha charlie", "delta echo", "delta foxtrot"]
    questions = [QUESTION] * 4
    labels = [1, 1, 0, 0]
    # fold 0 = the two positives, fold 1 = the two negatives
    fold_ids = [0, 0, 1, 1]
    grids = cross_fit_grids(paragraphs, questions, labels, fold_ids)

    assert grids["paragraphs"] == tuple(dict.fromkeys(paragraphs))
    matrix = grids["scorings"][PRIMARY_SCORING]
    # the positives are read by the fold-1 map (built from the negatives only):
    # that map has no positive for QUESTION, so its table is empty -> the floor
    floor = matrix[0, 0]
    assert floor < 0.0
    assert grids["floor"].tolist() == [True, True, False, False]
    # the negatives are read by the fold-0 map (the positives): the table
    # exists, but their paragraphs share no word with it
    assert matrix[2, 0] == 0.0
    # the full map would have scored the positives well above the floor
    from cluster_rag.text_processors.features.question_map import build_question_map

    full = build_question_map(paragraphs, questions, labels)
    full_grid = score_grid(grids["paragraphs"], grids["questions"], full, scoring=PRIMARY_SCORING)
    assert full_grid[0, 0] > 0.0


def test_the_control_probe_records_the_per_fold_means() -> None:
    """The unjudged cells' score level per fold (the vintage sensitivity check)."""
    paragraphs = ["alpha bravo", "alpha charlie", "delta echo", "delta foxtrot"]
    questions = [QUESTION] * 4
    labels = [1, 1, 0, 0]
    grids = cross_fit_grids(paragraphs, questions, labels, [0, 0, 1, 1], probe_cells=[(1, 0)])
    probe = grids["probe"][PRIMARY_SCORING]
    assert set(probe["per_fold_mean"]) == {"0", "1"}
    assert probe["full_map_mean"] is not None


def test_the_blind_permutation_preserves_the_question_counts() -> None:
    """The label-blind counterpart keeps the per-question pair counts."""
    questions = [QUESTION] * 3 + [EMPTY] * 2
    permutation = blind_question_permutation(len(questions), seed=42)
    assert sorted(permutation.tolist()) == list(range(5))
    permuted = [questions[int(index)] for index in permutation]
    assert permuted.count(QUESTION) == 3
    assert permuted.count(EMPTY) == 2
    # deterministic, and independent of the cross-fit's own permutation
    assert np.array_equal(permutation, blind_question_permutation(len(questions), seed=42))
