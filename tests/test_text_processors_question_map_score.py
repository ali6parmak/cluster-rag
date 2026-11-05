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
        score_grid(paragraphs, questions, question_map, scoring="sum_g2")


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


def test_the_z_scored_arms_use_the_z_score() -> None:
    """CS-50: the z arms sum the z-scores (the prior over the standard error)."""
    question_map, paragraphs, questions = _tiny()
    rows = {row.word: row for row in question_map.table_rows(QUESTION)}
    expected = sum(max(row.z_score, 0.0) for row in rows.values())

    positive = score_grid(paragraphs, questions, question_map, scoring="sum_zscore_positive")
    # row 0 is the positive paragraph: every table word is in it
    assert positive[0, 0] == pytest.approx(expected)
    # the signed arm differs from the restricted one whenever a z is negative
    signed = score_grid(paragraphs, questions, question_map, scoring="sum_zscore")
    assert set(np.sign(signed[:, 0])).issubset({-1.0, 0.0, 1.0})
    with pytest.raises(ValueError, match="Unknown scoring function"):
        score_grid(paragraphs, questions, question_map, scoring="sum_g2")


def test_the_class_restriction_filters_the_words() -> None:
    """CS-51: a filtered word contributes nothing; a filtered-empty table floors."""
    paragraphs = ["alpha bravo", "alpha bravo", "alpha bravo", "delta echo"]
    questions = ["q1", "q2", "q3", "q1"]
    labels = [1, 1, 1, 0]
    question_map = build_question_map(paragraphs, questions, labels)
    # alpha sits in all three labelled tables -> generic; bravo too
    assert question_map.word_classes["alpha"] == "generic"
    grid = score_grid(
        ("alpha bravo", "delta echo"),
        ("q1",),
        question_map,
        scoring="sum_zscore_positive",
    )
    restricted = score_grid(
        ("alpha bravo", "delta echo"),
        ("q1",),
        question_map,
        scoring="sum_zscore_positive",
        restrict=("distinctive",),
    )
    # the full grid scores the paragraph from its generic words; the restricted
    # one has an empty table for q1 -> the row's floor (below everything)
    assert grid[0, 0] > 0.0
    assert restricted[0, 0] < 0.0
    with pytest.raises(ValueError, match="Unknown class restriction"):
        score_grid(("a",), ("q1",), question_map, restrict=("bogus",))


def test_the_pos_restriction_filters_the_words() -> None:
    """CS-52: the grammatical filter keeps only the requested classes."""
    paragraphs = ["alpha bravo", "alpha charlie", "delta echo", "delta foxtrot"]
    question_map = build_question_map(
        paragraphs,
        ["q"] * 4,
        [1, 1, 0, 0],
        word_pos={"alpha": "noun", "bravo": "noun", "charlie": "verb"},
    )
    grid_paragraphs = ("alpha bravo", "alpha charlie")
    nouns = score_grid(
        grid_paragraphs,
        ("q",),
        question_map,
        scoring="sum_zscore_positive",
        restrict_pos=("noun",),
    )
    verbs = score_grid(
        grid_paragraphs,
        ("q",),
        question_map,
        scoring="sum_zscore_positive",
        restrict_pos=("verb",),
    )
    # row 0 keeps its two nouns; its only word under the verb filter is absent
    assert nouns[0, 0] > 0.0
    assert verbs[0, 0] == 0.0
    # row 1 loses alpha (a noun) and keeps charlie (the verb)
    assert verbs[1, 0] > 0.0
    assert nouns[1, 0] < nouns[0, 0]
    with pytest.raises(ValueError, match="Unknown POS restriction"):
        score_grid(grid_paragraphs, ("q",), question_map, restrict_pos=("adjective",))


def test_the_union_is_the_two_halves_summed_cell_for_cell() -> None:
    """CS-53: hand-authored halves -> the union's arithmetic and its floor."""
    from cluster_rag.text_processors.lexical.question_map_score import (
        UNION_SCORING,
        union_grids,
    )

    word_grids = {
        "scorings": {
            PRIMARY_SCORING: np.array([[1.0, 0.0], [0.5, -0.5]]),
            "support_count": np.array([[2.0, 0.0], [1.0, 0.0]]),
        },
        "floor": np.array([False, True]),
    }
    phrase_grids = {
        "scorings": {
            PRIMARY_SCORING: np.array([[0.25, 0.0], [0.5, 0.25]]),
            "support_count": np.array([[1.0, 0.0], [0.0, 0.0]]),
        },
        "floor": np.array([False, False]),
    }
    union = union_grids(word_grids, phrase_grids)

    assert union["scoring"] == PRIMARY_SCORING
    assert union["union"].tolist() == [[1.25, 0.0], [1.0, -0.25]]
    # the halves are kept beside the union, unreordered
    assert union["word"].tolist() == word_grids["scorings"][PRIMARY_SCORING].tolist()
    assert union["phrase"].tolist() == phrase_grids["scorings"][PRIMARY_SCORING].tolist()
    # the union's floor is the union of the two halves' flags: the word map's
    # floor on row 1 survives even though the phrase map can speak there
    assert union["word_floor"].tolist() == [False, True]
    assert union["phrase_floor"].tolist() == [False, False]
    assert union["word_count"].tolist() == [[2.0, 0.0], [1.0, 0.0]]
    assert union["phrase_count"].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    # the union's name is the shared one (the UPR CS-39 composition)
    assert UNION_SCORING == "sum_zscore_positive_phrase"


def test_the_union_refuses_a_missing_arm_or_a_ragged_grid() -> None:
    """The composition fails loudly rather than summing mismatched cells."""
    from cluster_rag.text_processors.lexical.question_map_score import union_grids

    grids = {"scorings": {PRIMARY_SCORING: np.zeros((2, 2))}, "floor": np.zeros(2, dtype=bool)}
    with pytest.raises(ValueError, match="do not carry the scoring"):
        union_grids(grids, {"scorings": {}, "floor": np.zeros(2, dtype=bool)})
    ragged = {"scorings": {PRIMARY_SCORING: np.zeros((3, 2))}, "floor": np.zeros(2, dtype=bool)}
    with pytest.raises(ValueError, match="same grid"):
        union_grids(grids, ragged)
    with pytest.raises(ValueError, match="Unknown scoring function"):
        union_grids(grids, grids, scoring="sum_g2")


def test_the_union_on_the_tiny_maps_is_hand_computable() -> None:
    """CS-53: the word z arm plus the phrase z arm, both from their own tables."""
    from cluster_rag.text_processors.features.phrase_map import phrase_units
    from cluster_rag.text_processors.lexical.question_map_score import union_grids

    scoring = "sum_zscore_positive"  # the registered union's arm (CS-50's primary)
    question_map, paragraphs, questions = _tiny()
    phrase_map = build_question_map(
        ["alpha bravo", "delta echo", "alpha bravo"],
        [QUESTION, QUESTION, EMPTY],
        [1, 0, 0],
        units=phrase_units,
    )
    word_grid = score_grid(paragraphs, questions, question_map, scoring=scoring)
    phrase_grid = score_grid(
        paragraphs,
        questions,
        phrase_map,
        scoring=scoring,
        units=phrase_units,
    )
    union = union_grids(
        {
            "scorings": {scoring: word_grid, "support_count": np.zeros(word_grid.shape)},
            "floor": np.array([False, True]),
        },
        {
            "scorings": {scoring: phrase_grid, "support_count": np.zeros(phrase_grid.shape)},
            "floor": np.array([False, True]),
        },
        scoring=scoring,
    )
    # row 0 ("alpha bravo"): two word rows' z + the one bigram's z
    word_rows = {row.word: row for row in question_map.table_rows(QUESTION)}
    phrase_rows = {row.word: row for row in phrase_map.table_rows(QUESTION)}
    expected = (
        max(word_rows["alpha"].z_score, 0.0)
        + max(word_rows["bravo"].z_score, 0.0)
        + max(phrase_rows["alpha bravo"].z_score, 0.0)
    )
    assert union["union"][0, 0] == pytest.approx(expected)
    assert union["union"][0, 1] == pytest.approx(word_grid[0, 1] + phrase_grid[0, 1])
    # the phrase table keeps only the bigram, so the phrase half is smaller
    assert abs(phrase_grid[0, 0]) < abs(word_grid[0, 0])


def test_the_empty_table_fills_are_hand_computed() -> None:
    """CS-56: the floor, the neutral median and the max on the same empty cells."""
    from cluster_rag.text_processors.lexical.question_map_score import FILLS

    paragraphs = ["alpha bravo", "delta echo", "alpha charlie", "foxtrot golf"]
    questions = ["q1", "q1", "q2", "q2"]
    labels = [1, 0, 1, 0]
    question_map = build_question_map(paragraphs, questions, labels)
    grid_paragraphs = ("alpha bravo", "foxtrot golf")
    grid_questions = ("q1", "q2", EMPTY)

    def _fill(fill: str) -> np.ndarray:
        return score_grid(
            grid_paragraphs,
            grid_questions,
            question_map,
            scoring="sum_zscore_positive",
            fill=fill,
        )

    floor, neutral, optimistic = _fill("floor"), _fill("neutral"), _fill("max")
    assert set(FILLS) == {"floor", "neutral", "max"}
    # the non-empty cells are identical under the three fills
    assert np.array_equal(floor[:, :2], neutral[:, :2])
    assert np.array_equal(floor[:, :2], optimistic[:, :2])
    # row 0: the two seen columns' values
    s1, s2 = neutral[0, 0], neutral[0, 1]
    assert floor[0, 2] == pytest.approx(min(s1, s2) - FLOOR_GAP)
    assert neutral[0, 2] == pytest.approx((s1 + s2) / 2.0)
    assert optimistic[0, 2] == pytest.approx(max(s1, s2))
    # row 1 sees nothing in either table: 0 for the neutral/max, the gap for the floor
    assert neutral[1, 2] == 0.0
    assert optimistic[1, 2] == 0.0
    assert floor[1, 2] == pytest.approx(-FLOOR_GAP)
    with pytest.raises(ValueError, match="Unknown empty-table fill"):
        score_grid(grid_paragraphs, grid_questions, question_map, fill="dense")
