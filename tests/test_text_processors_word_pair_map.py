"""Unit tests for the global word-pair table (the track's CS-59).

Hand-computed cases: the shared-word table's counts and estimator, the floor,
the two cross-fits (out-of-fold judged cells; a held-out question read from
the other questions' words), the blind's real-cell read, the arms and the
single-observation filter.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import DictionaryConfig
from cluster_rag.text_processors.features.word_pair_map import (
    WORD_PAIR_PRIMARY,
    build_word_pair_map,
    cross_fit_word_pair_questions,
    cross_fit_word_pair_seen,
    paragraph_fold_ids,
    question_fold_ids,
    score_word_pair_grid,
)
from cluster_rag.text_processors.lexical.lexicon import pair_measures, prior_total
from cluster_rag.text_processors.lexical.question_map_score import FLOOR_GAP

PARAGRAPHS = ["alpha bravo", "alpha bravo", "bravo echo", "delta echo"]
QUESTIONS = ["alpha", "alpha", "bravo", "bravo"]
LABELS = [1, 0, 1, 0]


def _table():
    return build_word_pair_map(PARAGRAPHS, QUESTIONS, LABELS)


def test_the_table_counts_only_the_shared_words() -> None:
    """df is the shared-pair count; paragraph_df the paragraph's own."""
    table = _table()
    assert table.n_pairs == 4
    assert table.n_positive == 2
    assert table.n_negative == 2
    assert set(table.rows) == {"alpha", "bravo"}
    alpha = table.rows["alpha"]
    assert (alpha.n_with, alpha.df, alpha.paragraph_df) == (1, 2, 2)
    bravo = table.rows["bravo"]
    # bravo is in pairs 0, 1 (paragraphs, not questions) and 2 (shared):
    # df = 1 (only pair 2 shares it), paragraph_df = 3.
    assert (bravo.n_with, bravo.df, bravo.paragraph_df) == (1, 1, 3)


def test_the_row_uses_the_cs50_estimator() -> None:
    """The row's z-score is ``pair_measures`` with the paragraph df as the prior."""
    table = _table()
    # The paragraph vocabulary: alpha 2, bravo 3, echo 2, delta 1.
    alpha0 = prior_total([2, 3, 2, 1])
    expected = pair_measures(1, 2, 2, 4, alpha0=alpha0, prior_df=2)
    assert table.rows["alpha"].z_score == pytest.approx(expected.z_score)
    assert table.rows["alpha"].log_odds_informative == pytest.approx(expected.log_odds_informative)


def test_the_floor_sits_below_every_seen_cell() -> None:
    """A pair with no known shared word carries its row's floor."""
    table = _table()
    grid = score_word_pair_grid(
        ("alpha bravo", "alpha echo"),
        ("alpha", "bravo"),
        table,
        fills=("floor", "neutral"),
    )
    floor = grid["scores"]["floor"]
    assert not grid["floored"][0, 0]
    assert not grid["floored"][0, 1]
    # row 1 ("alpha echo"): "alpha" is known, "bravo" is not shared -> floored
    assert grid["floored"][1, 1]
    assert floor[1, 1] == pytest.approx(floor[1, 0] - FLOOR_GAP)
    # the neutral fill is the row's median of the seen cells (one seen cell)
    assert grid["scores"]["neutral"][1, 1] == pytest.approx(grid["scores"]["neutral"][1, 0])


def test_the_support_count_arm_counts_the_known_shared_words() -> None:
    """The measure-free arm is the number of known shared words."""
    table = _table()
    grid = score_word_pair_grid(("alpha bravo",), ("alpha bravo",), table, scoring="support_count")
    # "alpha bravo" and "alpha bravo" share both table words
    assert grid["scores"]["floor"][0, 0] == 2.0
    assert grid["n_known"][0, 0] == 2


def test_the_single_observation_filter_drops_the_df_one_rows() -> None:
    """``min_df=2`` keeps only the rows resting on two or more pairs."""
    table = _table()
    assert table.rows["bravo"].df == 1
    filtered = build_word_pair_map(PARAGRAPHS, QUESTIONS, LABELS, min_df=2)
    assert set(filtered.rows) == {"alpha"}
    assert filtered.stats()["n_single_rows"] == 0


def test_the_pair_fold_cross_fit_scores_each_pair_out_of_fold() -> None:
    """A judged cell is the fold's table's cell, never the full table's."""
    paragraphs = ["alpha bravo", "alpha delta", "alpha bravo", "alpha delta"]
    questions = ["alpha bravo?", "alpha bravo?", "alpha delta?", "alpha delta?"]
    labels = [1, 0, 1, 0]
    fold_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    grids = cross_fit_word_pair_seen(
        paragraphs, questions, labels, fold_ids, scorings=(WORD_PAIR_PRIMARY,)
    )
    grid_paragraphs = ("alpha bravo", "alpha delta")
    grid_questions = ("alpha bravo?", "alpha delta?")
    fold_tables = {
        fold: build_word_pair_map(
            [paragraphs[index] for index in range(4) if int(fold_ids[index]) != fold],
            [questions[index] for index in range(4) if int(fold_ids[index]) != fold],
            [labels[index] for index in range(4) if int(fold_ids[index]) != fold],
        )
        for fold in (0, 1)
    }
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    assembled = grids["scorings"][WORD_PAIR_PRIMARY]["floor"]
    for pair, (row, column) in enumerate(cells):
        fold_grid = score_word_pair_grid(
            grid_paragraphs, grid_questions, fold_tables[int(fold_ids[pair])]
        )["scores"]["floor"]
        assert assembled[row, column] == pytest.approx(fold_grid[row, column])
    # the full table knows bravo (pair 0 is positive), the fold-0 table does
    # not (its training pairs 2 and 3 never share it): the assembled cell must
    # be the fold's, so it must differ from the full table's cell.
    full_table = build_word_pair_map(paragraphs, questions, labels)
    full_grid = score_word_pair_grid(grid_paragraphs, grid_questions, full_table)
    assert "bravo" in full_table.rows
    assert "bravo" not in fold_tables[0].rows
    assert assembled[0, 0] != pytest.approx(full_grid["scores"]["floor"][0, 0])


def test_the_question_folds_partition_the_questions() -> None:
    """Every question gets one fold; the seed reproduces the split."""
    questions = [f"q{index}" for index in range(7)]
    folds = question_fold_ids(questions, 3, seed=42)
    assert set(folds) == set(questions)
    assert sorted(folds.values()) == [0, 0, 0, 1, 1, 2, 2]
    assert question_fold_ids(questions, 3, seed=42) == folds
    assert question_fold_ids(questions, 3, seed=7) != folds


def test_a_held_out_question_reads_another_questions_words() -> None:
    """The held-out question's pair is scored when its shared word is known elsewhere."""
    paragraphs = ["alpha bravo", "alpha delta", "alpha bravo", "alpha delta"]
    questions = ["alpha bravo?", "alpha bravo?", "alpha delta?", "alpha delta?"]
    labels = [1, 0, 1, 0]
    folds = {"alpha bravo?": 0, "alpha delta?": 1}
    grids = cross_fit_word_pair_questions(
        paragraphs, questions, labels, folds, scorings=(WORD_PAIR_PRIMARY,)
    )
    # "alpha" is shared by both questions' pairs and is known in every fold;
    # "bravo" (question 1 only) and "delta" (question 2 only) are not shared
    # with the other fold's question, so they are silent.
    assert grids["n_shared"].tolist() == [2, 1, 1, 2]
    assert grids["n_known"].tolist() == [1, 1, 1, 1]
    assert not grids["floor"].any()


def test_the_blind_builds_from_the_permuted_questions_but_reads_the_real_cell() -> None:
    """The blind's table is scrambled; the judged cell is the real pair's."""
    # fold 0: pairs 0 (eval) and 1 (train); fold 1: pairs 2 and 3.
    paragraphs = ["alpha bravo", "alpha bravo", "bravo echo", "delta echo"]
    questions = ["alpha", "bravo", "bravo", "alpha"]
    labels = [1, 1, 1, 0]
    # pair 1 (train, fold 0) has the permuted question "bravo" and is positive:
    # the fold-0 blind table holds bravo. Pair 0's real question is "alpha"
    # (unknown to that table) but its permuted question is "bravo" (known), so
    # a real-cell read floors it and a permuted-cell read would not.
    permuted = ["bravo", "alpha", "alpha", "bravo"]
    grids = cross_fit_word_pair_seen(
        paragraphs,
        permuted,
        labels,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        eval_question_texts=questions,
        scorings=(WORD_PAIR_PRIMARY,),
    )
    assert grids["n_known"][0] == 0
    assert bool(grids["floor"][0])


def test_the_paragraph_folds_partition_the_paragraphs() -> None:
    """The paragraph split is its own stream and covers every paragraph once."""
    paragraphs = [f"paragraph number {index} alpha" for index in range(10)]
    folds = paragraph_fold_ids(paragraphs, 5, seed=42)
    assert set(folds) == set(paragraphs)
    assert sorted(folds.values()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert paragraph_fold_ids(paragraphs, 5, seed=42) == folds
    # the question split on the same texts is a different stream
    assert question_fold_ids(paragraphs, 5, seed=42) != folds
    assert paragraph_fold_ids(paragraphs, 5, seed=7) != folds


def test_a_held_out_paragraphs_unique_word_is_silent() -> None:
    """The paragraph holdout: a shared word carries the pair, a unique one does not."""
    paragraphs = ["alpha bravo", "alpha delta", "alpha echo", "bravo echo"]
    questions = ["alpha bravo", "alpha", "alpha", "bravo"]
    labels = [1, 0, 1, 0]
    # fold 0: paragraphs 0 and 1; fold 1: paragraphs 2 and 3
    grids = cross_fit_word_pair_seen(
        paragraphs,
        questions,
        labels,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        scorings=(WORD_PAIR_PRIMARY,),
    )
    # "alpha" is shared across the folds; "bravo" is unique to paragraph 0
    # (fold 0) and paragraph 3's training table reads it from paragraph 0.
    assert grids["n_shared"].tolist() == [2, 1, 1, 1]
    assert grids["n_known"].tolist() == [1, 1, 1, 1]
    assert not grids["floor"].any()


def test_the_arms_are_rejected_by_name() -> None:
    """An unknown arm or fill fails loudly."""
    table = _table()
    with pytest.raises(ValueError):
        score_word_pair_grid(("alpha",), ("alpha",), table, scoring="nope")
    with pytest.raises(ValueError):
        score_word_pair_grid(("alpha",), ("alpha",), table, fills=("nope",))


def test_the_word_pair_config_rejects_an_unknown_arm() -> None:
    """The config validator catches a typo before a run."""
    with pytest.raises(ValueError):
        DictionaryConfig(word_pairs=True, word_pair_primary="sum_nope")
    with pytest.raises(ValueError):
        DictionaryConfig(word_pairs=True, word_pair_scorings=("sum_nope",))
