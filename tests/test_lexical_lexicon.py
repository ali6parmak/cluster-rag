"""Unit tests for the corpus map's estimator (the track's CS-31).

The hand-computed cases: the informative-prior log-odds and its z-score on a
tiny 2×2 table, the G² statistic and its p-value, the conditional smoothing,
and the map's own guarantee that CS-23's numbers are untouched.
"""

from __future__ import annotations

import math

import pytest

from cluster_rag.text_processors.features.corpus_map import build_corpus_map, corpus_map_block
from cluster_rag.text_processors.lexical.lexicon import (
    ALPHA,
    conditional_log_odds,
    g2_p_value,
    g2_statistic,
    pair_measures,
    prior_total,
)

LABELS = ("A", "B")


def _map():
    """A tiny map: A's two texts carry "alpha", B's two carry "beta"."""
    texts = ["alpha one", "alpha two", "beta three", "beta four"]
    label_sets = [["A"], ["A"], ["B"], ["B"]]
    return build_corpus_map(texts, label_sets, LABELS)


def test_the_informative_log_odds_and_its_z_score_are_hand_computable() -> None:
    """The table (a=3, b=1, c=2, d=4) with the reference's own formulas."""
    n_with, df, n_label, n_texts = 3, 4, 5, 10
    alpha0 = prior_total([4, 3, 2, 1])  # a vocabulary of four words
    measures = pair_measures(n_with, df, n_label, n_texts, alpha0=alpha0)

    prior = df + ALPHA
    delta_label = math.log((n_with + prior) / (n_label + alpha0 - n_with - prior))
    delta_other = math.log(
        (df - n_with + prior) / (n_texts - n_label + alpha0 - (df - n_with) - prior)
    )
    expected_delta = delta_label - delta_other
    expected_sigma = math.sqrt(1 / (n_with + prior) + 1 / (df - n_with + prior))

    assert measures.log_odds_informative == pytest.approx(expected_delta)
    assert measures.standard_error == pytest.approx(expected_sigma)
    assert measures.z_score == pytest.approx(expected_delta / expected_sigma)


def test_the_g2_statistic_and_its_p_value_are_hand_computable() -> None:
    """The likelihood-ratio G² on the same table, and its chi-square 1-df p-value."""
    # a=3, b=1, c=2, d=4 -> the expected counts under independence are 2, 2, 3, 3
    g2 = g2_statistic(3, 4, 5, 10)
    expected = 2 * (
        3 * math.log(3 / 2) + 1 * math.log(1 / 2) + 2 * math.log(2 / 3) + 4 * math.log(4 / 3)
    )
    assert g2 == pytest.approx(expected)
    assert g2_p_value(g2) == pytest.approx(math.erfc(math.sqrt(expected / 2)))
    assert g2_p_value(0.0) == 1.0


def test_the_conditional_smoothing_only_touches_the_zero_cells() -> None:
    """The 0.5 goes where a cell is zero, and nowhere else."""
    # no cell is zero -> the raw log-odds
    assert conditional_log_odds(3, 4, 5, 10) == pytest.approx(math.log(3 / 1) + math.log(4 / 2))
    # b is zero -> the b cell gets the 0.5 and the others do not
    # a=1, b=0, c=1, d=1
    assert conditional_log_odds(1, 1, 2, 3) == pytest.approx(math.log(1 / 0.5) + math.log(1 / 1))


def test_the_prior_total_is_the_vocabulary_sum() -> None:
    """α₀ is the sum of the words' priors over the map's vocabulary."""
    assert prior_total([4, 3, 2, 1]) == pytest.approx(
        (4 + ALPHA) + (3 + ALPHA) + (2 + ALPHA) + (1 + ALPHA)
    )


def test_the_map_keeps_cs23s_numbers_and_adds_the_new_measures() -> None:
    """CS-23's log-odds is untouched; the new measures sit beside it."""
    corpus_map = _map()
    row = corpus_map.topic_rows("A")[0]
    # "alpha" is in both of A's texts and in no other: a=2, b=0, c=0, d=2
    # the CS-23 log-odds: log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))
    assert row.word == "alpha"
    assert row.log_odds == pytest.approx(math.log(2.5 / 0.5) + math.log(2.5 / 0.5))
    assert row.n_with == 2 and row.df == 2
    # the new measures are there and finite
    assert math.isfinite(row.log_odds_informative)
    assert row.standard_error > 0
    assert math.isfinite(row.z_score)
    assert row.g2 >= 0 and 0.0 <= row.p_value <= 1.0
    assert math.isfinite(row.log_odds_conditional)


def test_the_block_carries_the_estimator_summary() -> None:
    """The block gains the summary the change set's falsification is read against."""
    block = corpus_map_block(_map())
    estimator = block["estimator"]
    assert estimator["head_agreement"]["n_labels_with_a_table"] == 2
    assert set(estimator["heads"]) == {
        "log_odds",
        "log_odds_informative",
        "z_score",
        "p_value",
    }
    for label in ("A", "B"):
        for measure in estimator["heads"].values():
            assert measure[label]["word"]
    assert estimator["conditional_smoothing"]["n_pairs"] == block["corpus"]["n_pairs"]
