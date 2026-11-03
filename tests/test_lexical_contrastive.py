"""Tests for the **contrastive map** (the track's CS-40).

The contrast is the map's **read** of two tables against each other: the
words that discriminate a pair of labels. The tests are about its arithmetic
(the silence convention, the difference, the flags) and about the pieces the
measurement reads (the map's own score, the runner-up, the focus rule and the
novelty measures).
"""

from __future__ import annotations

import pytest

from cluster_rag.text_processors.lexical.contrastive import (
    CONTRAST_TOP_K,
    DISPLAY_TOP_K,
    NOVELTY_BAR,
    RELEVANCE_BAR,
    VOCABULARY_BAR,
    contrast_rows,
    corpus_contrast_rows,
    default_contrast_label,
    focus_label,
    map_score,
    novelty_share,
    relevance_share,
    shown_pairs,
    text_contrast_rows,
    vocabulary_share,
)


def row(
    word: str,
    log_odds: float,
    *,
    n_with: int = 1,
    df: int = 1,
    z_score: float = 0.0,
    word_class: str = "distinctive",
    is_name_word: bool = False,
) -> dict[str, object]:
    """One table row in the artifact's own shape."""
    return {
        "word": word,
        "n_with": n_with,
        "df": df,
        "lift": 1.0,
        "log_odds": log_odds,
        "z_score": z_score,
        "word_class": word_class,
        "is_name_word": is_name_word,
    }


# --- the contrast's arithmetic ---


def test_the_contrast_is_the_difference_of_the_two_log_odds() -> None:
    """A word in both tables: the difference of the two log-odds."""
    rows = contrast_rows(
        [row("torture", 5.0), row("shared", 1.0)],
        [row("torture", -2.0), row("shared", 0.5)],
    )
    by_word = {entry.word: entry for entry in rows}
    assert by_word["torture"].contrast == pytest.approx(7.0)
    assert by_word["torture"].log_odds_a == pytest.approx(5.0)
    assert by_word["torture"].log_odds_b == pytest.approx(-2.0)
    assert by_word["shared"].contrast == pytest.approx(0.5)
    # the union holds every word of the two tables
    assert set(by_word) == {"torture", "shared"}


def test_a_missing_pair_contributes_nothing() -> None:
    """The map's silence convention: a pair that is not in the table is 0."""
    rows = contrast_rows([row("torture", 5.0)], [row("gdp", 2.0)])
    by_word = {entry.word: entry for entry in rows}
    assert by_word["torture"].log_odds_b == pytest.approx(0.0)
    assert by_word["torture"].contrast == pytest.approx(5.0)
    assert by_word["torture"].in_b is False
    assert by_word["gdp"].log_odds_a == pytest.approx(0.0)
    assert by_word["gdp"].contrast == pytest.approx(-2.0)


def test_the_flags_say_which_table_holds_the_word() -> None:
    """The flags: in_a, in_b, in_text and the sign disagreement."""
    rows = contrast_rows(
        [row("torture", 5.0), row("only_a", 1.0)],
        [row("torture", -2.0), row("only_b", 3.0)],
        text_words=["torture"],
    )
    by_word = {entry.word: entry for entry in rows}
    assert by_word["torture"].in_a and by_word["torture"].in_b
    assert by_word["torture"].in_text
    assert by_word["torture"].sign_disagreement
    assert not by_word["only_a"].sign_disagreement
    assert not by_word["only_b"].in_a
    assert by_word["only_b"].sign_disagreement is False


def test_the_rows_are_ranked_by_the_size_of_the_contrast() -> None:
    """The strongest difference first, the tie-break is the word's own order."""
    rows = contrast_rows(
        [row("small", 1.0), row("big", 4.0)],
        [row("small", 0.5), row("big", 0.0)],
    )
    assert [entry.word for entry in rows] == ["big", "small"]


# --- the two halves ---


def test_the_corpus_half_shows_the_top_k_in_each_direction() -> None:
    """The corpus's own half: the top-k of each direction, never a null word."""
    a_rows = [row(f"up{i}", float(i)) for i in range(1, 15)]
    # the words that go with B: they pull B up, so the contrast is negative
    b_rows = [row(f"down{i}", float(i)) for i in range(1, 15)]
    rows = corpus_contrast_rows(a_rows, b_rows, top_k=3)
    assert len(rows) == 6
    # the strongest difference in each direction (the largest log-odds)
    assert {entry.word for entry in rows} == {
        "up14",
        "up13",
        "up12",
        "down14",
        "down13",
        "down12",
    }
    # the default is the map's own head size
    assert CONTRAST_TOP_K == 10


def test_a_word_with_no_contrast_is_not_shown() -> None:
    """A word that says nothing about the pair is not a discriminator."""
    rows = corpus_contrast_rows([row("same", 1.0)], [row("same", 1.0)])
    assert rows == ()


def test_the_text_half_is_the_texts_own_words() -> None:
    """The text's half: the text's words that are in either table."""
    rows = text_contrast_rows(
        [row("torture", 5.0), row("other", 1.0)],
        [row("gdp", 2.0)],
        text_words=["torture", "gdp", "nowhere"],
    )
    assert [entry.word for entry in rows] == ["torture", "gdp"]
    assert all(entry.in_text for entry in rows)


# --- the map's own voice ---


def test_the_map_score_is_the_sum_of_the_positive_z_scores() -> None:
    """The map's own arm (sum_zscore_positive), read from the table."""
    rows = [
        row("a", 1.0, z_score=2.0),
        row("b", 1.0, z_score=-3.0),
        row("c", 1.0, z_score=1.5),
        row("d", 1.0, z_score=4.0),
    ]
    # the text has the words a, b and c: the positive z-scores sum to 3.5
    assert map_score(rows, ["a", "b", "c"]) == pytest.approx(3.5)
    assert map_score(rows, ["nothing"]) == pytest.approx(0.0)


def test_the_runner_up_is_the_maps_own_best_non_gold_label() -> None:
    """The map's runner-up among the labels the text does not carry."""
    tables = {
        "A": [row("a", 1.0, z_score=5.0)],
        "B": [row("shared", 1.0, z_score=3.0)],
        "C": [row("shared", 1.0, z_score=1.0)],
        "D": [],
    }
    # the text's word is in B's and C's tables: B's score is the larger one
    assert default_contrast_label(tables, ["shared"], exclude=["A"], labels=list("ABCD")) == "B"
    # an empty table carries no evidence: it never wins
    assert default_contrast_label(tables, ["nothing"], labels=list("ABCD")) == ""


def test_the_runner_up_breaks_the_tie_by_the_label_spaces_own_order() -> None:
    """The tie-break is the label space's own order, so the reading is stable."""
    tables = {
        "B": [row("shared", 1.0, z_score=2.0)],
        "C": [row("shared", 1.0, z_score=2.0)],
    }
    assert default_contrast_label(tables, ["shared"], labels=["C", "B"]) == "C"
    assert default_contrast_label(tables, ["shared"], labels=["B", "C"]) == "B"


def test_the_focus_is_the_gold_label_the_text_carries_the_most() -> None:
    """The focus rule: the gold label whose own words the text carries most."""
    text = "Ensure the elimination of all forms of racial discrimination"
    labels = ["Anti-racism", "Discrimination", "Other"]
    assert focus_label(text, ["Discrimination", "Anti-racism"], labels) == "Discrimination"
    # a text with no gold label of the space falls back to the space's first
    assert focus_label(text, ["Not a label"], labels) == "Anti-racism"


# --- the novelty measures ---


def test_the_shown_pairs_are_the_three_places_the_screen_prints() -> None:
    """The screen prints a label's log-odds in three places, and no more."""
    tables = {
        "A": [row("first", 1.0), row("second", 1.0), row("third", 1.0)],
    }
    reverse = {"second": [{"label": "A", "log_odds": 1.0}]}
    shown = shown_pairs(
        tables,
        reverse,
        "A",
        text_words=["second", "elsewhere"],
        top_k=2,
    )
    # the top-2 rows of the table
    assert "first" in shown and "second" in shown
    assert "third" not in shown
    # the text's word in the table
    assert "second" in shown
    # the text's word and the labels it leans to
    reverse = {"elsewhere": [{"label": "A", "log_odds": 1.0}]}
    shown = shown_pairs(
        tables,
        reverse,
        "A",
        text_words=["elsewhere"],
        top_k=2,
    )
    assert "elsewhere" in shown


def test_the_novelty_share_counts_the_rows_the_screen_cannot_form() -> None:
    """A row is new when the screen does not print BOTH of the pair's numbers."""
    rows = contrast_rows([row("a", 2.0)], [row("b", 1.0)])
    assert novelty_share(rows, frozenset({"a"}), frozenset({"b"})) == pytest.approx(1.0)
    assert novelty_share(rows, frozenset({"a", "b"}), frozenset({"a", "b"})) == pytest.approx(0.0)


def test_the_vocabulary_share_is_the_corpuss_own_words() -> None:
    """The rows outside the name-overlap vocabulary are the corpus's own."""
    rows = contrast_rows(
        [row("torture", 5.0, is_name_word=True), row("own", 2.0)],
        [],
        text_words=["torture"],
    )
    # "torture" is a name word and one of the text's words; "own" is neither
    assert vocabulary_share(rows, text_words=["torture"]) == pytest.approx(0.5)


def test_the_relevance_share_is_the_texts_own_rows() -> None:
    """The share of the rows that are one of the text's own words."""
    rows = contrast_rows([row("a", 2.0), row("b", 1.0)], [], text_words=["a"])
    assert relevance_share(rows) == pytest.approx(0.5)


def test_the_registered_bars_are_half() -> None:
    """The pre-registered bars (TEXT_PROCESSING_PLAN.md section 7.5)."""
    assert pytest.approx(0.50) == VOCABULARY_BAR
    assert pytest.approx(0.50) == RELEVANCE_BAR
    assert pytest.approx(0.50) == NOVELTY_BAR
    assert DISPLAY_TOP_K == 5
