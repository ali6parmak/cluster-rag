"""Unit tests for the **phrase table** (the track's CS-39).

The hand-computed cases: the phrase unit (the bigrams over the track's own
token stream), the table's shape, the class split and the **pair interaction**
(how much more a phrase says than its two words), plus the union arm in the
predictor.
"""

from __future__ import annotations

import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.features.phrase_map import (
    build_phrase_map,
    phrase_interactions,
    phrase_map_block,
    phrase_units,
)
from cluster_rag.text_processors.lexical.corpus_map_score import (
    UNION_SCORING,
    score_cells,
)

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in

TEXTS = (
    "Continue to ensure that the rights of the child are protected",
    "The death penalty and early marriage are human rights issues",
)


def _map(texts=TEXTS, label_sets=(["A"], ["B"])):
    return build_corpus_map(list(texts), list(label_sets), LABELS)


def test_the_unit_is_the_bigrams_of_the_token_stream() -> None:
    """The stopwords stay: the genre's formulas live in them."""
    units = phrase_units(TEXTS[0])
    assert units[:3] == ["continue to", "to ensure", "ensure that"]
    assert "the rights" in units
    assert "rights of" in units


def test_the_table_holds_the_phrases_and_not_the_words() -> None:
    """The phrase table is the map over the bigrams, with the map's own shape."""
    phrase_map = build_phrase_map(list(TEXTS), [["A"], ["B"]], LABELS)
    word_map = _map()
    assert phrase_map.n_words > 0
    assert phrase_map.n_pairs == phrase_map.n_words  # one label per text here
    assert "continue to" in phrase_map.per_word
    assert "death penalty" in phrase_map.per_word
    # the word map's own words are not phrases
    assert "continue" in word_map.per_word
    assert "continue" not in phrase_map.per_word


def test_the_phrases_are_rare_where_the_words_are_not() -> None:
    """The shape: the phrase table is far more fragile than the word table."""
    phrase_map = build_phrase_map(list(TEXTS), [["A"], ["B"]], LABELS)
    block = phrase_map_block(phrase_map, _map())
    assert block["shape"]["n_phrases"] == phrase_map.n_words
    assert block["shape"]["n_pairs"] == phrase_map.n_pairs
    assert 0.0 <= block["shape"]["hapax_share"] <= 1.0


def test_the_interaction_is_the_phrase_minus_its_words() -> None:
    """The pair interaction: the phrase's log-odds minus its words'."""
    phrase_map = build_phrase_map(list(TEXTS), [["A"], ["B"]], LABELS)
    word_map = _map()
    interactions = phrase_interactions(phrase_map, word_map)
    assert len(interactions) == phrase_map.n_pairs
    for interaction in interactions:
        assert interaction.interaction == pytest.approx(
            interaction.log_odds - interaction.words_log_odds
        )


def test_the_block_carries_the_shape_the_classes_and_the_interaction() -> None:
    """The artifact's block: the three readings the change set asks for."""
    phrase_map = build_phrase_map(list(TEXTS), [["A"], ["B"]], LABELS)
    block = phrase_map_block(phrase_map, _map())
    assert block["available"] is True
    assert block["shape"]["n_texts"] == 2
    assert block["classes"]["generic_share"] >= 0.0
    assert block["classes"]["word_table"]["generic_share"] >= 0.0
    assert block["interaction"]["n_pairs"] == phrase_map.n_pairs
    assert len(block["interaction"]["top"]) == min(20, phrase_map.n_pairs)


def test_the_scorer_can_score_the_phrase_table() -> None:
    """The phrase arm's units are the text's bigrams."""
    phrase_map = build_phrase_map(list(TEXTS), [["A"], ["B"]], LABELS)
    scores = score_cells(
        list(TEXTS),
        phrase_map,
        scoring="support_count",
        units=phrase_units,
    )
    assert scores.shape == (2, 3)
    # the first text's bigrams are in A's table, the second's in B's
    assert scores[0, 0] > 0
    assert scores[1, 1] > 0


def test_the_union_arm_adds_the_two_maps() -> None:
    """The union: the word map's score plus the phrase map's own."""
    records = (
        Record(id="t1", text="alpha bravo", language="en", labels=["A"]),
        Record(id="t2", text="charlie delta", language="en", labels=["A"]),
        Record(id="t3", text="delta echo", language="en", labels=["B"]),
        Record(id="t4", text="echo foxtrot", language="en", labels=["B"]),
    )
    context = PredictorContext(labels=LABELS, train_records=records)
    predictor = build_predictor(
        "corpus_map", {"scoring": UNION_SCORING, "fit_side_k": 2, "folds": 2}, context, seed=42
    )
    scores = predictor.score(["alpha bravo"])
    assert scores.shape == (1, 3)
    fit_side = predictor.describe()["fit_side"]
    assert UNION_SCORING in fit_side["out_of_fold_micro_f1"]
    assert UNION_SCORING in fit_side["in_fold_micro_f1"]


def test_the_union_arm_is_absent_from_the_other_arms_blocks() -> None:
    """The earlier runs' artifacts keep their exact content (no union key)."""
    records = (
        Record(id="t1", text="alpha bravo", language="en", labels=["A"]),
        Record(id="t2", text="charlie delta", language="en", labels=["A"]),
        Record(id="t3", text="delta echo", language="en", labels=["B"]),
        Record(id="t4", text="echo foxtrot", language="en", labels=["B"]),
    )
    context = PredictorContext(labels=LABELS, train_records=records)
    predictor = build_predictor("corpus_map", {"fit_side_k": 2, "folds": 2}, context, seed=42)
    fit_side = predictor.describe()["fit_side"]
    assert UNION_SCORING not in fit_side["out_of_fold_micro_f1"]
    assert UNION_SCORING not in fit_side["in_fold_micro_f1"]
