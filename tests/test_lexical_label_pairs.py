"""Unit tests for the **label-pair relation** (the track's CS-38).

The hand-computed cases: the vocabulary half (the shared words, the Jaccard,
the containment), the co-occurrence half (the shared texts, the lift over
independence), the sibling flag, the "no vocabulary of their own" class and
the readout block's own shape.
"""

from __future__ import annotations

import math

import pytest

from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.label_pairs import (
    CONTAINMENT_THRESHOLD,
    NEAR_DUPLICATE_JACCARD,
    label_pair_relation,
    label_pairs_block,
    separation_bootstrap,
    vocabulary_less_labels,
)

LABELS = ("A", "B", "Z")  # Z is the unseen holdout-label stand-in


def _records():
    """Four texts: two with A, two with B, and a shared vocabulary."""
    return (
        ("alpha bravo", ["A"]),
        ("alpha charlie", ["A"]),
        ("delta echo", ["B"]),
        ("delta foxtrot", ["B"]),
    )


def _map_and_sets():
    texts = [text for text, _ in _records()]
    label_sets = [labels for _, labels in _records()]
    return build_corpus_map(texts, label_sets, LABELS), texts, label_sets


def _pair(relation, label_a, label_b):
    return next(
        pair for pair in relation.pairs if (pair.label_a, pair.label_b) == (label_a, label_b)
    )


def test_the_vocabulary_half_is_hand_computable() -> None:
    """A's table is {alpha, bravo, charlie}, B's is {delta, echo, foxtrot}."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets)
    pair = _pair(relation, "A", "B")
    assert pair.shared_words == 0
    assert pair.jaccard == 0.0
    assert pair.containment == 0.0


def test_the_cooccurrence_half_is_hand_computable() -> None:
    """A and B never share a text: the lift is 0 and the text Jaccard too."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets)
    pair = _pair(relation, "A", "B")
    assert pair.n_both == 0
    assert pair.lift == 0.0
    assert pair.text_jaccard == 0.0


def test_a_shared_word_shows_up_in_the_overlap() -> None:
    """One shared word in two tables of three: the Jaccard is 1/5."""
    texts = ["alpha bravo", "alpha charlie", "delta bravo", "delta echo"]
    label_sets = [["A"], ["A"], ["B"], ["B"]]
    corpus_map = build_corpus_map(texts, label_sets, LABELS)
    relation = label_pair_relation(corpus_map, label_sets)
    pair = _pair(relation, "A", "B")
    assert pair.shared_words == 1
    assert pair.jaccard == pytest.approx(1 / 5)
    # the containment: the smaller table (three words each) holds one shared word
    assert pair.containment == pytest.approx(1 / 3)


def test_the_lift_is_the_observed_over_the_expected_count() -> None:
    """Two labels sharing a text out of four: the lift is 2 over 2."""
    texts = ["alpha", "alpha", "delta", "delta"]
    label_sets = [["A", "B"], ["A", "B"], ["A"], ["B"]]
    corpus_map = build_corpus_map(texts, label_sets, LABELS)
    relation = label_pair_relation(corpus_map, label_sets)
    pair = _pair(relation, "A", "B")
    assert pair.n_both == 2
    # the expected count under independence: 3 * 3 / 4 = 2.25
    assert pair.lift == pytest.approx(2 / 2.25)
    assert pair.text_jaccard == pytest.approx(2 / 4)


def test_the_sibling_flag_carries_the_catalogs_claim() -> None:
    """Two labels that share a parent are siblings; the rest are not."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets, parents={"A": "P", "B": "P"})
    assert _pair(relation, "A", "B").sibling is True
    other = label_pair_relation(corpus_map, label_sets, parents={"A": "P", "B": "Q"})
    assert _pair(other, "A", "B").sibling is False


def test_the_label_without_a_table_has_no_summary_partner() -> None:
    """Z has no table: it has no partner and no words."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets)
    summary = next(row for row in relation.per_label if row.label == "Z")
    assert summary.n_words == 0
    assert summary.best_partner == ""
    assert summary.best_jaccard == 0.0
    assert summary.best_containment == 0.0


def test_the_vocabulary_less_class_holds_the_empty_tables() -> None:
    """Z has no vocabulary of its own: it is in the class."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets)
    block = vocabulary_less_labels(relation)
    assert block["empty_table"] == ["Z"]
    assert "Z" in block["all"]


def test_a_label_inside_another_is_in_the_class() -> None:
    """A table that lies inside a partner's table has no vocabulary of its own."""
    texts = ["alpha", "alpha", "delta", "delta"]
    label_sets = [["A"], ["A"], ["B"], ["B"]]
    corpus_map = build_corpus_map(texts, label_sets, LABELS)
    relation = label_pair_relation(corpus_map, label_sets)
    # A's table is {alpha}, B's is {delta}: neither is inside the other
    assert vocabulary_less_labels(relation)["subset_of_a_neighbour"] == []
    texts = ["alpha delta", "alpha", "delta", "delta"]
    label_sets = [["A"], ["A"], ["B"], ["B"]]
    corpus_map = build_corpus_map(texts, label_sets, LABELS)
    relation = label_pair_relation(corpus_map, label_sets)
    subset = vocabulary_less_labels(relation)["subset_of_a_neighbour"]
    # A's table is {alpha, delta} and B's is {delta, delta}: A's words are not
    # inside B's (B has one distinct word), and B's single word is inside A's
    assert [row["label"] for row in subset] == ["B"]
    assert subset[0]["share"] == pytest.approx(1.0)


def test_the_block_carries_the_readout_and_the_catalogs_comparison() -> None:
    """The artifact's block: the knobs, the near-duplicates, the class, the pairs."""
    corpus_map, _texts, label_sets = _map_and_sets()
    relation = label_pair_relation(corpus_map, label_sets, parents={"A": "P", "B": "P"})
    block = label_pairs_block(relation)
    assert block["knobs"]["near_duplicate_jaccard"] == NEAR_DUPLICATE_JACCARD
    assert block["knobs"]["containment_threshold"] == CONTAINMENT_THRESHOLD
    assert block["summary"]["n_pairs"] == 3  # the three label pairs
    assert block["summary"]["n_sibling_pairs"] == 1
    assert block["catalog_comparison"]["sibling_pairs"] == 1
    assert block["catalog_comparison"]["chance_rate"] == pytest.approx(1 / 3)
    assert len(block["per_pair"]) == 3
    assert {row["label"] for row in block["per_label"]} == {"A", "B", "Z"}
    assert block["near_duplicates"] == []  # no pair shares a word
    assert block["neighbours"]


def test_the_separation_bootstrap_is_deterministic() -> None:
    """The same corpus and seed give the same interval."""
    corpus_map, texts, label_sets = _map_and_sets()
    parents = {"A": "P", "B": "P"}
    first = separation_bootstrap(texts, label_sets, LABELS, parents=parents, resamples=8, seed=42)
    second = separation_bootstrap(texts, label_sets, LABELS, parents=parents, resamples=8, seed=42)
    assert first["point"] == second["point"]
    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["resamples"] == 8


def test_the_relation_refuses_a_corpus_that_does_not_match() -> None:
    """A mismatched corpus fails before anything is computed."""
    corpus_map, _texts, _label_sets = _map_and_sets()
    with pytest.raises(ValueError, match="texts"):
        label_pair_relation(corpus_map, [["A"]])


def test_the_lift_of_a_pair_that_never_occurs_is_zero_not_infinite() -> None:
    """No expected count, no lift (and no division by zero)."""
    texts = ["alpha", "delta"]
    label_sets = [["A"], ["B"]]
    corpus_map = build_corpus_map(texts, label_sets, LABELS)
    relation = label_pair_relation(corpus_map, label_sets)
    pair = _pair(relation, "A", "B")
    assert pair.lift == 0.0
    assert math.isfinite(pair.lift)
