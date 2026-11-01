"""Tests for the **label lexicon** (the track's CS-35): the label-side representation.

Contracts under test (TEXT_PROCESSING_PLAN.md section 7.4, CS-35):

- the **PIFA weight** — the TF-IDF centroid's coordinate for a (label, word)
  pair, computed from the corpus map's own counts;
- the **mined words** — the top K by that weight, in a deterministic order;
- the **expansion route** — a label with no train example gets the words its
  **name** matches in the corpus (exact, else a prefix match), with no labels
  involved, and the **name** itself when nothing matches;
- the artifact's **block** — the routes, the weights and the counts.
"""

from __future__ import annotations

import math

import pytest

from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.label_lexicon import (
    DEFAULT_EXPANSION_MIN_PREFIX,
    DEFAULT_TOP_K,
    ROUTE_EXPANDED,
    ROUTE_FALLBACK,
    ROUTE_MINED,
    build_label_lexicon,
    expansion_words,
    inverse_document_frequency,
    label_lexicon_block,
    pifa_weight,
)

# A tiny corpus: five texts, two labels. "detention" sits in two of the three
# "Detention" texts, so it outweighs the words that sit in one of them.
CORPUS = (
    ("Call for the courts to enforce the warrants", ("Detention",)),
    ("Detention and arrest must be inspected", ("Detention",)),
    ("Detention centres must be inspected daily", ("Detention",)),
    ("Ensure access to courts and legal aid for the poor", ("Poverty",)),
    ("School meals and poverty relief", ("Poverty",)),
)
LABELS = ("Detention", "Other", "Poverty")


def _map():
    """The tiny corpus map (the train split's own association)."""
    texts = [text for text, _ in CORPUS]
    label_sets = [labels for _, labels in CORPUS]
    return build_corpus_map(texts, label_sets, LABELS)


def test_the_inverse_document_frequency_is_the_smoothed_form() -> None:
    """log((1 + N) / (1 + df)): a word in every text has a zero weight."""
    assert inverse_document_frequency(0, 10) == pytest.approx(math.log(11 / 1))
    assert inverse_document_frequency(10, 10) == pytest.approx(0.0)


def test_the_pifa_weight_is_the_centroids_coordinate() -> None:
    """The weight is tf x idf: the centroid's coordinate for the word."""
    weight = pifa_weight(n_with=1, df=2, n_label=2, n_texts=10)
    assert weight == pytest.approx(0.5 * math.log(11 / 3))
    # the same word in more of the label's instances weighs more
    heavier = pifa_weight(n_with=2, df=2, n_label=2, n_texts=10)
    assert heavier == pytest.approx(1.0 * math.log(11 / 3))
    assert heavier > weight


def test_the_pifa_weight_rejects_impossible_counts() -> None:
    """A word cannot be in more texts than the label has."""
    with pytest.raises(ValueError, match="at least one train example"):
        pifa_weight(n_with=1, df=2, n_label=0, n_texts=10)
    with pytest.raises(ValueError, match="more texts than the label"):
        pifa_weight(n_with=3, df=4, n_label=2, n_texts=10)


def test_the_mined_words_are_the_top_coordinates() -> None:
    """The mined words are the top K by the PIFA weight, ties by the word."""
    lexicon = build_label_lexicon(_map(), top_k=2)
    entries = lexicon.words["Detention"]
    # "detention" sits in two of the label's three instances, so it outweighs
    # the words that sit in one of them; "inspected" ties with it (two of the
    # three) and the tie is broken alphabetically.
    assert [entry.word for entry in entries] == ["detention", "inspected"]
    assert entries[0].weight == entries[1].weight
    assert entries[0].weight > lexicon.words["Detention"][0].weight * 0.9
    assert lexicon.route("Detention") == ROUTE_MINED
    # the counts ride along
    first = entries[0]
    assert first.n_with == 2
    assert first.df == 2
    assert first.log_odds != 0.0


def test_the_mined_words_keep_the_counts_and_the_top_k() -> None:
    """The K is the number of words a label keeps."""
    lexicon = build_label_lexicon(_map(), top_k=1)
    assert len(lexicon.label_words("Detention")) == 1
    assert len(lexicon.label_words("Poverty")) == 1


def test_the_expansion_route_matches_the_name_against_the_corpus() -> None:
    """A label with no example gets the corpus words its name matches."""
    vocabulary = {"arrest": 3, "custody": 2, "school": 1}
    # a token that is in the vocabulary contributes exactly that word
    assert expansion_words("Arrest", vocabulary) == ("arrest",)
    # a token that is not gets the prefix matches (at least 4 shared characters)
    assert expansion_words("Arrests", vocabulary) == ("arrest",)
    # a token with no match contributes nothing
    assert expansion_words("Zebra", vocabulary) == ()


def test_the_expansion_route_needs_four_shared_characters() -> None:
    """The prefix match's minimum: four characters (so "refugees" -> "re" is out)."""
    vocabulary = {"refugees": 2, "re": 5, "seek": 1}
    assert expansion_words("Refugees", vocabulary) == ("refugees",)
    assert expansion_words("Refugee", vocabulary) == ("refugees",)
    assert expansion_words("Seekers", vocabulary) == ("seek",)


def test_the_expansion_route_orders_the_rarest_word_first() -> None:
    """The expansion's order: rising document frequency, ties alphabetically."""
    vocabulary = {"rights": 10, "general": 2, "association": 5}
    assert expansion_words("Rights and general association", vocabulary) == (
        "general",
        "association",
        "rights",
    )


def test_a_label_with_no_words_falls_back_to_its_name() -> None:
    """The fallback route: the label's own name is the text."""
    lexicon = build_label_lexicon(_map())
    assert lexicon.route("Other") == ROUTE_FALLBACK
    assert lexicon.label_words("Other") == ()
    assert lexicon.label_text("Other") == "Other"


def test_the_routes_are_counted_and_the_block_carries_the_numbers() -> None:
    """The block: the routes' summary and every label's words."""
    lexicon = build_label_lexicon(_map(), top_k=3)
    block = label_lexicon_block(lexicon)
    assert block["routes"] == {ROUTE_MINED: 2, ROUTE_EXPANDED: 0, ROUTE_FALLBACK: 1}
    assert block["top_k"] == 3
    assert block["expansion_min_prefix"] == DEFAULT_EXPANSION_MIN_PREFIX
    rows = {row["label"]: row for row in block["per_label"]}
    assert rows["Detention"]["route"] == ROUTE_MINED
    assert rows["Detention"]["n_words"] == 3
    assert rows["Detention"]["words"][0]["word"] == "detention"
    assert rows["Detention"]["words"][0]["weight"] > 0.0
    assert rows["Other"]["route"] == ROUTE_FALLBACK
    assert rows["Other"]["text"] == "Other"


def test_the_defaults_are_the_registered_ones() -> None:
    """K = 10 and a four-character prefix (the plan's numbers)."""
    assert DEFAULT_TOP_K == 10
    assert DEFAULT_EXPANSION_MIN_PREFIX == 4
