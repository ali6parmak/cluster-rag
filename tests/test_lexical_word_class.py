"""Unit tests for the **word-class readout** (the track's CS-33).

The hand-computed cases: the classes' own thresholds (a **share** of the
labelled tables, not a magic number), the readout's counts and signs, the
registered separation test, the map's class per pair and the name-word flag,
the **restriction** through the scorer and the predictor, and the surrogate's
own readout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import (
    build_corpus_map,
    corpus_map_block,
)
from cluster_rag.text_processors.lexical.corpus_map_score import score_cells
from cluster_rag.text_processors.lexical.word_class import (
    DISTINCTIVE_MAX_TABLES,
    SEPARATION_MARGIN,
    WORD_CLASSES,
    build_word_class_index,
    class_readout,
    classify_word,
    generic_threshold,
    label_name_words,
    separation_test,
)
from cluster_rag.text_processors.surrogate import word_class_readout

# Four labelled tables (so the generic class can exist: the threshold is a
# quarter of the labelled tables, with a floor of three) and one label with no
# train example (the unseen hold-out stand-in).
LABELS = ("Alpha", "Beta", "Gamma", "Delta", "Zeta")


def _records() -> tuple[Record, ...]:
    """Eight texts, two per labelled label.

    The words' own table counts (the class's input):

    - ``alpha`` in one table (Alpha) — **distinctive**;
    - ``beta`` in two tables (Alpha, Beta) — **distinctive**;
    - ``gamma`` in three tables (Beta, Gamma, Delta) — **generic** (the
      threshold is three);
    - ``delta`` in two tables (Beta, Delta) — **distinctive**.
    """
    return (
        Record(id="t1", text="alpha", language="en", labels=["Alpha"]),
        Record(id="t2", text="beta", language="en", labels=["Alpha"]),
        Record(id="t3", text="beta gamma", language="en", labels=["Beta"]),
        Record(id="t4", text="gamma delta", language="en", labels=["Beta"]),
        Record(id="t5", text="gamma", language="en", labels=["Gamma"]),
        Record(id="t6", text="gamma", language="en", labels=["Gamma"]),
        Record(id="t7", text="delta", language="en", labels=["Delta"]),
        Record(id="t8", text="gamma delta", language="en", labels=["Delta"]),
    )


def _map():
    """The tiny corpus map."""
    records = _records()
    return build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        LABELS,
    )


def _context() -> PredictorContext:
    """Train-only context: four labels with examples, one without."""
    return PredictorContext(labels=LABELS, train_records=_records())


def test_the_class_thresholds_are_hand_computable() -> None:
    """A quarter of the labelled tables is generic; one or two tables is distinctive."""
    assert generic_threshold(60) == 15
    assert generic_threshold(60) == math.ceil(60 * 0.25)
    # the floor: the classes never overlap on a small corpus
    assert generic_threshold(4) == DISTINCTIVE_MAX_TABLES + 1
    assert classify_word(1, 60) == "distinctive"
    assert classify_word(2, 60) == "distinctive"
    assert classify_word(3, 60) == "shared"
    assert classify_word(14, 60) == "shared"
    assert classify_word(15, 60) == "generic"
    assert classify_word(60, 60) == "generic"
    assert set(WORD_CLASSES) == {"distinctive", "shared", "generic"}


def test_a_word_in_no_table_has_no_class() -> None:
    """The classes cover the words that sit in at least one table."""
    with pytest.raises(ValueError, match="no class"):
        classify_word(0, 60)
    with pytest.raises(ValueError, match="non-negative"):
        generic_threshold(-1)


def test_the_readout_counts_the_pairs_and_the_signs() -> None:
    """The readout: the number of pairs, the two shares and the mean."""
    readout = class_readout(
        [
            ("distinctive", 2.0),
            ("distinctive", -1.0),
            ("distinctive", 4.0),
            ("generic", 0.5),
        ]
    )
    distinctive = readout["distinctive"]
    assert distinctive.n_pairs == 3
    assert distinctive.share_positive == pytest.approx(2 / 3)
    assert distinctive.share_negative == pytest.approx(1 / 3)
    assert distinctive.mean_value == pytest.approx(5 / 3)
    generic = readout["generic"]
    assert generic.n_pairs == 1
    assert generic.share_positive == 1.0
    assert generic.share_negative == 0.0
    assert generic.mean_value == pytest.approx(0.5)
    # an empty class is present and reads zero
    assert readout["shared"].n_pairs == 0


def test_the_readout_rejects_an_unknown_class() -> None:
    """A class name the module does not know is a programming error."""
    with pytest.raises(ValueError, match="Unknown word class"):
        class_readout([("topic", 1.0)])


def test_the_separation_test_needs_both_halves() -> None:
    """The registered falsification: the margin AND the lower positive share."""
    separating = class_readout(
        [("distinctive", 4.0), ("distinctive", 4.0), ("generic", 0.5), ("generic", -0.5)]
    )
    test = separation_test(separating)
    assert test["holds"] is True
    assert test["margin_met"] is True
    assert test["generic_share_positive_lower"] is True
    assert test["difference"] == pytest.approx(4.0)
    assert test["margin"] == SEPARATION_MARGIN

    # the margin is not met: the null holds even though the share is lower
    narrow = class_readout(
        [("distinctive", 0.6), ("distinctive", 0.6), ("generic", 0.5), ("generic", -0.5)]
    )
    assert separation_test(narrow)["holds"] is False
    assert separation_test(narrow)["margin_met"] is False

    # the share is not lower: the null holds even with a wide margin
    same_share = class_readout(
        [
            ("distinctive", 10.0),
            ("distinctive", 10.0),
            ("distinctive", -10.0),
            ("generic", 0.5),
            ("generic", 0.5),
        ]
    )
    assert separation_test(same_share)["holds"] is False
    assert separation_test(same_share)["margin_met"] is True
    assert separation_test(same_share)["generic_share_positive_lower"] is False


def test_the_map_assigns_the_class_and_the_name_flag() -> None:
    """The class comes from the word's own table count; the flag from the label."""
    corpus_map = _map()
    index = corpus_map.word_class_index()
    assert index.n_labelled_tables == 4
    assert index.generic_threshold == 3
    assert index.class_of("alpha") == "distinctive"
    assert index.class_of("beta") == "distinctive"
    assert index.class_of("delta") == "distinctive"
    assert index.class_of("gamma") == "generic"
    assert index.class_of("zulu") is None
    assert index.counts_by_class() == {"distinctive": 3, "shared": 0, "generic": 1}

    alpha_rows = corpus_map.topic_rows("Alpha")
    assert {row.word for row in alpha_rows} == {"alpha", "beta"}
    # the name word: "alpha" is in the label "Alpha"'s own name, "beta" is not
    assert next(row for row in alpha_rows if row.word == "alpha").is_name_word is True
    assert next(row for row in alpha_rows if row.word == "beta").is_name_word is False
    # the same word is a name word for one label and not for another
    gamma_rows = {row.word: row for row in corpus_map.topic_rows("Gamma")}
    assert gamma_rows["gamma"].is_name_word is True
    beta_rows = {row.word: row for row in corpus_map.topic_rows("Beta")}
    assert beta_rows["beta"].is_name_word is True
    assert beta_rows["gamma"].is_name_word is False


def test_the_name_words_are_the_labels_own_content_words() -> None:
    """The flag's vocabulary is the label's own content words."""
    assert label_name_words("Rights of the Child") == frozenset({"rights", "child"})
    assert label_name_words("Alpha") == frozenset({"alpha"})


def test_the_word_class_index_rejects_a_negative_count() -> None:
    """A negative table count is a programming error."""
    with pytest.raises(ValueError, match="negative table count"):
        build_word_class_index({"alpha": -1}, 4)


def test_the_restriction_drops_the_classes_from_the_scoring() -> None:
    """The restricted map scores as if the dropped words were not in the tables."""
    corpus_map = _map()
    # "gamma" sits in Beta's, Gamma's and Delta's tables
    full = score_cells(["gamma"], corpus_map, scoring="support_count")
    assert full[0, 1] == 1.0 and full[0, 2] == 1.0 and full[0, 3] == 1.0
    # the distinctive-only map does not see it at all
    distinctive = score_cells(
        ["gamma"], corpus_map, scoring="support_count", restrict=["distinctive"]
    )
    assert distinctive[0, 1] == 0.0 and distinctive[0, 2] == 0.0 and distinctive[0, 3] == 0.0
    # and the generic-only map sees only it
    generic = score_cells(["gamma"], corpus_map, scoring="support_count", restrict=["generic"])
    assert generic[0, 1] == 1.0 and generic[0, 2] == 1.0 and generic[0, 3] == 1.0
    assert generic[0, 0] == 0.0


def test_an_unknown_or_empty_restriction_fails() -> None:
    """A bad restriction fails loudly, before any run."""
    with pytest.raises(ValueError, match="Unknown word class"):
        score_cells(["gamma"], _map(), scoring="support_count", restrict=["topic"])
    with pytest.raises(ValueError, match="at least one word class"):
        score_cells(["gamma"], _map(), scoring="support_count", restrict=[])


def test_the_predictor_takes_the_restriction() -> None:
    """The predictor's ``restrict`` param restricts the map it scores with."""
    plain = build_predictor("corpus_map", {"scoring": "support_count"}, _context(), seed=42)
    restricted = build_predictor(
        "corpus_map",
        {"scoring": "support_count", "restrict": ["generic"]},
        _context(),
        seed=42,
    )
    assert restricted.score(["gamma"])[0, 2] == 1.0
    assert restricted.score(["alpha"])[0, 0] == 0.0
    assert plain.score(["gamma"])[0, 2] == 1.0
    # the lineage carries the restriction and the map's own classes
    described = restricted.describe()
    assert described["restrict"] == ["generic"]
    assert described["word_classes"]["generic_threshold"] == 3
    assert described["word_classes"]["words_by_class"]["generic"] == 1
    assert described["fit_side"]["restrict"] == ["generic"]


def test_the_predictor_refuses_a_bad_restriction() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    with pytest.raises(ValueError, match="unknown word class"):
        build_predictor("corpus_map", {"restrict": ["topic"]}, _context(), seed=0)
    with pytest.raises(ValueError, match="at least one"):
        build_predictor("corpus_map", {"restrict": []}, _context(), seed=0)
    with pytest.raises(ValueError, match="must be a list"):
        build_predictor("corpus_map", {"restrict": "generic"}, _context(), seed=0)


def test_the_map_artifact_carries_the_class_and_the_readout() -> None:
    """The block: the class per pair, the readout and the registered test."""
    block = corpus_map_block(_map(), split="train")
    classes = block["word_classes"]
    assert classes["n_labelled_tables"] == 4
    assert classes["generic_threshold"] == 3
    assert classes["words_by_class"] == {"distinctive": 3, "shared": 0, "generic": 1}
    assert classes["readout"]["generic"]["n_pairs"] == 3
    assert "mean_log_odds" in classes["readout"]["generic"]
    assert "holds" in classes["falsification"]
    # the pair's own row carries the class and the flag
    rows = {
        row["label"]: {word["word"]: word for word in row["words"]} for row in block["per_label"]
    }
    assert rows["Alpha"]["alpha"]["word_class"] == "distinctive"
    assert rows["Alpha"]["alpha"]["is_name_word"] is True
    assert rows["Alpha"]["beta"]["word_class"] == "distinctive"
    assert rows["Alpha"]["beta"]["is_name_word"] is False
    assert rows["Beta"]["gamma"]["word_class"] == "generic"
    # the reverse table carries the word's class too
    words = {row["word"]: row for row in block["per_word"]}
    assert words["gamma"]["word_class"] == "generic"


def test_the_surrogate_readout_splits_its_coefficients() -> None:
    """The model's own habits, read the way the corpus's readout is read."""
    vocabulary = ("alpha", "beta", "gamma")
    topics = ("Alpha", "Beta")
    coefficients = np.array(
        [
            [2.0, -1.0, 0.5],
            [0.0, 3.0, -2.0],
        ]
    )
    classes = {"alpha": "distinctive", "beta": "distinctive", "gamma": "generic"}
    block = word_class_readout(
        coefficients,
        vocabulary,
        topics,
        classes,
        name_words={"Alpha": frozenset({"alpha"}), "Beta": frozenset({"beta"})},
    )
    assert block["n_coefficients_in_a_class"] == 6
    assert block["n_coefficients_without_a_class"] == 0
    distinctive = block["readout"]["distinctive"]
    assert distinctive["n_pairs"] == 4
    assert distinctive["share_positive"] == pytest.approx(0.5)
    # the zero coefficient is neither positive nor negative
    assert distinctive["share_negative"] == pytest.approx(0.25)
    assert distinctive["mean_coefficient"] == pytest.approx(1.0)
    generic = block["readout"]["generic"]
    assert generic["n_pairs"] == 2
    assert generic["share_positive"] == pytest.approx(0.5)
    # the name-word flag: the two coefficients of the topics' own name words
    assert block["name_word_flag"]["n_pairs"] == 2
    assert block["name_word_flag"]["mean_coefficient"] == pytest.approx(2.5)


def test_the_surrogate_readout_counts_the_unclassified_words() -> None:
    """A word in no table has no class and is counted apart."""
    block = word_class_readout(
        np.array([[1.0, 2.0]]),
        ("alpha", "zulu"),
        ("Alpha",),
        {"alpha": "distinctive"},
    )
    assert block["n_coefficients_in_a_class"] == 1
    assert block["n_coefficients_without_a_class"] == 1


def test_the_surrogate_readout_checks_the_shapes() -> None:
    """A mismatched fit is a programming error."""
    with pytest.raises(ValueError, match="topic rows"):
        word_class_readout(np.zeros((2, 2)), ("alpha",), ("Alpha",), {})
    with pytest.raises(ValueError, match="feature columns"):
        word_class_readout(np.zeros((1, 3)), ("alpha", "beta"), ("Alpha",), {})
