"""Unit tests for the **grammatical classes** (the track's CS-34).

The hand-computed cases: the Penn tagset's grouping, the alignment of the
tags with the map's words (the track's own tokenization), the modal tag and
its tie-break, the ambiguity summary, the registered separation test, the
map's POS block and the **restriction** through the scorer and the predictor.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from cluster_rag.data import Record
from cluster_rag.predictors import build_predictor
from cluster_rag.predictors.base import PredictorContext
from cluster_rag.text_processors.features.corpus_map import build_corpus_map, corpus_map_block
from cluster_rag.text_processors.lexical.corpus_map_score import score_cells
from cluster_rag.text_processors.lexical.pos_class import (
    POS_CLASSES,
    ambiguity_summary,
    is_content_token,
    merge_distributions,
    modal_classes,
    modal_tag,
    pos_separation_test,
    tag_class,
    tag_counts_by_class,
    word_tag_counts,
)

LABELS = ("Alpha", "Beta", "Gamma", "Delta", "Zeta")


def _records() -> tuple[Record, ...]:
    """Eight texts, two per labelled label (the tiny map of CS-33's tests)."""
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


def _word_pos() -> dict[str, str]:
    """The tiny corpus's grammatical classes (by hand)."""
    return {
        "alpha": "noun",
        "beta": "verb",
        "gamma": "noun",
        "delta": "adjective",
    }


def _map():
    """The tiny corpus map, with its grammatical classes."""
    records = _records()
    return build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        LABELS,
        word_pos=_word_pos(),
    )


def test_the_penn_tags_are_grouped() -> None:
    """The coarse classes: the tagset's own groups, everything else is "other"."""
    assert tag_class("NN") == "noun"
    assert tag_class("NNS") == "noun"
    assert tag_class("NNP") == "noun"
    assert tag_class("VB") == "verb"
    assert tag_class("VBG") == "verb"
    assert tag_class("JJ") == "adjective"
    assert tag_class("RB") == "adverb"
    assert tag_class("CD") == "other"
    assert tag_class("XX") == "other"
    assert POS_CLASSES == ("noun", "verb", "adjective", "adverb", "other")


def test_the_tags_line_up_with_the_map_words() -> None:
    """The alignment is by position: the map's words are the content tokens."""
    text = "the alpha runs"
    # the track's own tokenization: "the", "alpha", "runs"
    tagged = [("the", "DT"), ("alpha", "NN"), ("runs", "VBZ")]
    counts = word_tag_counts(text, tagged)
    # the stopword is not a map word, so it never gets a tag
    assert set(counts) == {"alpha", "runs"}
    assert counts["alpha"] == Counter({"NN": 1})
    assert counts["runs"] == Counter({"VBZ": 1})
    assert is_content_token("alpha") is True
    assert is_content_token("the") is False
    assert is_content_token("a") is False


def test_a_misaligned_tagging_fails_loudly() -> None:
    """The tagger's tokens must be the track's own tokenization."""
    with pytest.raises(ValueError, match="tokens"):
        word_tag_counts("the alpha runs", [("the", "DT")])


def test_the_modal_tag_and_its_tie_break() -> None:
    """The modal tag is the most frequent one; the tie-break is the tagset's order."""
    assert modal_tag(Counter({"NN": 3, "VB": 1})) == "NN"
    assert modal_tag(Counter({"VB": 2, "NN": 2})) == "NN"  # the tie: the sorted tag
    assert modal_tag(Counter({"VB": 2, "NN": 1})) == "VB"
    assert modal_tag(Counter()) == ""


def test_the_corpus_distribution_merges_and_counts_the_ambiguity() -> None:
    """The per-word tag counts merge over the corpus; the ambiguity is reported."""
    first = {"alpha": Counter({"NN": 2}), "runs": Counter({"VB": 1})}
    second = {"alpha": Counter({"NN": 1, "VB": 1}), "runs": Counter({"VBZ": 1})}
    merged = merge_distributions([first, second])
    assert merged["alpha"] == Counter({"NN": 3, "VB": 1})
    assert merged["runs"] == Counter({"VB": 1, "VBZ": 1})
    summary = ambiguity_summary(merged)
    assert summary["n_words"] == 2
    assert summary["n_words_with_several_tags"] == 2
    assert summary["share_with_several_tags"] == pytest.approx(1.0)
    assert summary["words_with_several_tags"] == {
        "alpha": {"NN": 3, "VB": 1},
        "runs": {"VB": 1, "VBZ": 1},
    }
    # the class of a word is the class of its modal tag
    classes = modal_classes(merged)
    assert classes["alpha"] == "noun"
    assert classes["runs"] == "verb"
    # the per-tag counts: a word with two tags counts under both
    counts = tag_counts_by_class(merged)
    assert counts["NN"] == Counter({"NN": 3})
    assert counts["VB"] == Counter({"VB": 2})


def test_the_registered_separation_test() -> None:
    """CS-34's falsification: the nouns against the verbs."""
    from cluster_rag.text_processors.lexical.word_class import class_readout

    separating = class_readout(
        [("noun", 4.0), ("noun", 4.0), ("verb", 0.5), ("verb", -0.5)], classes=POS_CLASSES
    )
    test = pos_separation_test(separating)
    assert test["holds"] is True
    assert test["margin_met"] is True
    assert test["verb_share_positive_lower"] is True
    assert test["difference"] == pytest.approx(4.0)

    # the margin is not met
    narrow = class_readout(
        [("noun", 0.6), ("noun", 0.6), ("verb", 0.5), ("verb", -0.5)], classes=POS_CLASSES
    )
    assert pos_separation_test(narrow)["holds"] is False
    assert pos_separation_test(narrow)["margin_met"] is False

    # the share does not order the same way
    same_share = class_readout(
        [("noun", 10.0), ("noun", 10.0), ("noun", -10.0), ("verb", 0.5), ("verb", 0.5)],
        classes=POS_CLASSES,
    )
    assert pos_separation_test(same_share)["holds"] is False
    assert pos_separation_test(same_share)["verb_share_positive_lower"] is False


def test_the_map_artifact_carries_the_pos_block() -> None:
    """The block: the readout per class, the ambiguity and the registered test."""
    distribution = {
        "alpha": Counter({"NN": 2}),
        "beta": Counter({"VB": 1}),
        "gamma": Counter({"NN": 1, "VB": 1}),
        "delta": Counter({"JJ": 1}),
    }
    corpus = _map()
    block = corpus_map_block(
        corpus,
        split="train",
        tagger={"model_id": "averaged_perceptron_tagger_eng", "digest": "abc"},
        distribution=distribution,
    )
    pos = block["pos_classes"]
    assert pos["available"] is True
    assert pos["tagger"]["model_id"] == "averaged_perceptron_tagger_eng"
    assert pos["ambiguity"]["n_words"] == 4
    assert pos["ambiguity"]["n_words_with_several_tags"] == 1
    assert pos["words_by_class"]["noun"] == 2
    assert pos["readout_by_class"]["noun"]["n_pairs"] > 0
    assert "mean_log_odds" in pos["readout_by_class"]["noun"]
    assert "holds" in pos["falsification"]
    # the per-tag readout counts a word under every tag it takes
    assert "NN" in pos["readout_by_tag"]
    assert "VB" in pos["readout_by_tag"]
    # the pairs carry their grammatical class
    rows = {
        row["label"]: {word["word"]: word for word in row["words"]} for row in block["per_label"]
    }
    assert rows["Alpha"]["alpha"]["pos_class"] == "noun"
    assert rows["Alpha"]["beta"]["pos_class"] == "verb"


def test_a_map_without_tags_says_so() -> None:
    """An untagged corpus carries no grammatical readout (never a fabricated one)."""
    records = _records()
    corpus = build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        LABELS,
    )
    block = corpus_map_block(corpus, split="train")
    assert block["pos_classes"]["available"] is False
    assert "not tagged" in block["pos_classes"]["note"]


def test_the_restriction_drops_the_grammatical_classes() -> None:
    """The POS-restricted map scores as if the other words were not in the tables."""
    corpus = _map()
    # "gamma" sits in Beta's, Gamma's and Delta's tables
    full = score_cells(["gamma"], corpus, scoring="support_count")
    assert full[0, 1] == 1.0 and full[0, 2] == 1.0 and full[0, 3] == 1.0
    # the nouns-only map keeps it ("gamma" is a noun here)
    nouns = score_cells(["gamma"], corpus, scoring="support_count", restrict_pos=["noun"])
    assert nouns[0, 1] == 1.0 and nouns[0, 2] == 1.0 and nouns[0, 3] == 1.0
    # the verbs-only map does not see it at all
    verbs = score_cells(["gamma"], corpus, scoring="support_count", restrict_pos=["verb"])
    assert verbs[0, 1] == 0.0 and verbs[0, 2] == 0.0 and verbs[0, 3] == 0.0
    # and a word that is in no class at all is dropped by every restriction
    untagged = score_cells(["alpha"], corpus, scoring="support_count", restrict_pos=["noun"])
    assert untagged[0, 0] == 1.0


def test_an_unknown_grammatical_class_fails() -> None:
    """A bad restriction fails loudly, before any run."""
    with pytest.raises(ValueError, match="Unknown grammatical class"):
        score_cells(["gamma"], _map(), scoring="support_count", restrict_pos=["topic"])
    with pytest.raises(ValueError, match="at least one grammatical class"):
        score_cells(["gamma"], _map(), scoring="support_count", restrict_pos=[])


def test_the_predictor_takes_the_pos_restriction() -> None:
    """The predictor's ``restrict_pos`` restricts the map it scores with."""
    records = _records()
    context = PredictorContext(
        labels=LABELS,
        train_records=records,
        word_pos=_word_pos(),
    )
    predictor = build_predictor(
        "corpus_map",
        {"scoring": "support_count", "restrict_pos": ["verb"]},
        context,
        seed=42,
    )
    # "beta" is a verb here, "gamma" is not
    assert predictor.score(["beta"])[0, 0] == 1.0
    assert predictor.score(["gamma"])[0, 2] == 0.0
    described = predictor.describe()
    assert described["restrict_pos"] == ["verb"]
    assert described["fit_side"]["restrict_pos"] == ["verb"]


def test_the_predictor_refuses_a_pos_restriction_without_the_tags() -> None:
    """A bad parameter fails at construction, not halfway through a run."""
    records = _records()
    context = PredictorContext(labels=LABELS, train_records=records)
    with pytest.raises(ValueError, match="grammatical classes"):
        build_predictor(
            "corpus_map",
            {"restrict_pos": ["noun"]},
            context,
            seed=42,
        )
    with pytest.raises(ValueError, match="unknown grammatical"):
        build_predictor(
            "corpus_map",
            {"restrict_pos": ["topic"]},
            PredictorContext(labels=LABELS, train_records=records, word_pos=_word_pos()),
            seed=42,
        )
    with pytest.raises(ValueError, match="at least one"):
        build_predictor(
            "corpus_map",
            {"restrict_pos": []},
            PredictorContext(labels=LABELS, train_records=records, word_pos=_word_pos()),
            seed=42,
        )


def test_the_port_says_what_the_tagged_tokens_are() -> None:
    """The port's helper: the tokens of a tagged sequence."""
    from cluster_rag.pos.base import tagged_tokens

    assert tagged_tokens([("a", "DT"), ("b", "NN")]) == ("a", "b")
    assert tagged_tokens([]) == ()


def test_the_pure_package_does_not_import_the_tagger_client() -> None:
    """The track's package never imports the model client (the port only)."""
    import pathlib

    root = pathlib.Path(
        __import__("cluster_rag.text_processors", fromlist=["__file__"]).__file__
    ).parent
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(cluster_rag\.pos\.nltk_tagger|cluster_rag\.pos\.cache|nltk)\b"
    )
    for path in root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            assert not pattern.match(line), f"{path}: {line}"
