"""Unit tests for the label side's own words (the track's CS-55).

The hand-computed cases: the question vocabulary (the content words, no
stopwords), the label-side classes, the four arms' grids on a two-question toy,
and the blind counterpart (the questions permuted across the pairs).
"""

from __future__ import annotations

import math

import pytest

from cluster_rag.text_processors.lexical.question_lexicon import (
    LEXICON_ARMS,
    PRIMARY_LEXICON_ARM,
    build_question_lexicon,
    classify_question_words,
    lexicon_grid,
    lexicon_pair_scores,
    lexicon_readout,
)

Q1 = "Did the state torture detainees?"
Q2 = "Were the cases documented?"


def _lexicon():
    """Two questions, four paragraphs: a tiny label-side corpus."""
    questions = [Q1, Q2]
    lexicon = build_question_lexicon(questions)
    paragraphs = (
        "the state detained several detainees",  # Q1's words: state, detainees
        "the cases were documented",  # Q2's word: cases, documented
        "nothing relevant here",  # neither
        "torture and detainees were documented",  # Q1 (torture, detainees) + Q2
    )
    return lexicon, questions, paragraphs


def test_the_vocabulary_is_the_question_content_words() -> None:
    """The stopwords and the one-letter tokens never enter the channel."""
    lexicon, _, _ = _lexicon()
    assert lexicon.vocabularies[Q1] == frozenset({"state", "torture", "detainees"})
    # "the" and "were" are stopwords; "a" of a one-letter token would be dropped too
    assert "the" not in lexicon.vocabularies[Q2]
    assert lexicon.n_questions == 2
    # both questions share no content word
    assert lexicon.counts["state"] == 1


def test_the_label_side_classes_are_hand_computed() -> None:
    """CS-33's rule on the label side: distinctive 1-2, generic a quarter."""
    counts = {"w1": 1, "w2": 2, "w3": 3, "w4": 4}
    classes = classify_question_words(counts, n_questions=4)
    # the generic threshold for 4 questions is CS-33's max(3, ceil(1)) = 3
    assert classes["w1"] == "distinctive"
    assert classes["w2"] == "distinctive"
    assert classes["w3"] == "generic"
    assert classes["w4"] == "generic"


def test_the_four_arms_are_hand_computable() -> None:
    """The count, the idf, the topic-only and the register-only grids."""
    lexicon, questions, paragraphs = _lexicon()
    # force the classes so the arms are hand-computable on this toy: the
    # questions' words are all distinctive, so topic == all and the register
    # arm is empty
    grid_all = lexicon_grid(paragraphs, questions, lexicon.vocabularies, lexicon.classes)
    assert grid_all[0, 0] == 2.0  # state, detainees
    assert grid_all[0, 1] == 0.0
    assert grid_all[1, 1] == 2.0  # cases, documented
    assert grid_all[2, 0] == 0.0
    assert grid_all[3, 0] == 2.0  # torture, detainees
    # idf: the words are in one question each -> log(2/1) per word
    idf = lexicon.idf
    grid_idf = lexicon_grid(
        paragraphs, questions, lexicon.vocabularies, lexicon.classes, arm="idf", idf=idf
    )
    assert grid_idf[0, 0] == pytest.approx(2.0 * math.log(2.0))
    # every toy word is distinctive, so topic == all and register is empty
    grid_topic = lexicon_grid(
        paragraphs, questions, lexicon.vocabularies, lexicon.classes, arm="topic"
    )
    assert grid_topic.tolist() == grid_all.tolist()
    grid_register = lexicon_grid(
        paragraphs, questions, lexicon.vocabularies, lexicon.classes, arm="register"
    )
    assert grid_register.tolist() == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert set(LEXICON_ARMS) == {"all", "idf", "topic", "register"}
    assert PRIMARY_LEXICON_ARM == "all"
    with pytest.raises(ValueError, match="Unknown lexicon arm"):
        lexicon_grid(paragraphs, questions, lexicon.vocabularies, lexicon.classes, arm="bogus")


def test_the_blind_channel_is_the_permuted_question() -> None:
    """The blind pair scores read the OTHER question's vocabulary."""
    lexicon, questions, paragraphs = _lexicon()
    pair_paragraphs = ["the state detained several detainees", "the cases were documented"]
    real = lexicon_pair_scores(pair_paragraphs, [Q1, Q2], lexicon.vocabularies, lexicon.classes)
    blind = lexicon_pair_scores(pair_paragraphs, [Q2, Q1], lexicon.vocabularies, lexicon.classes)
    assert real.tolist() == [2.0, 2.0]
    assert blind.tolist() == [0.0, 0.0]
    # the readout: the vocabulary's shape and the zero-overlap share
    readout = lexicon_readout(lexicon, pair_paragraphs=pair_paragraphs, pair_questions=[Q1, Q2])
    assert readout["n_questions"] == 2
    assert readout["classes"]["distinctive"]["words"] == 5
    assert readout["paragraph_overlap"]["zero_share"] == 0.0
    assert readout["vocabulary_size"]["min"] == 2
    with pytest.raises(ValueError, match="same length"):
        lexicon_readout(lexicon, pair_paragraphs=["a"], pair_questions=[Q1, Q2])
