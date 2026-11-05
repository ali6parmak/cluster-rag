"""Unit tests for the question map (the track's CS-48).

The hand-computed cases: the counts / lift / log-odds of a tiny paired corpus
on one question, the empty table of a question with no positive, the 2×2
refusal, the cross-fit availability (which fold's map can speak about which
pair) and the readouts the change set's falsification is stated against.

The map is the OMCT twin of the UPR corpus map (CS-23): the row is the
reviewed pair, the "text" is the pair's paragraph and the "label" is the
pair's question. Nothing here touches a model or a network.
"""

from __future__ import annotations

import math

import pytest

from cluster_rag.text_processors.features.corpus_map import log_odds as corpus_log_odds
from cluster_rag.text_processors.features.question_map import (
    REGISTER_QUARTER,
    build_question_map,
    cross_fit_availability,
    question_map_block,
    readout_block,
)

QUESTION = "Were the cases documented?"

# The hand-computed log-odds of the tiny map (4 reviewed pairs for QUESTION,
# 2 positive and 2 negative):
#   alpha: in both positives, in no negative -> log(5) + log(5)
#   bravo: in one positive, in no negative   -> log(3) + log(5/3)
#   delta: in no positive                    -> not a table row at all
LOG_ODDS_ALPHA = math.log(5.0) + math.log(5.0)
LOG_ODDS_SINGLE = math.log(3.0) + math.log(5.0 / 3.0)


def _pairs() -> tuple[list[str], list[str], list[int]]:
    """Four reviewed pairs of one question: two positive, two negative."""
    paragraphs = [
        "alpha bravo",  # positive
        "alpha charlie",  # positive
        "delta echo",  # negative
        "delta foxtrot",  # negative
    ]
    questions = [QUESTION] * 4
    labels = [1, 1, 0, 0]
    return paragraphs, questions, labels


def test_hand_computed_counts_lift_and_log_odds() -> None:
    """The tiny map's one table, word by word (the UPR discipline)."""
    paragraphs, questions, labels = _pairs()
    question_map = build_question_map(paragraphs, questions, labels)

    assert question_map.n_reviewed_pairs == 4
    assert question_map.n_positive == 2
    assert question_map.n_negative == 2
    assert question_map.n_questions == 1
    assert question_map.n_questions_with_table == 1
    assert question_map.empty_questions == ()

    rows = {row.word: row for row in question_map.table_rows(QUESTION)}
    # only words that occur in at least one POSITIVE are table rows
    assert set(rows) == {"alpha", "bravo", "charlie"}

    alpha = rows["alpha"]
    assert (alpha.n_with, alpha.df) == (2, 2)
    assert alpha.lift == pytest.approx(2.0)  # (2/2) / (2/4)
    assert alpha.log_odds == pytest.approx(LOG_ODDS_ALPHA)

    bravo = rows["bravo"]
    assert (bravo.n_with, bravo.df) == (1, 1)
    assert bravo.lift == pytest.approx(2.0)  # (1/1) / (2/4)
    assert bravo.log_odds == pytest.approx(LOG_ODDS_SINGLE)

    # the table is ordered by the log-odds (the strongest first)
    assert [row.word for row in question_map.table_rows(QUESTION)][:2] == ["alpha", "bravo"]
    # the reverse table reads the same data the other way
    assert [row.question for row in question_map.per_word["alpha"]] == [QUESTION]
    # the global document frequency counts every reviewed pair's paragraph
    assert question_map.word_df["delta"] == 2
    assert question_map.word_df["alpha"] == 2


def test_a_question_with_no_positive_has_an_empty_table() -> None:
    """Silence is a state of the map, reported, never fabricated."""
    paragraphs, questions, labels = _pairs()
    paragraphs.append("delta golf")
    questions.append("A question with no positive?")
    labels.append(0)
    question_map = build_question_map(paragraphs, questions, labels)

    assert question_map.n_questions == 2
    assert question_map.table_rows("A question with no positive?") == ()
    assert "A question with no positive?" in question_map.empty_questions
    assert question_map.n_questions_with_table == 1
    # the word that only ever occurs in the negative pairs is still counted
    assert question_map.word_df["golf"] == 1
    assert "golf" not in question_map.per_word  # never in a positive: no table row


def test_counts_that_do_not_form_a_2x2_are_refused() -> None:
    """The shared formula refuses impossible counts (the UPR guard)."""
    with pytest.raises(ValueError, match="2x2"):
        corpus_log_odds(n_with=5, df=2, n_label=4, n_texts=6)
    with pytest.raises(ValueError, match="same length"):
        build_question_map(["a b"], ["q1", "q2"], [1])
    with pytest.raises(ValueError, match="0/1"):
        build_question_map(["a b"], ["q"], [2])


def test_the_cross_fit_availability_is_hand_computed() -> None:
    """Two folds: each pair is read against the map that never saw it."""
    paragraphs, questions, labels = _pairs()
    # fold 0 = the two positives, fold 1 = the two negatives
    folds = [0, 0, 1, 1]
    availability = cross_fit_availability(
        paragraphs,
        questions,
        labels,
        folds,
        documents=["Doc A"] * 4,
        pair_ids=["P-1", "P-2", "P-3", "P-4"],
    )

    # the positives are read against the fold built from the negatives only:
    # that map has no positive for the question, so its table is empty
    by_pair = {row["pair_id"]: row for row in availability["per_pair"]}
    assert by_pair["P-1"]["has_table"] is False
    assert by_pair["P-1"]["n_evidence"] == 0
    assert by_pair["P-2"]["has_table"] is False
    # the negatives are read against the fold built from the positives only:
    # the table exists, but their paragraphs share no word with it
    assert by_pair["P-3"]["has_table"] is True
    assert by_pair["P-3"]["n_evidence"] == 0
    assert by_pair["P-4"]["has_table"] is True

    overall = availability["overall"]
    assert overall["n_pairs"] == 4
    assert overall["with_table"] == 2
    assert overall["with_table_share"] == pytest.approx(0.5)
    assert overall["with_evidence"] == 0  # the map can speak about none of them
    assert overall["n_floor_pairs"] == 2
    assert availability["by_fold"][0]["n_pairs"] == 2
    assert availability["by_document"]["Doc A"]["with_table_share"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="at least two folds"):
        cross_fit_availability(paragraphs, questions, labels, [0, 0, 0, 0])


def test_the_thickness_readout_counts_words_shared_by_positives() -> None:
    """A word in two positives generalizes; a word in one is a memory."""
    paragraphs = ["alpha bravo", "alpha charlie", "delta echo", "delta foxtrot"]
    question_map = build_question_map(paragraphs, [QUESTION] * 4, [1, 1, 0, 0])
    readout = readout_block(question_map)

    assert readout["thickness"]["n_tables_with_shared_word"] == 1
    assert readout["thickness"]["median_shared_words"] == 1.0
    # the log-odds' head (alpha) is in both positives and in no negative
    assert readout["head_agreement"]["n_tables"] == 1
    assert readout["log_odds_head"]["n_only_in_positives"] == 1
    assert readout["register"]["quarter"] == 1  # a single question: a quarter is one table
    assert readout["register"]["n_words_in_quarter"] == 3


def test_the_block_carries_the_unit_and_the_shape() -> None:
    """The artifact's block says what the unit is and what the map holds."""
    paragraphs, questions, labels = _pairs()
    question_map = build_question_map(paragraphs, questions, labels)
    availability = cross_fit_availability(paragraphs, questions, labels, [0, 0, 1, 1])
    block = question_map_block(
        question_map,
        availability,
        folds=2,
        seed=42,
        n_paragraphs=4,
        n_documents=1,
    )

    assert block["unit"]["label"] == "the pair's question (the text of the question column)"
    assert block["corpus"]["n_reviewed_pairs"] == 4
    assert block["map"]["n_questions"] == 1
    assert block["map"]["n_words"] == 3
    assert block["map"]["n_stored_pairs"] == 3
    assert block["map"]["n_contrastable_questions"] == 1
    assert block["cross_fit"]["folds"] == 2
    assert block["availability"]["overall"]["n_pairs"] == 4
    assert set(block["tables"]) == {QUESTION}
    assert block["words"]["alpha"] == {"tables": 1, "df": 2}
    assert REGISTER_QUARTER == 0.25


def test_an_unjudged_cell_cannot_enter_the_map() -> None:
    """Subset B has no label: the builder accepts 0/1 only (the leakage guard)."""
    with pytest.raises(ValueError, match="0/1"):
        build_question_map(["a b"], ["q"], [-1])  # -1 is "unjudged", never a label
