"""Tests for the decision probe's pure core (the track's Stage 3.5).

Contracts under test:

- the **call plan**: the ``pair`` shape (the pair in one state) and the
  ``paragraph`` shape (the paragraph alone as the state, with the question in
  the instructions, batched up to 64 questions per call);
- the **synthetic competence control**: the positives share a distinctive
  phrase with their paragraph, the negatives share none, and the construction
  is deterministic;
- the **measurement**: the AUCs, the rank percentiles inside the *scored*
  cells of a paragraph, the calibration block and the control-side floor gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.text_processors.decision.probe import (
    GROUP_COMPETENCE,
    GROUP_CONTROL,
    GROUP_JUDGED,
    STATE_PAIR,
    STATE_PARAGRAPH,
    DecisionCell,
    build_competence_cells,
    distinctive_phrase,
    measure_competence_control,
    measure_decision_arm,
    plan_calls,
)

INSTRUCTIONS = "Is the paragraph about the same topic as the question?"


def _cell(
    index: int, *, group: str, label: int | None, paragraph: str, question: str
) -> DecisionCell:
    """One cell of a two-paragraph, two-question toy grid."""
    return DecisionCell(
        cell_id=f"c{index}",
        group=group,
        document="doc",
        paragraph_row=index % 2,
        question_col=index % 2,
        paragraph_text=paragraph,
        question_text=question,
        label=label,
    )


# -- the call plan ---------------------------------------------------------


def test_the_pair_shape_makes_one_call_per_cell() -> None:
    """The pair shape puts the pair in one state and asks one question per call."""
    cells = [
        _cell(0, group=GROUP_JUDGED, label=1, paragraph="p0", question="q0"),
        _cell(1, group=GROUP_JUDGED, label=0, paragraph="p1", question="q1"),
    ]
    calls = plan_calls(
        cells, state_shape=STATE_PAIR, instructions=INSTRUCTIONS, question_type="noul"
    )
    assert len(calls) == 2
    assert calls[0].state == {"paragraph": "p0", "question": "q0"}
    assert calls[0].specs[0].instructions == INSTRUCTIONS


def test_the_paragraph_shape_batches_a_paragraphs_questions() -> None:
    """The paragraph shape asks up to 64 questions about one paragraph."""
    cells = [
        _cell(index, group=GROUP_JUDGED, label=1, paragraph="p0", question=f"q{index}")
        for index in range(4)
    ]
    calls = plan_calls(
        cells,
        state_shape=STATE_PARAGRAPH,
        instructions=INSTRUCTIONS,
        question_type="noul",
        max_questions_per_call=3,
    )
    assert len(calls) == 2
    assert calls[0].state == {"paragraph": "p0"}
    assert len(calls[0].specs) == 3
    assert len(calls[1].specs) == 1
    assert calls[0].specs[0].instructions.startswith("Question: q0")
    assert calls[0].specs[0].instructions.endswith(INSTRUCTIONS)


def test_the_paragraph_shape_keeps_the_question_in_the_instructions() -> None:
    """The question is never in the state of the paragraph shape."""
    cells = [_cell(0, group=GROUP_JUDGED, label=1, paragraph="p0", question="q0")]
    calls = plan_calls(
        cells, state_shape=STATE_PARAGRAPH, instructions=INSTRUCTIONS, question_type="noul"
    )
    assert "question" not in calls[0].state
    assert calls[0].specs[0].instructions.startswith("Question: q0")


def test_the_call_plan_rejects_an_unknown_state_shape() -> None:
    """A typo in the state's shape fails loudly."""
    with pytest.raises(ValueError):
        plan_calls([], state_shape="pairrot", instructions=INSTRUCTIONS, question_type="noul")


def test_the_budget_truncates_the_paragraph_and_never_the_question() -> None:
    """A cell that does not fit loses paragraph characters, not question ones."""
    cells = [
        DecisionCell(
            cell_id="c0",
            group=GROUP_JUDGED,
            document="doc",
            paragraph_row=0,
            question_col=0,
            paragraph_text="p" * 5000,
            question_text="q" * 100,
            label=1,
        )
    ]
    calls = plan_calls(
        cells,
        state_shape=STATE_PAIR,
        instructions=INSTRUCTIONS,
        question_type="noul",
        max_input_tokens=100,
    )
    state = calls[0].state
    assert state["question"] == "q" * 100
    assert len(state["paragraph"]) < 5000
    assert calls[0].specs[0].instructions == INSTRUCTIONS


def test_the_budget_splits_a_batched_call() -> None:
    """A paragraph's questions are split into calls that fit the budget."""
    cells = [
        DecisionCell(
            cell_id=f"c{index}",
            group=GROUP_JUDGED,
            document="doc",
            paragraph_row=0,
            question_col=index,
            paragraph_text="p" * 100,
            question_text=f"q{index}" * 100,
            label=1,
        )
        for index in range(6)
    ]
    calls = plan_calls(
        cells,
        state_shape=STATE_PARAGRAPH,
        instructions=INSTRUCTIONS,
        question_type="noul",
        max_input_tokens=200,
    )
    assert len(calls) > 1
    assert sum(len(call.specs) for call in calls) == len(cells)


# -- the competence control ------------------------------------------------


def test_the_competence_control_shares_a_phrase_only_in_the_positives() -> None:
    """The positives quote a phrase of their paragraph, the negatives do not."""
    paragraphs = [
        "The Committee recalls that forced evictions rose sharply in rural areas",
        "The parliament approved the national budget for the health sector",
        "Three human rights defenders were detained during the reporting period",
        "The delegation visited prisons and heard complaints from inmates",
    ]
    cells = build_competence_cells(paragraphs, n_pairs=4)
    assert len(cells) == 8
    for index in range(0, 8, 2):
        positive, negative = cells[index], cells[index + 1]
        assert positive.label == 1
        assert negative.label == 0
        assert positive.paragraph_text == negative.paragraph_text
        phrase = positive.question_text.split('"')[1]
        other = negative.question_text.split('"')[1]
        assert phrase in positive.paragraph_text.lower()
        assert other not in positive.paragraph_text.lower()


def test_the_competence_control_is_deterministic() -> None:
    """The same corpus gives the same control."""
    paragraphs = [
        "The Committee recalls that forced evictions rose sharply in rural areas",
        "The parliament approved the national budget for the health sector",
    ]
    first = build_competence_cells(paragraphs, n_pairs=2)
    second = build_competence_cells(paragraphs, n_pairs=2)
    assert [cell.question_text for cell in first] == [cell.question_text for cell in second]


def test_the_distinctive_phrase_prefers_the_rarest_words() -> None:
    """The phrase is the window whose words are the rarest in the corpus."""
    paragraphs = [
        "The Committee recalls the state obligation to protect minorities",
        "The Committee recalls the state obligation to protect minorities",
    ]
    phrase = distinctive_phrase(paragraphs[0], {"the": 2, "committee": 2, "recalls": 2})
    assert phrase is not None
    assert "state" in phrase or "obligation" in phrase


def test_the_competence_control_measures_the_auc() -> None:
    """A model that sees the shared phrase separates the two groups."""
    cells = build_competence_cells(
        [
            "The Committee recalls that forced evictions rose sharply in rural areas",
            "The parliament approved the national budget for the health sector",
        ],
        n_pairs=2,
    )
    probabilities = [0.9, 0.1] * (len(cells) // 2)
    block = measure_competence_control(cells, probabilities)
    assert block["n_positive"] == 2
    assert block["n_negative"] == 2
    assert block["pair_auc"] == 1.0


# -- the measurement -------------------------------------------------------


def test_the_measurement_reports_the_aucs_and_the_control_side() -> None:
    """The arm's block carries the AUCs, the distributions and the floor gap."""
    cells = [
        _cell(0, group=GROUP_JUDGED, label=1, paragraph="p0", question="q0"),
        _cell(1, group=GROUP_JUDGED, label=0, paragraph="p1", question="q1"),
        _cell(2, group=GROUP_CONTROL, label=None, paragraph="p0", question="q1"),
    ]
    block = measure_decision_arm(cells, [0.9, 0.1, 0.2])
    assert block["pair_auc"]["overall"] == 1.0
    assert block["pair_auc"]["macro"] == 1.0
    assert block["pair_auc_decomposition"]["within_group_auc"] is not None
    assert block["score_distributions"]["label1"]["mean"] == 0.9
    assert block["score_distributions"]["control"]["mean"] == 0.2
    assert block["floor_gap"]["gap"] == pytest.approx(0.7)
    assert block["calibration"]["pooled"]["n"] == 2
    assert block["calibration"]["pooled"]["ece"] is not None


def test_the_rank_percentile_is_inside_the_scored_cells() -> None:
    """A cell's rank is measured inside the cells the run scored for its paragraph."""
    cells = [
        DecisionCell(
            cell_id="c0",
            group=GROUP_JUDGED,
            document="doc",
            paragraph_row=0,
            question_col=0,
            paragraph_text="p0",
            question_text="q0",
            label=1,
        ),
        DecisionCell(
            cell_id="c1",
            group=GROUP_JUDGED,
            document="doc",
            paragraph_row=0,
            question_col=1,
            paragraph_text="p0",
            question_text="q1",
            label=0,
        ),
    ]
    block = measure_decision_arm(cells, [0.9, 0.1])
    percentiles = block["negative_enrichment"]["rank_percentile"]
    assert percentiles["label1"]["quantiles"]["0.5"] == 0.0
    assert percentiles["label0"]["quantiles"]["0.5"] == 0.5
    assert block["hit_at_k"]["label1"]["hit@1"] == 1.0


def test_the_measurement_rejects_a_length_mismatch() -> None:
    """The cells and the probabilities must line up."""
    with pytest.raises(ValueError):
        measure_decision_arm([], [0.5])


def test_the_competence_group_is_never_pooled_with_the_control() -> None:
    """The competence cells live in their own group."""
    cells = build_competence_cells(
        [
            "The Committee recalls that forced evictions rose sharply in rural areas",
            "The parliament approved the national budget for the health sector",
        ],
        n_pairs=1,
    )
    assert all(cell.group == GROUP_COMPETENCE for cell in cells)
    assert np.isclose(measure_competence_control(cells, [0.9, 0.1])["pair_auc"], 1.0)
