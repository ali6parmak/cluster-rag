"""The decision probe's driver (``decide-omct``).

The text-processing track's Stage 3.5 (TEXT_PROCESSING_PLAN.md section 3):
**is the reviewed set learnable at all?** Every embedder sits at chance on it
(AUC 0.468 / 0.500 / 0.537) and the reviewed set is **rank-selected** (exp18's
model chose which pairs got reviewed), so those numbers may be measuring the
selection rather than the labels. A **decision model** — a frozen, zero-shot,
calibrated scorer — is the first *independent* ranker on this benchmark.

The driver orchestrates:

- the model-independent part (the dataset view: the grid, the strata, the
  seeded control sample) — built by :mod:`cluster_rag.omct_analysis`;
- the pure core (the question specs, the call plan, the **synthetic
  competence control** and the measurement) — in
  :mod:`cluster_rag.text_processors.decision`;
- the model client (the ``/v1/systemone`` caller behind the
  :class:`~cluster_rag.decision.base.Decider` port) — wired in by the CLI.

**The two negative subsets are never pooled** (the reviewer-verified hard
negatives and the unjudged-proxy control), and the **competence control** is
what tells *"the data is at the ceiling"* from *"the model is the wrong
shape"*: a chance-level result on the reviewed set is ambiguous without it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cluster_rag.decision.base import Decider, answer_probability
from cluster_rag.omct_analysis import DatasetView
from cluster_rag.preprocess import apply_question_transform
from cluster_rag.text_processors.decision.probe import (
    GROUP_CONTROL,
    GROUP_JUDGED,
    GROUP_UNJUDGED,
    DecisionCell,
    build_competence_cells,
    measure_competence_control,
    measure_decision_arm,
    plan_calls,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionArm:
    """One arm's prompt: the model's state shape, the question type and the task."""

    name: str
    state: str
    question_type: str
    instructions: str
    options: tuple[str, ...] = ()
    criteria: Mapping[str, str] = None  # type: ignore[assignment]
    positive_option: str | None = None
    max_input_tokens: int | None = None

    def __post_init__(self) -> None:
        """Normalize the criteria mapping (a frozen dataclass still needs a dict)."""
        if self.criteria is None:
            object.__setattr__(self, "criteria", {})


def _cell_id(code: str, paragraph_number: str, question_number: int) -> str:
    """The id of one grid cell (the pairwise file's own pair-id convention)."""
    return f"{code}-{paragraph_number}-Q{question_number}"


def build_cells(
    view: DatasetView,
    *,
    scope: str,
    question_transform: str,
    question_block_markers: Sequence[str],
) -> tuple[DecisionCell, ...]:
    """The cells a probe scores: the reviewed set plus the ordinary cells.

    The ``judged`` scope scores the 1,520 reviewed pairs and the 4,000
    unjudged-proxy control cells; the ``grid`` scope scores every cell of the
    same-document grid. The reviewed set's label-0 pairs (the reviewer-verified
    hard negatives) and the ordinary cells (the control, or the unjudged grid)
    are **separate groups** — they are never pooled.
    """
    codes = {report["document"]: report["code"] for report in view.documents}
    questions = list(view.questions)
    transformed = apply_question_transform(
        questions, question_transform, tuple(question_block_markers)
    )
    question_texts = dict(zip(questions, transformed, strict=True))

    cells: list[DecisionCell] = []
    for record, cell in zip(view.pairs, view.judged_cells, strict=True):
        row = view.paragraph_row[record.paragraph_text]
        column = view.question_col[record.question_text]
        cells.append(
            DecisionCell(
                cell_id=cell["cell_id"],
                group=GROUP_JUDGED,
                document=cell["document"],
                paragraph_row=row,
                question_col=column,
                paragraph_text=record.paragraph_text,
                question_text=question_texts[record.question_text],
                label=int(record.label),
                paragraph_number=cell["paragraph_number"],
                question_number=int(cell["question_number"]),
            )
        )

    if scope == "grid":
        judged_cells = {(cell.paragraph_row, cell.question_col) for cell in cells}
        for grid in view.grids:
            for row in grid.paragraph_rows:
                for column in grid.question_cols:
                    if (row, column) in judged_cells:
                        continue
                    cells.append(
                        DecisionCell(
                            cell_id=_cell_id(
                                codes[grid.document],
                                view.paragraph_numbers[row],
                                view.question_numbers[column],
                            ),
                            group=GROUP_UNJUDGED,
                            document=grid.document,
                            paragraph_row=row,
                            question_col=column,
                            paragraph_text=view.paragraphs[row],
                            question_text=question_texts[view.questions[column]],
                            label=None,
                            paragraph_number=view.paragraph_numbers[row],
                            question_number=view.question_numbers[column],
                        )
                    )
        return tuple(cells)

    for document, row, column in view.control.cells:
        cells.append(
            DecisionCell(
                cell_id=_cell_id(
                    codes[document],
                    view.paragraph_numbers[row],
                    view.question_numbers[column],
                ),
                group=GROUP_CONTROL,
                document=document,
                paragraph_row=row,
                question_col=column,
                paragraph_text=view.paragraphs[row],
                question_text=question_texts[view.questions[column]],
                label=None,
                paragraph_number=view.paragraph_numbers[row],
                question_number=view.question_numbers[column],
            )
        )
    return tuple(cells)


def score_cells(
    decider: Decider,
    cells: Sequence[DecisionCell],
    arm: DecisionArm,
    *,
    max_questions_per_call: int = 64,
) -> tuple[list[float], list[bool], int]:
    """Score the cells with the arm: one probability (and one truncation flag) per cell.

    The call plan comes from the pure core: the ``pair`` shape makes one call
    per cell, the ``paragraph`` shape batches a paragraph's cells (up to
    ``max_questions_per_call`` questions per call, and within the model's
    input budget). A cell whose paragraph had to be cut to fit the budget is
    flagged — a shortened input is never silent.

    Returns:
        The probabilities, the truncation flags and the number of **planned**
        calls (the arm's cost — the calls the cache may serve).
    """
    calls = plan_calls(
        cells,
        state_shape=arm.state,
        instructions=arm.instructions,
        question_type=arm.question_type,
        options=arm.options,
        criteria=arm.criteria,
        max_questions_per_call=max_questions_per_call,
        max_input_tokens=arm.max_input_tokens,
    )
    probabilities: list[float | None] = [None] * len(cells)
    truncated: list[bool] = [False] * len(cells)
    for call in calls:
        answers = decider.decide(
            call.state, dict(zip(call.question_names, call.specs, strict=True))
        )
        for name, index in zip(call.question_names, call.cell_indexes, strict=True):
            answer = answers.get(name)
            if answer is None:
                raise RuntimeError(
                    f"the decision call returned no answer for the question {name!r}"
                )
            probabilities[index] = answer_probability(answer, arm.positive_option)
            state_paragraph = call.state.get("paragraph", "")
            truncated[index] = len(state_paragraph) < len(cells[index].paragraph_text)
    missing = [index for index, value in enumerate(probabilities) if value is None]
    if missing:
        raise RuntimeError(f"{len(missing)} cells were not scored by the decision model")
    return (
        [float(value) for value in probabilities],  # type: ignore[arg-type]
        truncated,
        len(calls),
    )


def run_decision_arm(
    view: DatasetView,
    decider: Decider,
    arm: DecisionArm,
    *,
    scope: str,
    question_transform: str,
    question_block_markers: Sequence[str],
    competence_control: bool = True,
    competence_pairs: int = 100,
    max_questions_per_call: int = 64,
    bins: int = 10,
) -> dict[str, Any]:
    """Run one arm: score the cells, measure them and measure the competence control."""
    cells = build_cells(
        view,
        scope=scope,
        question_transform=question_transform,
        question_block_markers=question_block_markers,
    )
    probabilities, truncated, n_calls = score_cells(
        decider, cells, arm, max_questions_per_call=max_questions_per_call
    )
    ordinary_group = GROUP_CONTROL if scope != "grid" else GROUP_UNJUDGED
    measurement = measure_decision_arm(
        cells,
        probabilities,
        ordinary_group=ordinary_group,
        bins=bins,
    )
    block: dict[str, Any] = {
        "arm": arm.name,
        "state": arm.state,
        "question_type": arm.question_type,
        "instructions": arm.instructions,
        "options": list(arm.options),
        "criteria": dict(arm.criteria),
        "positive_option": arm.positive_option,
        "max_input_tokens": arm.max_input_tokens,
        "n_calls_planned": n_calls,
        "scope": scope,
        "measurement": measurement,
        "truncation": {
            "n_cells": len(cells),
            "n_truncated": sum(truncated),
            "share": (sum(truncated) / len(cells)) if cells else None,
            "note": (
                "the cells whose paragraph was cut to fit the model's input "
                "budget (the question and the instructions are never cut): the "
                "model saw a shortened paragraph, which is recorded here and "
                "flagged per cell"
            ),
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "group": cell.group,
                "document": cell.document,
                "label": cell.label,
                "probability": probability,
                "paragraph_truncated": cut,
            }
            for cell, probability, cut in zip(cells, probabilities, truncated, strict=True)
        ],
    }

    if competence_control:
        competence = build_competence_cells(list(view.paragraphs), n_pairs=competence_pairs)
        competence_probabilities, competence_truncated, competence_calls = score_cells(
            decider, competence, arm, max_questions_per_call=max_questions_per_call
        )
        block["competence_control"] = {
            **measure_competence_control(competence, competence_probabilities),
            "construction": {
                "question_template": 'Does the document discuss "{phrase}"?',
                "positive": "the question quotes a phrase of the paragraph (a rare window)",
                "negative": (
                    "the question quotes a phrase of another paragraph that shares "
                    "no distinctive word with this one, so the pair is unrelated"
                ),
                "phrase_words": 4,
                "n_truncated": sum(competence_truncated),
                "n_calls_planned": competence_calls,
                "note": (
                    "the control is synthetic: it measures whether the model can "
                    "detect a shared distinctive phrase at all, which is what "
                    "separates 'the reviewed set is at the ceiling' from 'the model "
                    "is the wrong shape'"
                ),
            },
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "label": cell.label,
                    "probability": probability,
                    "paragraph_truncated": cut,
                }
                for cell, probability, cut in zip(
                    competence, competence_probabilities, competence_truncated, strict=True
                )
            ],
        }
    return block
