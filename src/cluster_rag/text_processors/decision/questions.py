"""The decision models' question specs (the track's Stage 3.5, pure core).

A **decision model** answers *typed questions* about a *state*:

- ``noul`` — a boolean statement ("is the paragraph about the same topic as
  the question?"), answered with the probability that it is true;
- ``choice`` — a pick from 2-26 named options, answered with the picked option
  and the per-option probabilities;
- ``score`` — a rubric, answered with a score and the per-level probabilities.

The track's prompt grid declares two axes (TEXT_PROCESSING_PLAN.md section 3,
Stage 3.5.1): the **state's shape** (``pair`` = the pair in one state, vs.
``paragraph`` = the paragraph alone as the state, with the question in the
instructions — the shape that allows 64 questions per call) and the **question
type** (``noul`` vs. a two-option ``choice``, because the docs warn that
``noul`` can get "stuck on no").

**Where the question lives is the shape's business, not the task's.** The
config carries the *task* instruction ("is the paragraph about the same topic
as the question?"), and the two shapes place the question differently: the
``pair`` shape puts it in the state (next to the paragraph), the ``paragraph``
shape puts it in the instructions (in front of the task). A task instruction
that repeats the question is therefore rejected: it would put the question
into the call **twice**, which both biases the comparison and (for the smaller
model) breaks its input budget — the 512-token model cannot even take the
longest paragraph and question twice.

This module is the pure side of that: the question spec (the payload the
client sends), the prompt's rendering and the spec's validation. It imports
nothing but the standard library — no model client, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# The question types the track uses.
QUESTION_TYPE_NOUL = "noul"
QUESTION_TYPE_CHOICE = "choice"
QUESTION_TYPES: tuple[str, ...] = (QUESTION_TYPE_NOUL, QUESTION_TYPE_CHOICE)

# The prompt's placeholder for the question's text. A **task** instruction
# must not carry it (the question's placement is the shape's business).
QUESTION_PLACEHOLDER = "{question}"

# The question block the ``paragraph`` shape puts in front of the task. The
# label is structural (the shape's own convention), not part of the task.
QUESTION_BLOCK_TEMPLATE = "Question: {question}\n\n"

# The API's limits on a choice question's options.
MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 26


@dataclass(frozen=True)
class QuestionSpec:
    """One typed question of a decision call: the payload the client sends.

    The spec is **rendered**: its ``instructions`` already carry the question's
    text (the shape's rendering happens before the spec is built), so the
    client never has to know about the prompt.
    """

    type: str
    instructions: str
    options: tuple[str, ...] = ()
    criteria: Mapping[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """The question's JSON payload (what the ``/v1/systemone`` body carries)."""
        payload: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.type == QUESTION_TYPE_CHOICE:
            payload["options"] = list(self.options)
            payload["criteria"] = dict(self.criteria)
        return payload

    def describe(self) -> dict[str, Any]:
        """A JSON-serializable description of the spec (the artifact's provenance)."""
        return {
            "type": self.type,
            "instructions": self.instructions,
            "options": list(self.options),
            "criteria": dict(self.criteria),
        }


def render_instructions(task: str, question_text: str, *, shape: str = "paragraph") -> str:
    """Render the prompt's instructions for one question, given the state's shape.

    - the ``pair`` shape puts the question in the **state**, so the
      instructions are the task alone;
    - the ``paragraph`` shape puts the question in the **instructions**, in
      front of the task.

    Raises:
        ValueError: If the task carries the question's placeholder (the
            question's placement is the shape's business) or the shape is
            unknown.
    """
    if QUESTION_PLACEHOLDER in task:
        raise ValueError(
            f"the task instruction must not carry the {QUESTION_PLACEHOLDER!r} "
            "placeholder: the state's shape decides where the question goes (in "
            "the state for 'pair', in the instructions for 'paragraph')"
        )
    if shape == "pair":
        return task
    if shape == "paragraph":
        return QUESTION_BLOCK_TEMPLATE.replace(QUESTION_PLACEHOLDER, question_text) + task
    raise ValueError(f"unknown decision state shape {shape!r}; expected 'pair' or 'paragraph'")


def build_question_spec(
    question_type: str,
    instructions: str,
    *,
    options: tuple[str, ...] = (),
    criteria: Mapping[str, str] | None = None,
) -> QuestionSpec:
    """Build one question spec (the rendered instructions plus the type's fields).

    Raises:
        ValueError: If the type is unknown, the instructions are empty, or a
            choice question's options and criteria do not match.
    """
    if question_type not in QUESTION_TYPES:
        raise ValueError(
            f"unknown decision question type {question_type!r}; expected one of "
            f"{list(QUESTION_TYPES)}"
        )
    if not instructions.strip():
        raise ValueError("a decision question needs non-empty instructions")
    resolved = dict(criteria or {})
    if question_type == QUESTION_TYPE_CHOICE:
        if not MIN_CHOICE_OPTIONS <= len(options) <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"a choice question needs {MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS} options, "
                f"got {len(options)}"
            )
        if set(resolved) != set(options):
            raise ValueError(
                "a choice question's criteria must describe exactly its options: "
                f"criteria {sorted(resolved)} vs options {sorted(options)}"
            )
        return QuestionSpec(
            type=question_type,
            instructions=instructions,
            options=tuple(options),
            criteria=resolved,
        )
    if options or resolved:
        raise ValueError(f"a {question_type!r} question takes no options and no criteria")
    return QuestionSpec(type=question_type, instructions=instructions)
