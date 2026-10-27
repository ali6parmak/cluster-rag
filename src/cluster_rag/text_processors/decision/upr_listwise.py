"""The **listwise** second decision stage (the text-processing track's CS-12).

The pointwise arm asks 68 boolean questions per text ("is this text about
*<label>*?") and the probe's finding is that the decision model is a very
good **candidate finder** and a much weaker **ranker** — and the ranking is
where the pipeline's ``top_k k=2`` lives.

The listwise stage is a **second round** on top of the pointwise arm:

- **round 1** (unchanged): the 68 pointwise questions, whose scores are the
  candidate list (the round-1 top-N labels per text);
- **round 2**: **one** call per text asking "which of these 20 topics is the
  text about?", the 20 options being the round-1 top-20 and each option's
  ``criteria`` its **catalog description** (the CS-11 catalog) or, in the
  control variant, the label's own name.

The arm's final ranking is a **rank fusion** (parameter-free, so nothing is
fitted on the evaluated split): the round-2 order over the 20 candidates,
then the remaining 48 in the round-1 order. The arm's score matrix is the
negative final rank, so every rank-based metric (the decision view's top-2,
the nDCG, the recall@k, the shortlist recall) is well defined. The
**calibration** and the **probability compression** are explicitly **not
applicable** to this arm: a within-text distribution over 20 options is not a
per-cell probability.

Pure core: numpy + the standard library only (the track's decision-side
rule — no model client, no I/O, no global state).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.text_processors.decision.probe import (
    CHARS_PER_TOKEN,
    DecisionCall,
    truncate_to_budget,
)
from cluster_rag.text_processors.decision.questions import (
    QUESTION_TYPE_CHOICE,
    build_question_spec,
    render_instructions,
)

# The listwise question's default wording (the config can override it).
LISTWISE_QUESTION_TEXT = "Which of these topics is the text about?"

# The listwise task's default wording (the config can override it).
LISTWISE_TASK = "Pick the topic that the text is about."

# The state shape of the listwise stage: the text alone is the state (the
# question's options travel in the question spec).
LISTWISE_STATE = "paragraph"


def generic_criteria(label: str) -> str:
    """The **name-only** criterion of an option (the control variant).

    The control isolates the *listwise mechanism* from the *descriptions*: the
    same 20 options, the same question, but the criteria carry the label's
    own name instead of the catalog's description.

    Args:
        label: The label's name.

    Returns:
        The criterion's wording.
    """
    return f"the text is about {label}"


def build_listwise_spec(
    options: Sequence[str],
    criteria: Mapping[str, str],
    *,
    task: str = LISTWISE_TASK,
    question_text: str = LISTWISE_QUESTION_TEXT,
) -> Any:
    """The second round's one question: the choice over the candidates.

    Args:
        options: The candidate labels (2-26 of them; the API's limit).
        criteria: The per-option wording (exactly the options' names).
        task: The task instruction (the question's placement is the shape's
            business, so the task must not repeat the question).
        question_text: The question's wording.

    Returns:
        The :class:`~cluster_rag.text_processors.decision.questions.QuestionSpec`
        of the call.

    Raises:
        ValueError: If the options and the criteria do not match (the
            question spec's own contract).
    """
    rendered = render_instructions(task, question_text, shape=LISTWISE_STATE)
    return build_question_spec(
        QUESTION_TYPE_CHOICE,
        rendered,
        options=tuple(options),
        criteria=dict(criteria),
    )


def plan_listwise_calls(
    texts: Sequence[str],
    option_sets: Sequence[Sequence[str]],
    criteria_sets: Sequence[Mapping[str, str]],
    *,
    task: str = LISTWISE_TASK,
    question_text: str = LISTWISE_QUESTION_TEXT,
    max_input_tokens: int | None = None,
) -> list[DecisionCall]:
    """Plan the second round's calls: **one** call per text.

    Each call carries its own options and criteria (the candidates differ per
    text), so the pointwise planner cannot be reused. The **input budget**
    counts the options and the criteria as well as the instructions and the
    state: a 20-option call with 20 descriptions is much larger than the
    pointwise one, and a call that silently exceeded the model's context
    would be a silent truncation of the *question*.

    Args:
        texts: The texts (one call each).
        option_sets: The candidate labels of every text.
        criteria_sets: The per-option wording of every text.
        task: The task instruction.
        question_text: The question's wording.
        max_input_tokens: The model's input budget (``None`` = no limit).

    Returns:
        The planned calls (one per text).

    Raises:
        ValueError: If the texts and the option sets disagree.
    """
    if not (len(texts) == len(option_sets) == len(criteria_sets)):
        raise ValueError(
            "the texts, the option sets and the criteria sets must have the same "
            f"length, got {len(texts)}, {len(option_sets)} and {len(criteria_sets)}"
        )
    calls: list[DecisionCall] = []
    for index, text in enumerate(texts):
        spec = build_listwise_spec(
            option_sets[index],
            criteria_sets[index],
            task=task,
            question_text=question_text,
        )
        paragraph = text
        if max_input_tokens is not None:
            budget = max_input_tokens * CHARS_PER_TOKEN
            fixed = len(spec.instructions) + sum(len(option) for option in spec.options)
            fixed += sum(len(value) for value in spec.criteria.values())
            paragraph = truncate_to_budget(text, int(budget - fixed))
        calls.append(
            DecisionCall(
                state={"paragraph": paragraph},
                cell_indexes=(index,),
                question_names=(f"t{index}",),
                specs=(spec,),
            )
        )
    return calls


def top_candidates(scores: np.ndarray, n_candidates: int) -> list[tuple[int, ...]]:
    """Every text's round-1 top-N candidates (the columns, best first).

    Args:
        scores: The round-1 ``(n_texts, n_labels)`` score matrix.
        n_candidates: How many candidates per text.

    Returns:
        The candidate columns of every text.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"the score matrix must be 2-dimensional, got {matrix.shape}")
    n_candidates = max(1, min(int(n_candidates), matrix.shape[1]))
    order = np.argsort(-matrix, axis=1, kind="stable")
    return [tuple(int(column) for column in row[:n_candidates]) for row in order]


def fuse_rankings(
    round_one: np.ndarray,
    round_two: np.ndarray,
) -> np.ndarray:
    """The rank fusion: the round-2 order over the candidates, then round 1.

    The fused score matrix holds the **negative final rank** of every cell, so
    the harness's rank-based metrics (the decision view's top-k, the nDCG, the
    recall@k, the shortlist recall) read the fused order directly. The
    cells that are **not** candidates of their text (the round-2 matrix holds
    ``NaN`` there) keep the round-1 order below the candidates.

    Args:
        round_one: The round-1 ``(n_texts, n_labels)`` score matrix.
        round_two: The round-2 probabilities (``NaN`` outside the candidates).

    Returns:
        The fused score matrix (the same shape).

    Raises:
        ValueError: If the two matrices' shapes disagree.
    """
    first = np.asarray(round_one, dtype=np.float64)
    second = np.asarray(round_two, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"the two rounds must share one shape, got {first.shape} and {second.shape}"
        )
    fused = np.zeros(first.shape, dtype=np.float64)
    for row in range(first.shape[0]):
        candidates = np.flatnonzero(~np.isnan(second[row]))
        rest = np.flatnonzero(np.isnan(second[row]))
        rest_order = rest[np.argsort(-first[row, rest], kind="stable")]
        if candidates.size:
            candidates = candidates[np.argsort(-second[row, candidates], kind="stable")]
        order = np.concatenate([candidates, rest_order])
        fused[row, order] = -np.arange(order.size, dtype=np.float64)
    return fused


def listwise_measurement_note() -> dict[str, Any]:
    """The artifact's note on the two blocks that do not apply to the arm."""
    return {
        "applicable": False,
        "note": (
            "not applicable to the listwise arm: the round-2 answer is a "
            "within-text distribution over 20 options, not a per-cell "
            "probability, so neither the calibration nor the probability "
            "compression has a meaning here. The arm's score matrix is the "
            "negative final rank."
        ),
    }
