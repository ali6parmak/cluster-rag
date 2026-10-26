"""Stage-3.5 decision probe sweep (CS-7a).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.5.1) asks the question the whole
track has been circling: **is the reviewed set learnable at all?** Every
embedder sits at chance on it (AUC 0.468 / 0.500 / 0.537) and the reviewed set
is **rank-selected** (exp18's model chose which pairs got reviewed), so those
numbers may be measuring the selection rather than the labels. A **decision
model** — a frozen, zero-shot, probability-answering scorer — is the first
*independent* ranker on this benchmark.

**The grid** (the plan's *declared prompt grid*, two axes, both config values):

- the **state's shape**: `pair` (the paragraph and the question in one state)
  against `paragraph` (the paragraph alone as the state, with the question in
  the instructions — the shape that allows up to 64 questions per call and is
  the cheap one);
- the **question type**: `noul` (a boolean statement, answered with the
  probability that it is true) against `choice` (a two-option pick, because
  the docs warn that a `noul` question can get "stuck on no").

Two models (both local, both pinned by digest): `tev1` (a Qwen3.5-4B fine-tune
by Together AI) and `laya` (a ModernBERT-large fine-tune by Convai
Innovations). Eight runs: 2 models x 2 state shapes x 2 question types.

**The competence control is not optional.** A chance-level result on the
reviewed set is ambiguous: *the data is at the ceiling* or *the model is the
wrong shape*. The synthetic control (constructed pairs that share a distinctive
phrase against pairs that share none) separates the two, so it runs beside
every arm.

**The scope is `judged`**: the 1,520 reviewed pairs plus the seeded
unjudged-proxy control sample (the same 4,000 cells the Stage-0 runs draw, so
the control side is comparable). The `grid` scope (the whole 44,467-cell
same-document grid) is Stage 3.5.2's job (CS-7b), where the decision arm
joins the `analyze-omct` arms.

**The prompt is a config value, not a code constant**: the template, the state
shape, the question type and the digests all live in the config (the shared
`experiments/decision_pool.py` for the model specs), and every artifact
records them.
"""

from __future__ import annotations

from decision_pool import CHOICE_CRITERIA, CHOICE_OPTIONS, DECISION_MODELS

from cluster_rag.config import (
    AnalysisConfig,
    DecisionConfig,
    DecisionModelConfig,
)
from cluster_rag.sweep import DecideOmctCell, define_sweep

# The control size: the seeded, proportional sample of the unjudged
# same-document cells (the plan's default) — the same cells the Stage-0,
# Stage-1, Stage-2 and Stage-3 runs draw, so the control side is comparable.
CONTROL_SIZE = 4000

# The prompt version: part of the decision cache's key and of the artifact's
# provenance (a re-prompted arm never reads the old arm's answers).
PROMPT_VERSION = "v1"

# The two state shapes of the grid.
STATES: tuple[str, ...] = ("pair", "paragraph")

# The two question types of the grid.
QUESTION_TYPES: tuple[str, ...] = ("noul", "choice")


def _arm(model: str, state: str, question_type: str) -> DecisionModelConfig:
    """One arm's model config: the shared model's spec plus the grid's axes."""
    base = DECISION_MODELS[model]
    if question_type == "choice":
        return base.model_copy(
            update={
                "state": state,
                "question_type": "choice",
                "options": CHOICE_OPTIONS,
                "criteria": CHOICE_CRITERIA,
                "positive_option": CHOICE_OPTIONS[0],
            }
        )
    return base.model_copy(update={"state": state, "question_type": "noul"})


cells = [
    DecideOmctCell(
        label=f"decide-{model}-{state}-{question_type}",
        analysis=AnalysisConfig(
            control_size=CONTROL_SIZE,
            question_transform="strip_block",
            decision=DecisionConfig(
                models=(
                    _arm(model, state, question_type).model_copy(
                        update={"name": f"{model}-{state}-{question_type}"}
                    ),
                ),
                scope="judged",
                competence_control=True,
                competence_pairs=100,
                prompt_version=PROMPT_VERSION,
            ),
        ),
    )
    for model in ("tev1", "laya")
    for state in STATES
    for question_type in QUESTION_TYPES
]

SWEEP = define_sweep(cells, dataset="omct", label="stage35_decision")
