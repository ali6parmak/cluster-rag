"""The UPR decision probe: the decision models on the UPR data (CS-9).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-9) puts the **decision models** on
the UPR data as a **measurement**, not a pipeline component: a decision model
(Ollama's model *type*: a state plus typed questions, answered with
probabilities in one forward pass) answers the UPR-native question **"is this
text about *<label>*?"** per label, and the probe measures what its
probability buys.

**The prompt grid** (the track's declared grid, the UPR question type):

- the **question type**: `noul` (a statement) and `choice` (the two-option
  "yes"/"no"), because the docs warn that a `noul` question can get "stuck on
  no";
- the **state shape**: `paragraph` (the text alone is the state, the question
  is in the instructions — the shape that allows up to 64 questions per call)
  and `pair` (the text and the question in one state — one call per cell, so
  it is measured on the **sample** only: the full val grid would need ~44,900
  calls per arm).

**The scope.** The probe measures a **seeded sample** of the val split's texts
(the same texts for every arm, so the arms are paired): the full val grid is
~44,880 cells, and the decision models cost one call per ~64 cells in the
`paragraph` shape (~1,320 calls per model) and one call per cell in the `pair`
shape (~44,880 calls).

**The yardsticks.** Beside every arm: the **synthetic competence control**
(without it a chance-level result is ambiguous between "the data is at the
ceiling" and "the model is the wrong shape") and the **label-blind
counterpart** (the same model, the same prompts, every text's label list
replaced by a deterministic shift) — the yardstick that turns the cost into a
quality claim.
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    ChunkerSpec,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
)
from cluster_rag.sweep import DecideUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The UPR-native **task** instruction. The question's placement is the state
# shape's business, so the task must not repeat the question (that would put
# it into the call twice — which biases the grid's comparison and breaks the
# smaller model's input budget).
INSTRUCTIONS = "Is the text about the topic in the question?"

# The choice question's two options and their descriptions.
CHOICE_OPTIONS: tuple[str, ...] = ("yes", "no")
CHOICE_CRITERIA: dict[str, str] = {
    "yes": "the text is about the topic in the question",
    "no": "the text is not about the topic in the question",
}

# The pinned digests (read from the server on 2026-10-26) and the models'
# measured input budgets (the server's hard context).
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
LAYA_DIGEST = "1f196b378202837961e8856ceeb36af8b03c4b6158f6b02f15cc2b0a99b65e5a"
TEV1_MAX_INPUT_TOKENS = 2500
LAYA_MAX_INPUT_TOKENS = 512

# The probe's arms: the model x the question type x the state shape. The
# `pair` shape is one call per (text, label) cell, so it is measured on a
# small sample (the shape's cost and its sanity, not its grid numbers).
ARMS: tuple[tuple[str, str, str, int], ...] = (
    ("tev1", "noul", "paragraph", 150),
    ("tev1", "choice", "paragraph", 150),
    ("laya", "noul", "paragraph", 150),
    ("laya", "choice", "paragraph", 150),
    ("tev1", "choice", "pair", 20),
)


def _model(
    name: str,
    question_type: str,
    state: str,
    max_input_tokens: int,
) -> DecisionModelConfig:
    """One model's config for one arm of the grid."""
    return DecisionModelConfig(
        name=f"{name}-{state}-{question_type}",
        model_id=f"{name}:latest",
        digest=TEV1_DIGEST if name == "tev1" else LAYA_DIGEST,
        state=state,
        question_type=question_type,
        instructions=INSTRUCTIONS,
        options=CHOICE_OPTIONS if question_type == "choice" else (),
        criteria=CHOICE_CRITERIA if question_type == "choice" else {},
        positive_option="yes" if question_type == "choice" else None,
        max_input_tokens=max_input_tokens,
    )


cells = [
    DecideUprCell(
        label=f"upr-decide-{name}-{state}-{question_type}",
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            decision=UprDecisionConfig(
                models=(
                    _model(
                        name,
                        question_type,
                        state,
                        TEV1_MAX_INPUT_TOKENS if name == "tev1" else LAYA_MAX_INPUT_TOKENS,
                    ),
                ),
                scope="sample",
                sample_size=sample_size,
                seed=42,
                competence_control=True,
                competence_pairs=50,
                prompt_version="v1",
                host="",
                timeout=600.0,
                max_questions_per_call=64,
                bins=10,
                mismatched_control=True,
            ),
        ),
    )
    for name, question_type, state, sample_size in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_decision")
