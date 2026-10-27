"""The UPR decision probe on the **full val grid** (the track's CS-9b).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-9b) closes the scope deviation the
CS-9 probe declared: the probe scored a **seeded sample** of the val split
(150 of the 660 texts) and left the full grid, the **safe-negative rule** and
the **shortlist rerank** to this follow-up.

**The scope.** `upr_analysis.decision.scope: grid` scores the whole
(text x label) grid of the val split — 660 texts x 68 labels = 44,880 cells —
on the **tev1** arms (the `paragraph` shape, the `noul` and the `choice`
types). The `paragraph` shape is the cheap one (2 calls per text against the
`pair` shape's 68), so the full grid costs ~1,320 calls per arm.

**The two new blocks** (both in the `decisions.json` artifact):

- the **safe-negative rule** in its three forms (the fixed 0.5 cut, the
  fit-side threshold, the margin rule) with the **label-blind counterpart**
  beside each. The rule's fit side is the **train** split, so no operating
  point is fitted on the evaluated split;
- the **shortlist rerank**: the champion's top-K labels per text, re-ordered by
  the decision model's own probability (the full grid is already scored, so
  the rerank costs no additional calls), with the shortlist's own recall as
  the ceiling.

**The yardsticks.** The label-blind counterpart (the same model, the same
prompts, the wrong pairing) beside every rule form, and the synthetic
competence control beside every arm.
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    ChunkerSpec,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprDecisionRuleConfig,
)
from cluster_rag.sweep import DecideUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The UPR-native **task** instruction (the same one the CS-9 probe used, so
# the pointwise arm's calls are cache hits and the two runs are paired).
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
TEV1_MAX_INPUT_TOKENS = 2500

# The arms: the question types of the champion model, on the **full val grid**.
# `laya` and the `pair` shape stay at their CS-9 samples (a second decision
# model is closed as a lever, and the `pair` shape's full grid is 44,880 calls
# per arm for a shape whose cost is already measured).
QUESTION_TYPES: tuple[str, ...] = ("noul", "choice")


def _model(question_type: str) -> DecisionModelConfig:
    """One arm's model config (the champion model, the paragraph shape)."""
    return DecisionModelConfig(
        name=f"tev1-paragraph-{question_type}",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type=question_type,
        instructions=INSTRUCTIONS,
        options=CHOICE_OPTIONS if question_type == "choice" else (),
        criteria=CHOICE_CRITERIA if question_type == "choice" else {},
        positive_option="yes" if question_type == "choice" else None,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


cells = [
    DecideUprCell(
        label=f"upr-decide-tev1-paragraph-{question_type}-grid",
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            decision=UprDecisionConfig(
                models=(_model(question_type),),
                scope="grid",
                seed=42,
                competence_control=True,
                competence_pairs=50,
                prompt_version="v1",
                host="",
                timeout=600.0,
                max_questions_per_call=64,
                bins=10,
                mismatched_control=True,
                rule=UprDecisionRuleConfig(enabled=True),
            ),
        ),
    )
    for question_type in QUESTION_TYPES
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_decision_grid")
