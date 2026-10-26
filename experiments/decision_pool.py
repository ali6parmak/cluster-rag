"""The decision models: card-verified configs shared by the track's sweep files.

The text-processing track's Stage 3.5 probes two **decision models** — Ollama's
new model *type*, which answers typed questions with probabilities in one
forward pass (no decoder, no generation):

- ``tev1:latest`` — a Qwen3.5-4B fine-tune by Together AI (4.61 GB);
- ``laya:latest`` — a ModernBERT-large fine-tune by Convai Innovations
  (0.84 GB).

Both are local and **pinned by digest**: the client reads the server's
``/api/tags`` and the run *fails* when the digest differs from the pinned one,
so a re-pulled model can never be measured under an old arm's name.

**The input budgets** were measured on 2026-10-26 (the plan's Stage 3.5): the
server refuses a prompt longer than the model's context, and the measured
limits are **512 tokens for laya** and **~2,539 tokens for tev1**. The corpus's
longest paragraph (2,255 characters) and longest question (1,560 characters)
do not fit in laya's budget, so the planner truncates the *state's paragraph*
(the question and the instructions are never cut) and the artifact records
every truncated cell.

The prompt is the track's **declared prompt grid** (the state's shape × the
question type), and every variant is a config value, so the artifact records
exactly what was asked.
"""

from __future__ import annotations

from cluster_rag.config import DecisionModelConfig

# The **task** instruction. The question's placement is the state shape's
# business (in the state for "pair", in the instructions for "paragraph"), so
# the task must not repeat the question: that would put it into the call twice
# — which biases the grid's comparison and breaks laya's input budget.
INSTRUCTIONS = "Is the paragraph about the same topic as the question?"

# The choice question's two options and their descriptions. The docs warn that
# a `noul` question can get "stuck on no", so the two-option choice is the
# other half of the prompt grid.
CHOICE_OPTIONS: tuple[str, ...] = ("yes", "no")
CHOICE_CRITERIA: dict[str, str] = {
    "yes": "the paragraph and the question are about the same topic",
    "no": "the paragraph and the question are about different topics",
}

# The pinned digests (read from the server on 2026-10-26).
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
LAYA_DIGEST = "1f196b378202837961e8856ceeb36af8b03c4b6158f6b02f15cc2b0a99b65e5a"

# The models' measured input budgets in tokens (the server's hard context).
TEV1_MAX_INPUT_TOKENS = 2500
LAYA_MAX_INPUT_TOKENS = 512

# The two models, by their handle.
DECISION_MODELS: dict[str, DecisionModelConfig] = {
    "tev1": DecisionModelConfig(
        name="tev1",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        instructions=INSTRUCTIONS,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    ),
    "laya": DecisionModelConfig(
        name="laya",
        model_id="laya:latest",
        digest=LAYA_DIGEST,
        instructions=INSTRUCTIONS,
        max_input_tokens=LAYA_MAX_INPUT_TOKENS,
    ),
}
