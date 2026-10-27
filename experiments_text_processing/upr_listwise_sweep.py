"""The **listwise** second decision stage on UPR (the track's CS-12).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-12) adds a **second round** to the
decision probe: round 1 is the existing 68 pointwise `noul` questions per
text (unchanged), and round 2 is **one** call per text asking "which of these
20 topics is the text about?", the 20 options being the round-1 top-20 and
each option's `criteria` its **catalog description** (the CS-11 catalog).

The hypothesis: the decision model is excellent at **finding** the candidates
and much weaker at **ordering** them — and the ordering is where the
pipeline's `top_k k=2` lives. The arm's final ranking is a **rank fusion**
(the round-2 order over the candidates, then the round-1 order for the rest),
so nothing is fitted on the evaluated split.

**The two arms** are the mechanism and its control:

- `upr-decide-tev1-paragraph-noul-listwise` — the criteria carry the
  catalog's **descriptions** (the main arm);
- `upr-decide-tev1-paragraph-noul-listwise-nodesc` — the criteria carry the
  label's **own name** (the control that isolates the *listwise mechanism*
  from the *descriptions*).

Both run on the **full val grid** (`scope: grid`), beside the **label-blind
counterpart** (the wrong pairing, the same pipeline) and the synthetic
competence control.
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    ChunkerSpec,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprListwiseConfig,
)
from cluster_rag.sweep import DecideUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The UPR-native **task** instruction (the CS-9 probe's, so the pointwise
# round's calls are cache hits and the two runs are paired).
INSTRUCTIONS = "Is the text about the topic in the question?"

# The pinned digest (read from the server on 2026-10-26) and the model's
# measured input budget (the server's hard context).
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500

# The arms: the criteria's wording (the catalog's descriptions against the
# label's own name).
ARMS: tuple[tuple[str, str], ...] = (
    ("upr-decide-tev1-paragraph-noul-listwise", "catalog"),
    ("upr-decide-tev1-paragraph-noul-listwise-nodesc", "name"),
)


def _model() -> DecisionModelConfig:
    """The champion model (the paragraph shape, the `noul` question type)."""
    return DecisionModelConfig(
        name="tev1-paragraph-noul",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type="noul",
        instructions=INSTRUCTIONS,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


cells = [
    DecideUprCell(
        label=label,
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            decision=UprDecisionConfig(
                models=(_model(),),
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
                listwise=UprListwiseConfig(
                    enabled=True,
                    candidates=20,
                    criteria=criteria,
                    prompt_version="v2",
                ),
            ),
        ),
    )
    for label, criteria in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_listwise")
