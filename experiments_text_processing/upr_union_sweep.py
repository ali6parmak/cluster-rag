"""The **union** candidate generator as a number (the track's CS-14).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-14) measures the union — the
per-row **z-max** of the champion's and the decision model's scores — and
nothing else. The union is **not built** and enters no pipeline: the
2026-10-22 trim put "union shortlists" out of the track's menu (they are
main-plan Phase 4/6 machinery), so this run produces a **number for PLAN**.

The run reuses the CS-9b grid's arm (the whole val grid, tev1, the paragraph
shape, the `noul` type), so the decision calls are cache hits and the only
new work is the block itself.
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

INSTRUCTIONS = "Is the text about the topic in the question?"
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500

cells = [
    DecideUprCell(
        label="upr-decide-tev1-paragraph-noul-grid-union",
        k=2,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            decision=UprDecisionConfig(
                models=(
                    DecisionModelConfig(
                        name="tev1-paragraph-noul",
                        model_id="tev1:latest",
                        digest=TEV1_DIGEST,
                        state="paragraph",
                        question_type="noul",
                        instructions=INSTRUCTIONS,
                        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
                    ),
                ),
                scope="grid",
                seed=42,
                competence_control=False,
                prompt_version="v1",
                host="",
                timeout=600.0,
                max_questions_per_call=64,
                bins=10,
                mismatched_control=True,
                union=True,
            ),
        ),
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_union")
