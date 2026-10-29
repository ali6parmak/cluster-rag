"""The track's **hit map** on the UPR val split (the track's CS-18).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-18) pre-registers the round: the
lexical feature's own per-word decomposition (Version A, exact and free) and
the decision model's sentence-level ablation (Version B, the only version that
perturbs the input and therefore the only one that says anything causal).

**The two arms.** The plan's own instruction is "measure the sentence count
before scheduling": on the val split 653 of the 660 texts (98.94%) are a single
sentence, so the sentence-level arm can ablate almost nothing (8 variants over
7 texts, and four of those seven are false splits at "decree No. 170"). The
plan's **amendment of 2026-10-29** therefore adds the **clause-level** arm
(the track's own atom chunker with the clause splitting on): 348 texts are
ablatable there, holding 799 gold cells. Both arms run; the sentence arm's
near-null is reported as a near-null.

**What the arms share.** The same text set (the whole val split, pre-registered
— never a sample chosen after looking at the results), the same pinned decision
model (the very same `tev1` the CS-13 feature used, so the map and the feature
are read against one model and the intact-text calls are that run's cache hits),
the same prompt, and the **label-blind counterpart** beside the ablation (the
yardstick that says whether the attribution is label-driven).

**The runs** (fixed in the plan's amendment, before the first call):

| run | the ablation unit | new calls |
|---|---|---|
| `upr-hitmap-sentence` | the plan's sentence unit | 32 |
| `upr-hitmap-clause` | the amendment's clause unit | 2,508 |
"""

from __future__ import annotations

from embedder_pool import MODELS  # noqa: F401  (the shared model pool stays importable)

from cluster_rag.config import (
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    UprHitMapConfig,
)
from cluster_rag.sweep import HitMapUprCell, define_sweep

CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

# The UPR-native **task** instruction (the same one the CS-9b grid run used, so
# the intact-text calls are that run's cache hits and the map and the feature
# are read against one prompt).
INSTRUCTIONS = "Is the text about the topic in the question?"

# The pinned digest (read from the server on 2026-10-26) and the model's
# measured input budget (the server's hard context).
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500

# The two arms: the plan's sentence unit and the amendment's clause unit.
UNITS: tuple[str, ...] = ("sentence", "clause")


def _model() -> DecisionModelConfig:
    """The decision model the map reads (the champion model, the paragraph shape).

    The map has no model of its own: it reads the CS-9b run's pinned arm, so the
    map and the CS-13 feature are read against the same model.
    """
    return DecisionModelConfig(
        name="tev1-paragraph-noul",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type="noul",
        instructions=INSTRUCTIONS,
        options=(),
        criteria={},
        positive_option=None,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


cells = [
    HitMapUprCell(
        label=f"upr-hitmap-{unit}",
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
                competence_control=False,
                competence_pairs=0,
                prompt_version="v1",
                host="",
                timeout=600.0,
                max_questions_per_call=64,
                bins=10,
                mismatched_control=True,
            ),
            hitmap=UprHitMapConfig(
                unit=unit,
                include_lexical_map=True,
                include_label_blind=True,
                catch_all_labels=("General", "Other"),
            ),
        ),
    )
    for unit in UNITS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_hitmap")
