"""Stage-3.5.3 sweep (CS-7c): the shortlist rerank.

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.5.3): the cosine's **top-K
candidates** per paragraph are re-decided by the decision model, and the
metric is the reviewed set's positives' **hit@1 / hit@K** — the number
PLAN's Phase-6 reranker slot needs. The decision models are not trained on
OMCT, so nothing leaks (the leakage rule (i) is about *fitted* question-level
priors; a frozen model's question text carries no OMCT label).

**What it adds** to the CS-7b run: the rerank's block. The shortlist is the
cosine's own top-K, the rerank reorders it by the decision model's
probability, and the numbers sit between the two references:

- the **cosine's own hit@1** (the geometry alone);
- the **decision model's own hit@1** (the decision model alone, i.e.
  reranking everything);
- the **reranked hit@1** (the hybrid), with the shortlist's own recall as
  the ceiling a rerank cannot beat.

The numbers come from the grid the decision arm already scored, so the sweep
costs **no new model calls**.

**The arm.** `tev1` with the two-option `choice` — the CS-7a verdict's
winner (the best within-document AUC, 0.7767, and the best calibration) — in
the `paragraph` shape (the cheap one: 339 calls against 5,520 for the
`pair` shape, 16x cheaper for 0.006 of AUC, which decides the grid scope's
cost).

**The runs.** Three runs, one per Stage-0 model: the embedding model is the
cosine's reference and the learned arm's rule, the decision arm is the same
in all three (it does not depend on an embedder). The Stage-3 block is on,
so the artifact carries the learned arm's rule and the decision model's rule
side by side.

**The cost.** The decision grid is 44,467 cells: ~1,400 calls for the arm
and ~1,400 for its label-blind counterpart, in the `paragraph` shape. The
decision models' own cache makes the second and third run's grid free.
"""

from __future__ import annotations

from decision_pool import CHOICE_CRITERIA, CHOICE_OPTIONS, DECISION_MODELS
from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    ChunkerSpec,
    CombinerSpec,
    DecisionConfig,
    FeaturePackConfig,
    PairedBootstrapConfig,
    Stage3Config,
    Stage3ReportingConfig,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The Stage-0 models: the cosine's reference and the learned arm's rule.
STAGE3_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells — the same cells the Stage-0 runs draw.
CONTROL_SIZE = 4000

# The feature pack's paragraph chunker: the Stage-2 verdict's paragraph-side
# window (25 words, stride 12). The question side stays whole.
FEATURE_CHUNKER = ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12})

# The combiner: the sklearn logistic regression (the Stage-3.1 baseline).
COMBINER = CombinerSpec(name="logistic", params={"C": 1.0, "max_iter": 1000, "standardize": True})

# The control cells' training weight: the control is *presumed* negative, so
# it is down-weighted and cannot dominate the fit.
CONTROL_WEIGHT = 0.3

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The v0 pack's four groups (the Stage-3.1 feature set).
V0_GROUPS: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")

# The decision arm: the CS-7a winner (tev1, the two-option choice, the
# cheap `paragraph` shape).
DECISION_MODEL = "tev1"
DECISION_STATE = "paragraph"
DECISION_QUESTION_TYPE = "choice"
PROMPT_VERSION = "v1"


def _decision_arm():
    """The decision arm's model config (the shared spec plus the grid's axes)."""
    base = DECISION_MODELS[DECISION_MODEL]
    return base.model_copy(
        update={
            "state": DECISION_STATE,
            "question_type": DECISION_QUESTION_TYPE,
            "options": CHOICE_OPTIONS,
            "criteria": CHOICE_CRITERIA,
            "positive_option": CHOICE_OPTIONS[0],
        }
    )


cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage35-rerank",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            bootstrap=BOOTSTRAP,
            stage3=Stage3Config(
                features=FeaturePackConfig(
                    paragraph_chunker=FEATURE_CHUNKER,
                    question_chunker=None,
                    groups=V0_GROUPS,
                ),
                combiner=COMBINER,
                control_weight=CONTROL_WEIGHT,
                reporting=Stage3ReportingConfig(enabled=True),
            ),
            decision=DecisionConfig(
                models=(_decision_arm(),),
                scope="grid",
                competence_control=False,
                mismatched_control=True,
                prompt_version=PROMPT_VERSION,
            ),
        ),
    )
    for model in STAGE3_MODELS
]

SWEEP = define_sweep(cells, dataset="omct", label="stage35_rerank")
