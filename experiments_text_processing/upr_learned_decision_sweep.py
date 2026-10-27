"""The decision probability as a **pack feature** on UPR (the track's CS-13).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-13) adds one more column to the
UPR pack's ``(n_texts, n_labels, n_features)`` cube: the decision model's
probability for the (text, label) cell.

**The hypothesis.** The pack's lexical block finds five of the eight unseen
labels because the **name-vs-text** mechanism is identical for a seen and an
unseen label, so its fitted weight transfers. The decision probability has
the same property — it is computed from the label's **name**, identically for
a held-out label — so its fitted weight should transfer too. The fit side has
**zero** positives for the eight unseen labels (0 of 3,087 train gold cells),
so the combiner must **transfer** the weight: that is the hypothesis, and it
is what to measure.

**The arms** are the two learners (the plan's CS-6b defers `lightgbm` until
sklearn's HistGB is beaten, and HistGB is already a core dep and beats the
logistic on UPR), each with the real feature and with the **label-blind**
counterpart (the same model, the same prompts, the wrong pairing — the
yardstick that says how much of the feature's contribution is the model's own
label-blind signal).

The control is `upr-analyze-learned-cat-control` (the same pack and the same
combiner **without** the feature), so every comparison is paired over the
texts (the tool `tools/compare_upr_arms.py`).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionFeatureConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
    UprLearnedReportingConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
DECISION_PACK: tuple[str, ...] = (*FULL_PACK, "decision")

# The pinned decision model (the CS-9 probe's champion model: the paragraph
# shape, the `noul` question type) and its measured input budget.
TEV1_DIGEST = "7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740"
TEV1_MAX_INPUT_TOKENS = 2500
INSTRUCTIONS = "Is the text about the topic in the question?"


def _decision_model() -> DecisionModelConfig:
    """The decision model whose probability becomes the pack's feature."""
    return DecisionModelConfig(
        name="tev1-paragraph-noul",
        model_id="tev1:latest",
        digest=TEV1_DIGEST,
        state="paragraph",
        question_type="noul",
        instructions=INSTRUCTIONS,
        max_input_tokens=TEV1_MAX_INPUT_TOKENS,
    )


def _learned(combiner: str, *, label_blind: bool) -> UprLearnedConfig:
    """The CS-8f grid's best arm plus the decision feature."""
    return UprLearnedConfig(
        enabled=True,
        groups=DECISION_PACK,
        top_k=3,
        chunker=ATOMS,
        profile_top_k=3,
        profile_ngram_size=5,
        combiner=CombinerSpec(name=combiner),
        control_weight=0.3,
        floors=UprLearnedFloorsConfig(
            chance=True,
            control_role=True,
            random_vectors=True,
            permutation_resamples=5,
            random_seed=42,
        ),
        reporting=UprLearnedReportingConfig(
            enabled=True,
            bins=10,
            threshold_quantiles=99,
        ),
        decision_feature=UprDecisionFeatureConfig(
            enabled=True,
            model=_decision_model(),
            prompt_version="v1",
            max_questions_per_call=64,
            label_blind=label_blind,
        ),
        seed=42,
    )


# The arms: (the run's label, the combiner, the label-blind counterpart).
ARMS: tuple[tuple[str, str, bool], ...] = (
    ("upr-analyze-learned-dec-histgb", "histgb", False),
    ("upr-analyze-learned-dec-logistic", "logistic", False),
    ("upr-analyze-learned-dec-blind-histgb", "histgb", True),
    ("upr-analyze-learned-dec-blind-logistic", "logistic", True),
)

cells = [
    AnalyzeUprCell(
        label=label,
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
        k=CHAMPION_K,
        split="val",
        embedding_run=CHAMPION_EMBEDDING_RUN,
        label_embedding_run=CHAMPION_LABEL_EMBEDDING_RUN,
        embed=MODELS["embeddinggemma"],
        analysis=UprAnalysisConfig(
            arms=(),
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
            learned=_learned(combiner, label_blind=label_blind),
        ),
    )
    for label, combiner, label_blind in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_decision")
