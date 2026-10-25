"""Stage-3 sweep: the feature pack + the logistic combiner (CS-5).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.1) replaces the hand-written
scoring rules with a **table of numbers per pair** (the feature pack v0) and a
**learned scorer** over it (the logistic regression), evaluated behind the
plan's **document-level holdout**: for every document the combiner is fitted
on the other two documents and predicts the held-out third, and the
out-of-fold scores of every judged pair and of every control cell form one
score matrix.

**Why one cell per model and not one per (model, rotation).** The out-of-fold
protocol needs all three rotations of a model *in the same run*: every judged
pair and every control cell must be scored by the model that never saw its
document, and the three rotations' scores form one matrix. A run per
rotation could only hold a third of the matrix. So each cell is one model
and holds its three rotations.

**The training data** (the reviewer's settled decision): the fit side's
judged pairs — the verified positives and the reviewer-verified hard
negatives — plus the fit side's *unjudged control cells* as extra presumed
negatives, down-weighted (weight 0.3: the control's true positive rate is
unmeasured, so it must not dominate the fit). The control is never presented
as ground truth.

**The feature pack** (v0, the plan's section-2 list minus the length
feature, which the reviewer's decision leaves out because length is a
document proxy on this data): the global and best chunk cosines, the top-K
chunk similarities and their spread, the soft lexical overlap and the char
n-grams, the question-part coverage and the question-type one-hots. The
paragraph side is chunked with the Stage-2 verdict's window (25 words,
stride 12) and the question side stays whole ("chunk the paragraph, never
the question").

The models are the three Stage-0 models (`embeddinggemma`, `voyage-4-nano`,
`arctic2`); their configs come from the shared pool
(`experiments/embedder_pool.py`), so the track's runs use the same model ids,
prefixes and pinned revisions as the main pipeline's sweeps. The control set
is model-independent and seeded, so every run draws **exactly the same
control cells** as the Stage-0, Stage-1 and Stage-2 runs.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    ChunkerSpec,
    CombinerSpec,
    FeaturePackConfig,
    PairedBootstrapConfig,
    Stage3Config,
)
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The Stage-0 models (the same three the Stage-0, Stage-1 and Stage-2
# artifacts used).
STAGE3_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells (the plan's default). The same size and seed as the
# Stage-0 runs, so the control cells are the same cells.
CONTROL_SIZE = 4000

# The feature pack's paragraph chunker: the Stage-2 verdict's paragraph-side
# window (25 words, stride 12). The question side stays whole.
FEATURE_CHUNKER = ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12})

# The K of the top-K chunk similarities (the spread and the peak margin are
# measured over the same K best chunk cosines).
TOP_K = 3

# The combiner: the sklearn logistic regression (the Stage-3.1 baseline).
COMBINER = CombinerSpec(name="logistic", params={"C": 1.0, "max_iter": 1000, "standardize": True})

# The control cells' training weight: the control is *presumed* negative
# (its true positive rate is unmeasured), so it is down-weighted and cannot
# dominate the fit.
CONTROL_WEIGHT = 0.3

# The label-permutation floor: the shuffled labels must land the AUC at
# chance.
PERMUTATION_RESAMPLES = 5

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            bootstrap=BOOTSTRAP,
            stage3=Stage3Config(
                features=FeaturePackConfig(
                    paragraph_chunker=FEATURE_CHUNKER,
                    question_chunker=None,
                    top_k=TOP_K,
                ),
                combiner=COMBINER,
                control_weight=CONTROL_WEIGHT,
                permutation_resamples=PERMUTATION_RESAMPLES,
            ),
        ),
    )
    for model in STAGE3_MODELS
]

SWEEP = define_sweep(cells, dataset="omct", label="stage3_features")
