"""Stage-3.1b sweep: the feature-group ablation (CS-6a).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.1b) asks the two questions the
Stage-3.1 run left open: **which block of the feature pack carries the
gain**, and **is the learned arm's worse hit@1 (the retrieval loss) the
question-type block's doing**.

**The unit of the ablation is the feature group.** The v0 pack's 15 features
fall into four groups, each one a block of the same kind of signal:

- `embedding` (6): the full-text cosine and the five chunk features;
- `lexical` (3): the soft lexical overlap in both directions and the char
  n-grams;
- `question_part` (1): the answer options' soft coverage;
- `question_type` (5): the question-type one-hots (options present, a "no
  data" option, a yes/no question, a frequency question, the option count).

The pack computes every feature; the **combiner sees only the selected
groups' columns** (`analysis.stage3.features.groups`). Nothing else changes:
the same document-level holdout (three rotations, one out-of-fold score
matrix), the same training data (the fit side's judged pairs plus its
unjudged control cells as down-weighted presumed negatives), the same two
floors (the label-permutation chance floor and the control-role floor) and
the same paired statistics against the full-text cosine.

**The grid.** Eight group sets per model:

- `all` — the whole pack: the reference, and the run that must reproduce the
  Stage-3.1 numbers field by field (the same config, the same seeds);
- `no-qtype`, `no-qpart`, `no-lex`, `no-embed` — the leave-one-out grid:
  the block whose removal costs the most carries the gain;
- `embed-only`, `lex-only`, `qtype-only` — the single-block arms: what a
  block achieves on its own. `qtype-only` is a **question-level prior** by
  construction (its features do not look at the paragraph at all), so it is
  the arm that shows how much of the gain is a prior which would not
  transfer to a new corpus; the run records how many of the holdout's
  questions the fit side also judged, so the number can be read against it.

Three models (the Stage-0 models), so 24 runs. Every run is one model and
one group set, and it holds all three rotations — the out-of-fold protocol
needs them together.

The models come from the shared pool (`experiments/embedder_pool.py`), so the
track's runs use the same model ids, prefixes and pinned revisions as the
main pipeline's sweeps. The control set is model-independent and seeded, so
every run draws **exactly the same control cells** as the Stage-0, Stage-1,
Stage-2 and Stage-3.1 runs.
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

# The Stage-0 models (the same three the Stage-0, Stage-1, Stage-2 and
# Stage-3.1 artifacts used).
STAGE3_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells (the plan's default). The same size and seed as the
# earlier runs, so the control cells are the same cells.
CONTROL_SIZE = 4000

# The feature pack's paragraph chunker: the Stage-2 verdict's paragraph-side
# window (25 words, stride 12). The question side stays whole.
FEATURE_CHUNKER = ChunkerSpec(name="word_windows", params={"n_words": 25, "stride": 12})

# The K of the top-K chunk similarities (the spread and the peak margin are
# measured over the same K best chunk cosines).
TOP_K = 3

# The combiner: the sklearn logistic regression (the Stage-3.1 baseline) —
# the ablation changes the features, not the learner.
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

# The feature-group sets of the ablation: the leave-one-out grid plus the
# single-block arms, with the whole pack as the reference.
GROUP_SETS: dict[str, tuple[str, ...]] = {
    "all": ("embedding", "lexical", "question_part", "question_type"),
    "no-qtype": ("embedding", "lexical", "question_part"),
    "no-qpart": ("embedding", "lexical", "question_type"),
    "no-lex": ("embedding", "question_part", "question_type"),
    "no-embed": ("lexical", "question_part", "question_type"),
    "embed-only": ("embedding",),
    "lex-only": ("lexical",),
    "qtype-only": ("question_type",),
}

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3-{name}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            bootstrap=BOOTSTRAP,
            stage3=Stage3Config(
                features=FeaturePackConfig(
                    paragraph_chunker=FEATURE_CHUNKER,
                    question_chunker=None,
                    top_k=TOP_K,
                    groups=groups,
                ),
                combiner=COMBINER,
                control_weight=CONTROL_WEIGHT,
                permutation_resamples=PERMUTATION_RESAMPLES,
            ),
        ),
    )
    for model in STAGE3_MODELS
    for name, groups in GROUP_SETS.items()
]

SWEEP = define_sweep(cells, dataset="omct", label="stage3_ablation")
