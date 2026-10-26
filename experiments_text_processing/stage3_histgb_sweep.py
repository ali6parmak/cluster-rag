"""Stage-3.2 sweep: the gradient-boosted combiner (CS-6b).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.2) puts a **gradient-boosted**
learner next to the Stage-3.1 logistic regression, behind the same
``Combiner`` port: scikit-learn's ``HistGradientBoostingClassifier`.

**Why the question is worth asking, given Stage 3.1b.** The feature-group
ablation measured that the learned arm's gain is the **lexical** block's (a
bag-of-words overlap over the question's stem), not the embedding geometry's
and not the question-type prior's. So the bar for a better learner is
explicit: it has to beat a *three-feature* logistic regression (0.6461
within-document AUC on all three models) and it must not cost more hit@1. A
tree ensemble can only earn something if the signal is not linear in the
features or if the blocks interact.

**The grid.** The gradient booster on three feature-group sets, on the three
Stage-0 models:

- `all` — the whole pack (15 features): the same pack the Stage-3.1 and
  Stage-3.1b arms used, so the learner is the only thing that changes;
- `lex-only` — the lexical block alone (3 features): the block the
  Stage-3.1b ablation identified as the carrier of the gain, and the block a
  tree ensemble has the least room to improve on (3 features, mostly
  monotone);
- `embed-only` — the embedding block alone (6 features): the block the
  ablation found to add *no* label-driven signal, i.e. the block where a
  better learner would have to find something the logistic could not.

The **logistic partners are the Stage-3.1b runs** (`omct-analyze-<model>-
stage3-{all,lex-only,embed-only}`): the same protocol, the same control
cells, the same rotations and the same seeds, so the two learners' arms are
read on the same footing.

**The hyperparameters are scikit-learn's defaults, pinned explicitly** (the
sweep spells them out, so the frozen config and the artifact record the
effective values). They are *not* tuned on this data: the track has already
measured that a fit-side selection did not transfer (the Stage-2 semantic
threshold: the fit side's choice did not predict the holdout's), so a
fit-side grid would buy a number that does not generalise.

**The floors and the statistics are the Stage-3 ones, unchanged**: the
document-level holdout (three rotations, one out-of-fold score matrix), the
training data (the fit side's judged pairs plus its unjudged control cells as
down-weighted presumed negatives), the two floors (the label-permutation
chance floor and the control-role floor) and the paired statistics against
the full-text cosine.

The models come from the shared pool (`experiments/embedder_pool.py`), so the
track's runs use the same model ids, prefixes and pinned revisions as the main
pipeline's sweeps. The control set is model-independent and seeded, so every
run draws **exactly the same control cells** as the Stage-0, Stage-1,
Stage-2, Stage-3.1 and Stage-3.1b runs.
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

# The Stage-0 models (the same three the Stage-0, Stage-1, Stage-2, Stage-3.1
# and Stage-3.1b artifacts used).
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

# The combiner: the gradient-boosted learner (the Stage-3.2 subject). The
# hyperparameters are scikit-learn's defaults, pinned explicitly so the
# frozen config and the artifact record them. Early stopping is off: the fit
# side's rows are the fit side's rows, and the fit stays deterministic.
COMBINER = CombinerSpec(
    name="histgb",
    params={
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_depth": None,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 42,
    },
)

# The control cells' training weight: the control is *presumed* negative
# (its true positive rate is unmeasured), so it is down-weighted and cannot
# dominate the fit.
CONTROL_WEIGHT = 0.3

# The label-permutation floor: the shuffled labels must land the AUC at
# chance.
PERMUTATION_RESAMPLES = 5

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The feature-group sets of the grid (the Stage-3.1b vocabulary, so the
# arms' names match their logistic partners).
GROUP_SETS: dict[str, tuple[str, ...]] = {
    "all": ("embedding", "lexical", "question_part", "question_type"),
    "lex-only": ("lexical",),
    "embed-only": ("embedding",),
}

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3-histgb-{name}",
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

SWEEP = define_sweep(cells, dataset="omct", label="stage3_histgb")
