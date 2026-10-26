"""Stage-3 random-vector floor sweep (CS-6d).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3, "Sanity floors") asks for **two**
floors, so that a "win" can be checked against chance: the **label-permutation**
floor (done in CS-5: the fit side's labels are destroyed) and the
**random-vector** floor — *the arm is fitted on random vectors instead of the
model's embeddings, so a "win" can be checked against what the training design
alone buys*.

**Why this question is worth asking now.** The Stage-3.1b ablation showed the
learned arm's gain is the **lexical** block's, and the embedding geometry adds
no label-driven signal beyond the training design's own prior. The Stage-3.3
chunk profile is **entirely** embedding-derived. The random floor is the
missing measurement: it puts noise in the same feature slots as the real
embeddings and asks how much of an arm's number survives.

**What the floor replaces.** The feature pack's **embedding sides** only (the
full-text cosine, the chunk cosines and the chunk profile's units). The
pack's lexical, question-part and question-type features never used an
embedding, so they stay real — that is what "random vectors instead of the
model's embeddings" means. The run's **baseline arm** (the full-text cosine
the arm is compared against) keeps the model's embeddings, so the track's guard
(`tools/verify_analysis_baseline.py`) still holds and the arm-vs-cosine
comparison stays meaningful.

**The random vectors.** A seeded random unit vector of the **model's
dimension** per (kind, text): `sha256(seed | kind | text)` seeds a
`numpy.random.default_rng`, the vector is drawn from `uniform(-1, 1)` and
L2-normalized. The seed is the run's own `config.seed` (42), the recipe is
recorded in the artifact, and the vectors are never written to the embedding
cache (they must not pollute the model's cache).

**The grid.** The same three group sets as the Stage-3.3 sweep, so every
CS-6c arm gets its random-vector floor and the two runs can be compared with
`tools/compare_stage3_arms.py` (a paired per-pair bootstrap of the
within-document AUC difference):

- `base` — the v0 pack's four groups: the random floor of the Stage-3.1 pack;
- `full` — the v0 pack plus the chunk profile: the random floor of the full
  pack;
- `profile` — the chunk profile **alone**: the random floor of the block that
  is entirely embedding-derived. Its features are pure noise, so the arm
  must sit at chance — the sharpest statement available about that block.

Three models (the Stage-0 models), so 9 runs. Every run holds all three
rotations — the out-of-fold protocol needs them together.

Nothing else changes: the same document-level holdout (three rotations, one
out-of-fold score matrix), the same training data (the fit side's judged
pairs plus its unjudged control cells as down-weighted presumed negatives),
the same two floors (the label-permutation chance floor and the control-role
floor), the same paired statistics against the full-text cosine and the same
combiner (the logistic regression — the learner is not the lever, CS-6b).

The models come from the shared pool (`experiments/embedder_pool.py`), so the
track's runs use the same model ids, prefixes and pinned revisions as the main
pipeline's sweeps. The control set is model-independent and seeded, so every
run draws **exactly the same control cells** as the Stage-0, Stage-1, Stage-2
and Stage-3 runs.
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
# Stage-3 artifacts used).
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

# The chunk profile's own knobs: how many ranks of the cross-chunk profile are
# kept per granularity, and the nominal size of the word pieces.
PROFILE_TOP_K = 3
PROFILE_NGRAM_SIZE = 5

# The combiner: the sklearn logistic regression (the Stage-3.1 baseline) —
# the floor changes the features' space, not the learner.
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

# The v0 pack's four groups (the Stage-3.1 feature set) and the Stage-3.3
# chunk-profile group.
V0_GROUPS: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")
PROFILE_GROUP: tuple[str, ...] = ("chunk_profile",)

# The group sets of the sweep: the same three as the Stage-3.3 sweep, so
# every CS-6c arm has its random-vector floor.
GROUP_SETS: dict[str, tuple[str, ...]] = {
    "base": V0_GROUPS,
    "full": V0_GROUPS + PROFILE_GROUP,
    "profile": PROFILE_GROUP,
}

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3-rand-{name}",
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
                    profile_top_k=PROFILE_TOP_K,
                    profile_ngram_size=PROFILE_NGRAM_SIZE,
                ),
                combiner=COMBINER,
                control_weight=CONTROL_WEIGHT,
                permutation_resamples=PERMUTATION_RESAMPLES,
                embeddings="random",
            ),
        ),
    )
    for model in STAGE3_MODELS
    for name, groups in GROUP_SETS.items()
]

SWEEP = define_sweep(cells, dataset="omct", label="stage3_random_floor")
