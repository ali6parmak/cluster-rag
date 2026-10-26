"""Stage-3.3 sweep: the chunk-profile features (CS-6c).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3.3) asks for the **chunk-profile
features** folded into the feature pack: the ranked cross-chunk match profile
(exp17's lineage) and the alignment kinds (exp16's lineage). The Stage-3.1b
ablation and the Stage-3.2 learner comparison left the features as the only
lever: the gain is the **lexical** block's, the embedding geometry adds no
label-driven signal, and a better learner does not help.

**The new block** (`chunk_profile`, the pack's fifth group):

- the **ranked cross-chunk match profile** — both sides of a pair are cut
  into sentences and into word pieces, every piece×piece cosine is computed
  and the top ranks are kept (`cprof_sent_top1..3`, `cprof_ngram_top1..3`).
  It says *where* the paragraph and the question match and how peaked the
  match is, not how similar they are on average;
- the **alignment kinds** (`cprof_align_sent` / `cprof_align_ngram`,
  `cprof_align_lexical` / `cprof_align_semantic_only`) — which granularity
  produced the best alignment and whether the two aligned units share a
  content word.

exp16's third feature, `log(1+df)` of the matched phrase **in the labeled
corpus**, is **not** in the pack: it is a corpus-level statistic, and the
pack's contract is pair-local (the reviewer's decision, 2026-10-26).

**The grid.** Three group sets per model:

- `base` — the v0 pack's four groups: the **control**. It must reproduce the
  Stage-3.1b `all` arm's numbers field for field (the same config, the same
  seeds, the same control cells), which is the check that the new block
  changed nothing in the old features;
- `full` — the v0 pack plus the new block: the new features *in the full
  pack*;
- `profile` — the new block alone: the new features *measured alone*.

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

# The chunk profile's own knobs: how many ranks of the cross-chunk profile
# are kept per granularity, and the nominal size of the word pieces.
PROFILE_TOP_K = 3
PROFILE_NGRAM_SIZE = 5

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

# The v0 pack's four groups (the Stage-3.1 feature set) and the Stage-3.3
# chunk-profile group.
V0_GROUPS: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")
PROFILE_GROUP: tuple[str, ...] = ("chunk_profile",)

# The group sets of the sweep: the control (the v0 pack), the full pack (the
# v0 pack plus the new block) and the new block alone.
GROUP_SETS: dict[str, tuple[str, ...]] = {
    "base": V0_GROUPS,
    "full": V0_GROUPS + PROFILE_GROUP,
    "profile": PROFILE_GROUP,
}

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3-cp-{name}",
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
            ),
        ),
    )
    for model in STAGE3_MODELS
    for name, groups in GROUP_SETS.items()
]

SWEEP = define_sweep(cells, dataset="omct", label="stage3_chunk_profile")
