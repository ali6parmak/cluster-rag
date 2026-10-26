"""Stage-3 reporting sweep (CS-6e).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 3, "Reporting") asks for *"the
calibration curves + thresholds per document; pooled and per-document macro;
both negative subsets separately"*, and section 7's **CS-6e** adds the
**safe-negative margin** measured against the **control-role floor** — the
track's durable deliverable (section 8: *"a measured 'which unjudged pairs
are safe negatives' rule"*).

**What this sweep measures.** The Stage-3 runs so far reported the *arm's
quality* (the AUCs, the floors, the hit@k). This sweep adds the *reporting*:
the reliability diagram of the arm's probability and of the full-text cosine
(on a probability scale, by a fit-side calibration map), the **per-document
threshold** (chosen on the fit side, applied to the held-out document) and
the **safe-negative margin rule** (a cell is a safe negative when it sits far
enough below its own paragraph's best candidate) with its reach over the
whole unjudged grid and its harm on the verified positives, beside the same
numbers for the **control-role floor**.

**The grid.**

- the three Stage-0 models, so the reporting sits next to the Stage-3
  artifacts of the same models;
- the two feature-group sets that could ship: `base` (the v0 pack's four
  groups) and `full` (the v0 pack plus the chunk profile). The `profile`
  group set (the chunk profile alone) is left out: the Stage-3.3 and CS-6d
  verdicts already established that the block is a re-measurement of the
  embedding geometry and never the lever, so it cannot be the shipped rule;
- the two **fit-side criteria** of the per-document threshold: `floor_gap`
  (the verified positives against the unjudged control cells — the rule's
  target negative class) and `youden` (the verified positives against the
  reviewer-verified hard negatives — the reviewed set's own ground truth).
  The reviewer asked for the criterion to be settled by measurement, so both
  run: the artifact records each document's fit-side choice and its holdout's
  operating point, and the log says which criterion transfers.

Twelve runs: 3 models x 2 group sets x 2 criteria.

**Nothing else changes.** The same document-level holdout (three rotations,
one out-of-fold score matrix), the same training data (the fit side's judged
pairs plus its unjudged control cells as down-weighted presumed negatives),
the same two floors, the same paired statistics against the full-text cosine
and the same combiner (the logistic regression — the learner is not the
lever, CS-6b). The fit side's own rows are scored by an **inner**
document-level cross-fit, so the threshold's and the margin's choices never
rest on a model that saw the holdout.

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
    Stage3ReportingConfig,
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

# The combiner: the sklearn logistic regression (the Stage-3.1 baseline).
COMBINER = CombinerSpec(name="logistic", params={"C": 1.0, "max_iter": 1000, "standardize": True})

# The control cells' training weight: the control is *presumed* negative
# (its true positive rate is unmeasured), so it is down-weighted and cannot
# dominate the fit.
CONTROL_WEIGHT = 0.3

# The label-permutation floor: the shuffled labels must land the AUC at
# chance. The control-role floor (the same permutation, the control cells
# keeping their role) is the yardstick the rule is measured against.
PERMUTATION_RESAMPLES = 5

# The paired bootstrap of the arm-vs-baseline AUC difference.
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The v0 pack's four groups (the Stage-3.1 feature set) and the Stage-3.3
# chunk-profile group.
V0_GROUPS: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")
PROFILE_GROUP: tuple[str, ...] = ("chunk_profile",)

# The group sets of the sweep: the two arms that could ship the rule.
GROUP_SETS: dict[str, tuple[str, ...]] = {
    "base": V0_GROUPS,
    "full": V0_GROUPS + PROFILE_GROUP,
}

# The two fit-side criteria of the per-document threshold (the reviewer asked
# for the criterion to be settled by measurement, so both run).
THRESHOLD_CRITERIA: tuple[str, ...] = ("floor_gap", "youden")

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{model}-stage3-report-{name}-{criterion}",
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
                embeddings="model",
                reporting=Stage3ReportingConfig(
                    enabled=True,
                    threshold_criterion=criterion,
                ),
            ),
        ),
    )
    for model in STAGE3_MODELS
    for name, groups in GROUP_SETS.items()
    for criterion in THRESHOLD_CRITERIA
]

SWEEP = define_sweep(cells, dataset="omct", label="stage3_reporting")
