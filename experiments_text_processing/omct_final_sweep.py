"""The OMCT **final** table: the baselines and the arms that worked.

Every other file in this directory is a **stage** sweep: each one answers one
question of the plan and measures a whole family of arms, so its table section
is a research record, not a result table. This file re-runs, in one sweep,
only the arms whose verdict was a **win**, so the meeting table is short and
every row is a finding.

**One cell per embedding model**, and each cell's run carries the whole story
for that model in one artifact:

- the **full-text cosine** (the run's reference arm) — the baseline; every
  other number is a difference against it.
  (log: "Notes on the Stage-0 numbers")
- the **paragraph chunked, peak kept** (`words25` + `max` / `peak:3:0.05`) —
  the run's structural arms; the Stage-2 verdict, +0.01 to +0.04
  within-document AUC on 2 of 3 models.
  (log: "Stage 2 — the verdict")
- the **late chunking** (`voyage-4-nano` only) — the run's late arms; the
  ladder's one winner, +0.0859 within-document AUC.
  (log: "Stage 2.7 — the late chunking")
- the **learned feature pack** — the run's `stage3` block; the pack beats the
  cosine by +0.10 to +0.15 within-document AUC.
  (log: "Stage 3.1 — the feature pack + the logistic combiner")
- the **decision model on the grid** — the run's decision arm; the track's
  best arm, 0.7701 within-document AUC.
  (log: "Stage 3.5.2 — the decision arm on the grid + the rule's decider")
- the **safe-negative rule** — the decision arm's `rule` block; the
  deliverable: the reach, the cost and the label-blind counterpart.
  (log: "Stage 3.5.2")
- the **shortlist rerank** — the decision arm's `shortlist_rerank` block;
  retrieve-then-rerank, the positives' hit@1 0.1565 against the cosine's
  0.1050. (log: "Stage 3.5.3 — the shortlist rerank")

**What is deliberately NOT in it.** The arms whose verdict was a loss: the
paragraph-side *atoms* (Stage 1), the *question-side* chunking and the
symmetric variants (Stage 2), the *chunk header* (Stage 2.6), the *semantic*
chunker's threshold (it does not transfer across documents), the *chunk
profile* and the *question-type* block (the random-vector floor says they are
not the lever), the *gradient-boosted* combiner (on OMCT it is below the
logistic on 8 of 9 arms) and the *sibling-max backoff*. Their runs and their
verdicts stay in the stage sweeps and in the track log; they are not here.

**The controls that ARE in it** are the ones that make a claim checkable: the
decision arm's **label-blind counterpart** (the same model, the wrong pairing)
rides along inside the grid arm, the **safe-negative rule** is measured in its
three forms beside it, and the pack's own **floors** (the label-permutation
chance floor, the control-role floor) ride along inside the `stage3` block. A
row without its control is not a finding.

**The scope.** OMCT is the **auxiliary** pairwise data: 1,520 reviewed pairs
(524 verified positives + 996 reviewer-verified hard negatives) and the seeded
4,000-cell control over the 42,947 unjudged cells. Nothing here is a UPR
metric; every number annotates a method choice. The **headline is the
within-document AUC** and the pooled AUC goes beside it, never alone (58.7% of
the pooled positive-negative pairs are cross-document). The two negative
subsets are never pooled: A is the reviewer-verified ground truth, B is the
unjudged control.

**The cost.** The embedding arms are cache hits (the models are warm in the
embedding cache) and the decision arms are cache hits (the decision models'
own disk cache, keyed by model id, digest, prompt version, state and question
specs — so the prompt and the pinned digest must not change). The decision
arms still need the Ollama server running: the client reads `/api/tags` and
**fails** on a digest mismatch, before it consults the cache.

**How to run it.**

    uv run cluster-rag sweep --file experiments_text_processing/omct_final_sweep.py

The rows land in `docs/experiments_table_text_processing_omct.md` under the
section ``## OMCT — final_results`` (auto-appended; never hand-edited).
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

# The three Stage-0 models: the cosine's reference and every embedding arm's
# model. The configs come from the shared pool, so the run ids, the model ids,
# the prefixes and the pinned revisions are the main pipeline's.
MODELS_3: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: the seeded, proportional sample of the unjudged
# same-document cells. The same size and the same seed as every stage sweep,
# so **every run in this sweep draws exactly the same 4,000 control cells**
# and the control side is comparable across the whole table.
CONTROL_SIZE = 4000

# The paired bootstrap of the arm-vs-baseline AUC difference (the plan's
# Stage-2.5 statistic: without it a +/-0.03 AUC effect is unresolvable).
BOOTSTRAP = PairedBootstrapConfig(resamples=2000, seed=42)

# The Stage-2 verdict's paragraph-side window (25 words, stride 12) and the
# two **peak-keeping** rules. The diluted rules (`top_k_mean`, `decayed_top_k`)
# and the question-side chunking are excluded on purpose: their verdict was a
# loss or a wash.
WINDOW_PARAMS: dict[str, object] = {"n_words": 25, "stride": 12}
WINDOWS = ChunkerSpec(name="word_windows", params=dict(WINDOW_PARAMS))
PEAK_RULES: tuple[str, ...] = ("max", "peak:3:0.05")

# The late chunking needs **token-level** embeddings, which Ollama's
# /api/embed cannot return, so the arm is **HF-only** and only `voyage-4-nano`
# qualifies. The comparison is therefore **within-model** (the late arm against
# the early arm of the same run), never against the other two models' numbers.
HF_MODEL = "voyage-4-nano"

# The feature pack's paragraph chunker: the same 25-word window, so the pack
# and the structural arms are scored on the same geometry.
FEATURE_CHUNKER = ChunkerSpec(name="word_windows", params=dict(WINDOW_PARAMS))

# The pack's groups: the v0 set. The chunk-profile block is **excluded** —
# the random-vector floor says it is the embedding geometry re-measured, not a
# new signal (the OMCT verdict; the UPR one agrees).
V0_GROUPS: tuple[str, ...] = ("embedding", "lexical", "question_part", "question_type")

# The combiner: the sklearn logistic regression. On OMCT the gradient-boosted
# combiner is *below* the logistic on 8 of the 9 arms ("the learner is not the
# lever" here), so the logistic is the one that ships on this dataset.
COMBINER = CombinerSpec(name="logistic", params={"C": 1.0, "max_iter": 1000, "standardize": True})

# The control cells' training weight: the control is *presumed* negative (its
# true positive rate is unmeasured), so it is down-weighted and cannot
# dominate the fit. Never ground truth.
CONTROL_WEIGHT = 0.3

# The decision arm: tev1 with the two-option `choice`, in the cheap
# `paragraph` shape (339 calls against the `pair` shape's 5,520). The prompt
# version is part of the decision cache's key, so it must not change or the
# runs stop being cache hits.
PROMPT_VERSION = "v1"
DECISION_MODEL = "tev1"
DECISION_STATE = "paragraph"
DECISION_QUESTION_TYPE = "choice"


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


def _stage3_block() -> Stage3Config:
    """The learned pack's block (the pack, the combiner, the floors, the rule)."""
    return Stage3Config(
        features=FeaturePackConfig(
            paragraph_chunker=FEATURE_CHUNKER,
            question_chunker=None,
            groups=V0_GROUPS,
        ),
        combiner=COMBINER,
        control_weight=CONTROL_WEIGHT,
        reporting=Stage3ReportingConfig(enabled=True),
    )


cells = [
    AnalyzeOmctCell(
        label=f"analyze-final-{model}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=model, embed=MODELS[model]),),
            control_size=CONTROL_SIZE,
            # the structural arms: the paragraph chunked, the peak kept
            # ("chunk the paragraph, never the question")
            chunker=WINDOWS,
            question_chunker=None,
            aggregations=PEAK_RULES,
            # the late-chunked arms: the same windows, but the document is
            # embedded once and its token-level embeddings are pooled per
            # chunk. HF-only.
            late_chunker=WINDOWS if model == HF_MODEL else None,
            bootstrap=BOOTSTRAP,
            # the learned pack (+ its floors and the reporting blocks)
            stage3=_stage3_block(),
            # the decision arm on the whole grid: the best scorer, the
            # safe-negative rule's three forms with the label-blind
            # counterpart beside each, and the shortlist rerank.
            decision=DecisionConfig(
                models=(_decision_arm(),),
                scope="grid",
                competence_control=False,
                mismatched_control=True,
                prompt_version=PROMPT_VERSION,
            ),
        ),
    )
    for model in MODELS_3
]

SWEEP = define_sweep(cells, dataset="omct", label="final_results")
