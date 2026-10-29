"""The UPR **final** table: the baselines and the arms that worked.

Every other file in this directory is a **stage** sweep: each one answers one
question of the plan and measures a whole family of arms, so its table section
is a research record, not a result table. This file re-runs, in one sweep,
only the arms whose verdict was a **win**, so the meeting table is short and
every row is a finding.

**The scope: the ``val`` split only.** Nothing here touches ``test``. The
``test`` split stays reserved for the final adopted config, and the one test
run in the repo belongs to the main pipeline's Phase-4 model selection, not
to this track.

- the **whole-text champion** (the reference arm of every run) — the baseline;
  every other number is a difference against it.
  (log: "CS-8a — the UPR side: the command, the plumbing, the background")
- the **chunked retrieval** (`retr_atoms`, `retr_words15`) — the ladder's win:
  **+0.109 to +0.127 micro-F1 on the 300+ character texts** and exactly zero
  on the texts that are not split. (log: "CS-8b — the segmentation ladder")
- the **pure zero-shot geometry** (`pair_words15`) — the champion's kNN and
  prototype terms push **12.7 points of gold cells out of the top-50** that
  this geometry keeps, so the champion is a worse *candidate generator* than
  the geometry it is built on. (log: "CS-8b", the pair-scoring family)
- the **learned feature pack** (the gradient-boosted combiner) — a worse
  *classifier* (0.4818 against the champion's 0.5479) and a much better
  *candidate generator* (the shortlist R@50 0.8909); it also finds **five of
  the eight** held-out labels. (log: "CS-8d", "CS-8e")
- the **pack + the label catalog's channels** — the description channel is a
  real, label-driven win: **+0.0258 [+0.0091, +0.0433]** of micro-F1 over the
  same pack without it. (log: "CS-11 — the label catalog")
- the **pack + the decision probability** — the track's largest UPR result:
  **0.5965** (the champion's 0.5479), the unseen bucket **0.5697** against
  the champion's 0.0000. (log: "CS-13 — the decision probability as a feature")
- the **pack + the label-blind decision probability** — the control that says
  how much of the feature is the model's own label reading: **93%**.
  (log: "CS-13")

**What is deliberately NOT in it.** The arms whose verdict was a loss: the
*chunk header*, the *late chunking* (a null on UPR: the texts are too short
for a document's context to matter), the *listwise* second stage, the
*sibling-max backoff*, the *parent-category* features (near-empty on their
own), the **pure threshold** decision rule, the *symmetric* label side
(degenerate: the labels are 1-3 words) and the zero-shot rerank round (a null
in progress, and its log section does not exist yet). Their runs and their
verdicts stay in the stage sweeps and in the track log; they are not here.

**The controls that ARE in it** are the ones that make a claim checkable: the
**label-blind counterpart** of the decision feature (the same model, the wrong
pairing), the pack's own **floors** (the label-permutation chance floor, the
control-role floor, the random-vector floor) and the **unseen bucket** (the
eight labels held out of train, which no geometry can suggest). A row without
its control is not a finding.

**What is in the artifacts but NOT in this table's rows.** The decision
model's **own** arm (the micro-F1 0.5502, the shortlist R@50 0.9987), the
**safe-negative rule** and the **union**'s recall curve belong to the
``decide-upr`` command family, which has its own table shape (28 columns, not
this one's 25). Their rows are in the ``## UPR — upr_decision_grid`` and
``## UPR — upr_union`` sections of the same table file, and the decision
model's probability is the *input* to the last row of this table.

**The cost.** The embedding runs are cache hits (the models are warm in the
embedding cache) and the decision calls are cache hits (the decision models'
own disk cache, keyed by model id, digest, prompt version, state and question
specs — so the prompt and the pinned digest must not change). The decision
feature still needs the Ollama server running: the client reads ``/api/tags``
and **fails** on a digest mismatch, before it consults the cache.

**How to run it.**

    uv run cluster-rag sweep --file experiments_text_processing/upr_final_sweep.py

The rows land in ``docs/experiments_table_text_processing_upr.md`` under the
section ``## UPR — final_results`` (auto-appended; never hand-edited).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprArmConfig,
    UprDecisionFeatureConfig,
    UprLearnedConfig,
    UprLearnedFloorsConfig,
    UprLearnedReportingConfig,
)
from cluster_rag.sweep import AnalyzeUprCell, define_sweep

# The standing champion (PLAN.md Phase 4): embeddinggemma, the fusion of the
# three retrieval signals, the kNN's k=10 and the `top_k k=2` decision.
CHAMPION_PREDICTOR = "fusion"
CHAMPION_PARAMS: dict[str, object] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
CHAMPION_K = 2
CHAMPION_EMBEDDING_RUN = "upr-embed-embeddinggemma"
CHAMPION_LABEL_EMBEDDING_RUN = "upr-labelembed-embeddinggemma-about"

# The run whose numbers the whole-text reference arm must reproduce (the
# track's guard: the reference arm IS the main pipeline's own scoring).
CHAMPION_EVAL_RUN = "upr-eval-fusionc-embeddinggemma-k10-c1"

# The text-length buckets (characters): the axis the chunking hypothesis is
# about on UPR — a pooled number hides the long-text tail.
LENGTH_BUCKETS: tuple[int, ...] = (100, 200, 300)

# The pack's chunker: the **atoms** (the CS-8b ladder's best chunker on UPR —
# the sentence boundaries beat the word windows, the opposite of the OMCT
# verdict).
ATOMS = ChunkerSpec(name="atoms", params={"split_clauses": True})

# The ladder's two window sizes: 15 words (the arm that carries the biggest
# long-text gain) and the atoms (the ladder's best chunker). The 25- and
# 40-word windows and the sentence and semantic chunkers are excluded: their
# verdicts were a wash or a null.
WORD_WINDOWS_15 = ChunkerSpec(name="word_windows", params={"n_words": 15, "stride": 7})

# The pack's feature groups. The **chunk profile** is in the base pack (it is
# the CS-8e grid's best arm) and the **catalog-gated** groups
# (`label_part`, `label_type`) are the CS-11 win; the **decision** group is
# the CS-13 win.
FULL_PACK: tuple[str, ...] = ("embedding", "lexical", "chunk_profile")
CATALOG_PACK: tuple[str, ...] = (*FULL_PACK, "label_part", "label_type")
DECISION_PACK: tuple[str, ...] = (*FULL_PACK, "decision")

# The pinned decision model (the CS-9 probe's champion: the paragraph shape,
# the `noul` question type) and its measured input budget. The digest is part
# of the decision cache's key, so it must not change or the runs stop being
# cache hits.
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


def _learned(groups: tuple[str, ...], *, label_blind: bool = False) -> UprLearnedConfig:
    """The CS-8f grid's best arm (the gradient-boosted combiner), with the
    pack's groups and, optionally, the decision feature."""
    return UprLearnedConfig(
        enabled=True,
        groups=groups,
        top_k=3,
        chunker=ATOMS,
        profile_top_k=3,
        profile_ngram_size=5,
        combiner=CombinerSpec(name="histgb"),
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
            enabled="decision" in groups,
            model=_decision_model() if "decision" in groups else None,
            prompt_version="v1",
            max_questions_per_call=64,
            label_blind=label_blind,
        ),
        seed=42,
    )


# The reduced ladder: the two retrieval arms that carry the long-text gain and
# the one pair-scoring arm that carries the shortlist contrast. The full
# ladder (13 families, 153 arms) is in `upr_ladder_sweep.py`.
LADDER_ARMS: tuple[UprArmConfig, ...] = (
    UprArmConfig(
        name="retr_atoms",
        chunker=ATOMS,
        aggregations=("max",),
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
    ),
    UprArmConfig(
        name="retr_words15",
        chunker=WORD_WINDOWS_15,
        aggregations=("max",),
        predictor=CHAMPION_PREDICTOR,
        params=CHAMPION_PARAMS,
    ),
    UprArmConfig(
        name="pair_words15",
        chunker=WORD_WINDOWS_15,
        aggregations=("max",),
        predictor="zeroshot",
    ),
)

# The arms: (the run's label, the pack's groups or None for no learned arm,
# the label-blind counterpart).
ARMS: tuple[tuple[str, tuple[str, ...] | None, bool], ...] = (
    # the **baseline**: the champion's own scoring, the whole-text reference
    # arm and nothing else.
    ("upr-analyze-final-champion", None, False),
    # the **ladder's win**: the chunked retrieval arms and the zero-shot
    # geometry (the shortlist contrast). No learned arm: this run is about
    # the segmentation.
    ("upr-analyze-final-ladder", None, False),
    # the **learned pack** (the control for the two arms below: the same pack
    # and the same combiner, without the extra channel).
    ("upr-analyze-final-learned", FULL_PACK, False),
    # the **catalog's channels** (the CS-11 win).
    ("upr-analyze-final-catalog", CATALOG_PACK, False),
    # the **decision probability as a feature** (the CS-13 win) and its
    # label-blind counterpart (the yardstick).
    ("upr-analyze-final-decision-feature", DECISION_PACK, False),
    ("upr-analyze-final-decision-feature-blind", DECISION_PACK, True),
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
            arms=LADDER_ARMS if label.endswith("-ladder") else (),
            length_buckets=LENGTH_BUCKETS,
            reference_eval_run=CHAMPION_EVAL_RUN,
            # the baseline and the ladder cells carry **no** learned arm: the
            # champion's own scoring is the whole-text reference arm, and the
            # ladder's arms are the segmentation.
            learned=(
                UprLearnedConfig() if groups is None else _learned(groups, label_blind=label_blind)
            ),
        ),
    )
    for label, groups, label_blind in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="final_results")
