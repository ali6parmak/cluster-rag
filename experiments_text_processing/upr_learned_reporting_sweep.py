"""The UPR reporting blocks: the calibration and the decision-rule comparison (CS-8f).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8f) asks for the two reporting
blocks on the UPR side, on the **best arm** of the CS-8e grid (the
gradient-boosted combiner on the full pack — the best decision-view number and
the best shortlist recall the track has measured on UPR):

- the **calibration**: the reliability diagram of the arm's probability
  (the expected calibration error, the Brier score, the per-bin records),
  pooled and per label bucket and per text-length bucket;
- the **decision-rule comparison**: the three rules (`top_k`, `score_threshold`
  and `top_k_min_score`) on the same arm. The threshold rules' cut is chosen
  on the **fit side** (the train split) by the Youden separation, so the
  evaluated split never picks the operating point.

The second run repeats the block with the **logistic** combiner on the same
pack, so the two learners' calibration can be read side by side (the
track's discipline: a comparison, never two independent numbers).
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import (
    ChunkerSpec,
    CombinerSpec,
    UprAnalysisConfig,
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

# The two arms: the best arm of the CS-8e grid and the logistic on the same
# pack (so the two learners' calibration can be compared).
ARMS: tuple[tuple[str, str], ...] = (
    ("histgb", "histgb"),
    ("logistic", "logistic"),
)

cells = [
    AnalyzeUprCell(
        label=f"upr-analyze-learned-report-{name}",
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
            learned=UprLearnedConfig(
                enabled=True,
                groups=FULL_PACK,
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
                seed=42,
            ),
        ),
    )
    for name, combiner in ARMS
]

SWEEP = define_sweep(cells, dataset="upr", label="upr_learned_reporting")
