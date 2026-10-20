"""UPR threshold refinement (Phase 4): recalibrate near the per-config peak.

The rule comparison (`## UPR — thresholds`, champion embeddinggemma-raw +
score_threshold t=1.75: 0.5988 micro-F1) settled the rule family. The
top_k2-stage runner-up `upr-abl-jina-v5-strip` (0.5486, section
`## UPR — preprocess_ablation`) has its own fused scale (c=0.5, so the
zero-shot term's numeric range is about half) — same-model thresholds from
the champion's scale do not transfer. Grid placement from a throwaway
val-calibration snippet (val is the tuning split): jina-strip's st peak sits
around 2.1, its top_k_min_score peak around 1.9.

6 cells, section "## UPR — thresholds_refine"; run ids
`upr-eval-thr-<model>-<rule>t<t>`; the winner of THIS vs the champion rule
cell decides the single Phase-4-exit test config (test otherwise untouched).

    uv run cluster-rag sweep --file experiments/threshold_refine_sweep.py
"""

from __future__ import annotations

from cluster_rag.config import EvalDecisionConfig
from cluster_rag.sweep import EvalCell, define_sweep

MODEL: str = "jina-v5"
# jina-strip tuned weights (preprocess_ablation section)
FUSION_PARAMS: dict[str, float] = {"a": 0.5, "b": 1.0, "c": 0.5, "k": 10}
EMBEDDING_RUN: str = "upr-embed-jina-v5-strip"
LABEL_EMBEDDING_RUN: str = "upr-labelembed-jina-v5-strip-about"
ST_THRESHOLDS: tuple[float, ...] = (1.9, 2.1, 2.3)
TKMS_THRESHOLDS: tuple[float, ...] = (1.75, 1.9, 2.1)

cells: list[EvalCell] = [
    EvalCell(
        label=f"upr-eval-thr-{MODEL}-st{t:g}",
        predictor="fusion",
        params=FUSION_PARAMS,
        decision=EvalDecisionConfig(rule="score_threshold", threshold=t),
        embedding_run=EMBEDDING_RUN,
        label_embedding_run=LABEL_EMBEDDING_RUN,
    )
    for t in ST_THRESHOLDS
] + [
    EvalCell(
        label=f"upr-eval-thr-{MODEL}-tk2m{t:g}",
        predictor="fusion",
        params=FUSION_PARAMS,
        decision=EvalDecisionConfig(rule="top_k_min_score", k=2, min_score=t),
        embedding_run=EMBEDDING_RUN,
        label_embedding_run=LABEL_EMBEDDING_RUN,
    )
    for t in TKMS_THRESHOLDS
]

SWEEP = define_sweep(cells, dataset="upr", label="thresholds_refine")
