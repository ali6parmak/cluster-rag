"""UPR decision-rule sweep (Phase 4): top_k vs score_threshold vs min-score top-k.

The fusion weights are done (champion `upr-eval-fusionc-embeddinggemma-k10-c1`,
0.5479 micro-F1, UPR val, in the `## UPR — fusion_c` section — NOT re-run
here); this grid varies only the **decision rule** on the champion's fused
scores (embeddinggemma, fusion a=0.5, b=1.0, c=1.0, knn k=10) with:
- rank-rule context: top_k k ∈ {1, 3} (k=2 is the incumbent cell, already run),
- `score_threshold` (predict every label with score >= threshold),
- `top_k_min_score` (k=2 but drop any top-2 label below min_score).

Grid placement — a throwaway val-calibration snippet (val is the tuning
split) found the champion's rank-2 fused scores separate gold from non-gold
labels widely (not-gold q75 ≈ 0.70 vs gold q25 ≈ 1.64), so the threshold grid
brackets that gap: t ∈ {0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.6, 1.75,
1.9, 2.0} (the fused scale itself depends on the pinned weights, so these
values ride on the champion cell). All 24 cells below are the artifact of
record; nothing here is a test-set number (val only, top_k-family decisions
otherwise in their schema defaults).

Section "## UPR — thresholds"; run ids `upr-eval-thr-embeddinggemma-<rule><t>`.
Thresholding is calibrated on val; the chosen operating point is carried
(unedited) to the single Phase-4-exit test run.

    uv run cluster-rag sweep --file experiments/threshold_sweep.py
"""

from __future__ import annotations

from cluster_rag.config import EvalDecisionConfig
from cluster_rag.sweep import EvalCell, define_sweep

MODEL: str = "embeddinggemma"  # the tuned champion model
FUSION_PARAMS: dict[str, float] = {"a": 0.5, "b": 1.0, "c": 1.0, "k": 10}
EMBEDDING_RUN: str = f"upr-embed-{MODEL}"
LABEL_EMBEDDING_RUN: str = f"upr-labelembed-{MODEL}-about"
THRESHOLDS: tuple[float, ...] = (0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.6, 1.75, 1.9, 2.0)

cells: list[EvalCell] = []
for k in (1, 3):  # rank-rule context; k=2 is the incumbent cell (fusion_c section)
    cells.append(
        EvalCell(
            label=f"upr-eval-thr-{MODEL}-tk{k}",
            predictor="fusion",
            params=FUSION_PARAMS,
            k=k,
            embedding_run=EMBEDDING_RUN,
            label_embedding_run=LABEL_EMBEDDING_RUN,
        )
    )
for t in THRESHOLDS:
    cells.append(
        EvalCell(
            label=f"upr-eval-thr-{MODEL}-st{t:g}",
            predictor="fusion",
            params=FUSION_PARAMS,
            decision=EvalDecisionConfig(rule="score_threshold", threshold=t),
            embedding_run=EMBEDDING_RUN,
            label_embedding_run=LABEL_EMBEDDING_RUN,
        )
    )
    cells.append(
        EvalCell(
            label=f"upr-eval-thr-{MODEL}-tk2m{t:g}",
            predictor="fusion",
            params=FUSION_PARAMS,
            decision=EvalDecisionConfig(rule="top_k_min_score", k=2, min_score=t),
            embedding_run=EMBEDDING_RUN,
            label_embedding_run=LABEL_EMBEDDING_RUN,
        )
    )

SWEEP = define_sweep(cells, dataset="upr", label="thresholds")
