"""UPR Phase-4-exit test run (one cell, test split, never touched before).

The val-evidence champion across ALL Phase-4 dimensions (model + preprocess +
fusion weights + decision rule): embeddinggemma raw + fusion(a=0.5, b=1.0,
c=1.0, knn k=10) + decision `score_threshold` t=1.75 — the top-k decision was
beaten on val by the thresholded rules (rule comparison `## UPR — thresholds`:
0.5988 vs 0.5479 micro-F1), and the preprocess ablation ("## UPR —
preprocess_ablation") did not displace the raw champion (its only
top_k2-stage rival, `upr-abl-jina-v5-strip`, stays below under its own
recalibrated thresholds — `## UPR — thresholds_refine`). The operating point
(t=1.75) is the val-calibrated value, carried to test UNEDITED.

Single test-split cell, run id `upr-eval-best-embeddinggemma-st1.75-test`;
compare against the test floors `upr-eval-{mf,random}-topk2-test` (0.1413 /
0.0304 micro-F1). This is the only test-split number of Phase 4; val-only
grids everywhere else.

    uv run cluster-rag sweep --file experiments/final_test_eval.py
"""

from __future__ import annotations

from cluster_rag.config import EvalDecisionConfig
from cluster_rag.sweep import EvalCell, define_sweep

cells: list[EvalCell] = [
    EvalCell(
        label="upr-eval-best-embeddinggemma-st1.75-test",
        predictor="fusion",
        params={"a": 0.5, "b": 1.0, "c": 1.0, "k": 10},
        decision=EvalDecisionConfig(rule="score_threshold", threshold=1.75),
        split="test",
        embedding_run="upr-embed-embeddinggemma",
        label_embedding_run="upr-labelembed-embeddinggemma-about",
    )
]

SWEEP = define_sweep(cells, dataset="upr", label="phase4_final")
