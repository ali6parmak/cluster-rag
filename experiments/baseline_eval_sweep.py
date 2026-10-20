"""Trivial-baseline evaluation sweep on the UPR dataset — the reference sweep file.

Cells: {most_frequent, random} x top_k k in {1, 2, 3} on **val**, then the
k=2 pair on **test** once (the reported reference numbers; test is touched
only once per AGENTS.md). Pure CPU — baselines need no embeddings. Cell
labels are the run ids (``upr-`` prefix per the run-id grammar) and resolve
to ``results/upr-eval-*/``; per-cell frozen configs, metrics.json and
comparison-table rows (``## UPR — baselines``) are exactly what standalone
`evaluate` runs produce:

    uv run cluster-rag sweep --file experiments/baseline_eval_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

PREDICTORS: tuple[tuple[str, str], ...] = (("most_frequent", "mf"), ("random", "random"))
VAL_KS: tuple[int, ...] = (1, 2, 3)

# Every label is explicit (no auto-names); `k` is the shorthand for the
# top_k decision, `split` names the fixed evaluation split.
cells = [
    EvalCell(label=f"upr-eval-{short}-topk{k}-val", predictor=name, k=k, split="val")
    for name, short in PREDICTORS
    for k in VAL_KS
]
# Test is touched once per predictor at the default operating point (k = 2).
cells += [
    EvalCell(label="upr-eval-mf-topk2-test", predictor="most_frequent", k=2, split="test"),
    EvalCell(label="upr-eval-random-topk2-test", predictor="random", k=2, split="test"),
]

# One sweep = one dataset ("upr": label prefix + allowed commands enforced);
# `label` names the sweep's table section in docs/experiments_table_upr.md.
SWEEP = define_sweep(cells, dataset="upr", label="baselines")
