"""Phase 3 trivial-baseline evaluation sweep (one UPR run per cell, aux-free).

Cells: {most_frequent, random} x top_k k in {1, 2, 3} on **val**, then the
k=2 pair on **test** once (the reported reference numbers; test is touched
only once per AGENTS.md). Pure CPU — baselines need no embeddings. Run ids
`ev-<predictor>-<decision>-<split>` (UPR runs, no aux- prefix); per-cell
frozen configs, metrics.json and auto table rows are exactly what single
`evaluate` runs produce:

    uv run cluster-rag sweep --file experiments/baseline_eval_sweep.py
"""

from cluster_rag.sweep import SweepCell, define_sweep

BASE = "configs/base.yaml"


def _eval_cell(label: str, predictor: str, k: int, split: str) -> SweepCell:
    """One evaluate cell: predictor + top-k decision + fixed split."""
    return SweepCell(
        label=label,
        command="evaluate",
        overrides={
            "eval": {
                "predictor": {"name": predictor},
                "decision": {"rule": "top_k", "k": k},
                "split": split,
            }
        },
    )


PREDICTORS = (("most_frequent", "mf"), ("random", "random"))
VAL_KS = (1, 2, 3)

cells = [
    _eval_cell(f"ev-{short}-topk{k}-val", name, k, "val")
    for name, short in PREDICTORS
    for k in VAL_KS
]
# Test is touched once per predictor at the default operating point (k = 2).
cells += [
    _eval_cell("ev-mf-topk2-test", "most_frequent", 2, "test"),
    _eval_cell("ev-random-topk2-test", "random", 2, "test"),
]

SWEEP = define_sweep(cells, base=BASE)
