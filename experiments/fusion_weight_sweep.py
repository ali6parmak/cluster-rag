"""UPR fusion weight sweep (Phase 4): a·knn + b·proto on the four semifinalists.

Checkbox C named {embeddinggemma, jina-v5, voyage-4-nano, arctic2} as the
semifinalists (UPR val pool cut, tie/near-tie documented in
`docs/experiments.md`); this grid is the reviewer's swap point — change
``SEMIFINALISTS`` if the 4th slot is redirected, nothing else.

Grid (b fixed at 1: with the rank-based top_k decision only the a:b ratio
matters — the pure mix corners are already covered by the pool-cut cells,
knn at k=10/k=20 and prototype as a=0):

- kNN component k ∈ {10, 20} (each semifinalist's best k from the pool cut
  sits in this pair)
- a ∈ {0.25, 0.5, 1, 2, 4, 8} -> 48 cells, section "## UPR — fusion_weights"

Run ids: `upr-eval-fusion-<model>-k<k>-a<a>`; decision top_k k=2 (as the whole
pool cut, so every number stays apples-to-apples); val only, test untouched.

    uv run cluster-rag sweep --file experiments/fusion_weight_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

SEMIFINALISTS: tuple[str, ...] = ("embeddinggemma", "jina-v5", "voyage-4-nano", "arctic2")
KNN_KS: tuple[int, ...] = (10, 20)
KNN_WEIGHTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # b = 1.0

cells: list[EvalCell] = []
for detail in SEMIFINALISTS:
    embedding_run = f"upr-embed-{detail}"
    for k in KNN_KS:
        for a in KNN_WEIGHTS:
            cells.append(
                EvalCell(
                    label=f"upr-eval-fusion-{detail}-k{k}-a{a:g}",
                    predictor="fusion",
                    params={"a": a, "b": 1.0, "k": k},
                    k=2,
                    embedding_run=embedding_run,
                )
            )

SWEEP = define_sweep(cells, dataset="upr", label="fusion_weights")
