"""UPR pool-cut sweep (Phase 4 opening checkbox C): knn / prototype / max-sim on val.

The val-driven cut of the 14-embedder pool to 3-4 semifinalists ("embed all,
ablate few" — no 14^3 factorial). For every pool model from the shared table
(`experiments/embedder_pool.py`; its `upr-embed-<model>` runs already hold the
document+query matrices), three predictor families score the fixed **val**
split. The decision rule is `top_k k=2` for every cell, so micro-F1 is
apples-to-apples with the `upr-eval-mf/topk2-*` trivial baselines (mean gold
labels/record 1.91); the ranking view (P@5 / R@10 / nDCG@5 / shortlist
recall@m) is decision-independent. Each metrics.json carries the
per-frequency-bucket numbers (head/torso/tail/unseen) plus the embedding
run's full lineage.

Cells (=run ids):

- knn, k in {1, 3, 5, 10, 20}    -> `upr-eval-knn-<model>-k<k>`
- prototype (renormalized mean)  -> `upr-eval-proto-<model>`
- max-sim (max cosine per label) -> `upr-eval-maxsim-<model>`

98 cells, CPU-only, a few seconds each (nothing is re-encoded at eval time —
the cells consume their prior embedding artifacts through
`eval.embedding_run_id`). All rows land under "## UPR — pool_cut".

    uv run cluster-rag sweep --file experiments/pool_cut_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS  # the shared card-verified pool (no drift)

from cluster_rag.sweep import EvalCell, define_sweep

EMBED_RUN_DETAILS: tuple[str, ...] = tuple(MODELS)
KNN_KS: tuple[int, ...] = (1, 3, 5, 10, 20)

cells: list[EvalCell] = []
for detail in EMBED_RUN_DETAILS:
    embedding_run = f"upr-embed-{detail}"
    for k in KNN_KS:
        cells.append(
            EvalCell(
                label=f"upr-eval-knn-{detail}-k{k}",
                predictor="knn",
                params={"k": k},
                k=2,
                embedding_run=embedding_run,
            )
        )
    cells.append(
        EvalCell(
            label=f"upr-eval-proto-{detail}",
            predictor="prototype",
            k=2,
            embedding_run=embedding_run,
        )
    )
    cells.append(
        EvalCell(
            label=f"upr-eval-maxsim-{detail}",
            predictor="maxsim",
            k=2,
            embedding_run=embedding_run,
        )
    )

SWEEP = define_sweep(cells, dataset="upr", label="pool_cut")
