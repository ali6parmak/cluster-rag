"""UPR fusion c-weight sweep (Phase 4): adding the zero-shot term to the mix.

Grid over the winning zero-shot template only: the zero-shot sweep
(``## UPR — zeroshot``, 2026-10-20) picked variant ``about`` (template
"This recommendation is about {label}." — 3/4 models, best 4-model mean) — so the
label side is fixed at ``upr-labelembed-<model>-about`` (all four built). The
a-grid (``## UPR — fusion_weights``) settled the knn/proto ratio per
reviewer's log note: proto-heavy ratios win (every model's best ``a`` ≤ 0.5,
landscape flat) — fixing ``a = 0.5, b = 1`` isolates the ``c`` effect.

Grid: a ∈ {0.5} (fixed), b ∈ {1.0} (fixed), c ∈ {0.1, 0.25, 0.5, 1.0},
k ∈ {10, 20} -> 32 cells, section "## UPR — fusion_c". Run ids:
`upr-eval-fusionc-<model>-k<k>-c<c>`; decision top_k k=2 (the whole Phase-4
comparability), val only, test untouched. CPU-only: embedding + label
artifacts are prebuilt, nothing re-encodes.

    uv run cluster-rag sweep --file experiments/fusion_c_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

SEMIFINALISTS: tuple[str, ...] = ("embeddinggemma", "jina-v5", "voyage-4-nano", "arctic2")
KNN_KS: tuple[int, ...] = (10, 20)
FUSION_A: float = 0.5  # best a per the fusion_weights grid (a ≤ 0.5 wins)
FUSION_B: float = 1.0  # scale anchor
ZEROSHOT_WEIGHTS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
LABEL_VARIANT: str = "about"  # zero-shot sweep winner

cells: list[EvalCell] = []
for detail in SEMIFINALISTS:
    embedding_run = f"upr-embed-{detail}"
    label_embedding_run = f"upr-labelembed-{detail}-{LABEL_VARIANT}"
    for k in KNN_KS:
        for c in ZEROSHOT_WEIGHTS:
            cells.append(
                EvalCell(
                    label=f"upr-eval-fusionc-{detail}-k{k}-c{c:g}",
                    predictor="fusion",
                    params={"a": FUSION_A, "b": FUSION_B, "c": c, "k": k},
                    k=2,
                    embedding_run=embedding_run,
                    label_embedding_run=label_embedding_run,
                )
            )

SWEEP = define_sweep(cells, dataset="upr", label="fusion_c")
