"""UPR preprocess ablation (Phase 4): strip/lower vs raw on the semifinalists.

"Embed all, ablate few": the preprocess variants are ablated on the four
checkbox-C semifinalists ONLY, at each model's **tuned** fusion weights from
the fusion grids (a=0.5, b=1; c and knn-k per model below), decision top_k
k=2 — the same decision every fusion number was decided under. Baselines =
the existing raw fusionc cells (`upr-eval-fusionc-<model>-k<k>-c<c>`, section
`## UPR — fusion_c`; NOT re-run).

Embedding the ablated variants re-encodes everything affected (the
preprocess version is part of the cache key): per model and variant one
document,query UPR embed + one `about` label-text artifact under the same
preprocessing (the zero-shot query side must live on the same scale), then a
val evaluate cell. 24 runs total, section "## UPR — preprocess_ablation";
run ids `upr-embed-<model>-<variant>` / `upr-labelembed-<model>-<variant>-about`
/ `upr-eval-abl-<model>-<variant>`. The two Ollama cells need `ollama serve`;
HF cells ride the local GPU.

    uv run cluster-rag sweep --file experiments/preprocess_ablation_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import LabelTextsConfig, PreprocessConfig
from cluster_rag.sweep import EmbedCell, EvalCell, LabelEmbedCell, define_sweep

SEMIFINALISTS: tuple[str, ...] = ("embeddinggemma", "jina-v5", "voyage-4-nano", "arctic2")
# per model tuned fusion weights (Phase 4 log): a=0.5, b=1, zero-shot c, knn k
TUNED_C: dict[str, float] = {
    "embeddinggemma": 1.0,
    "jina-v5": 0.5,
    "voyage-4-nano": 0.5,
    "arctic2": 0.25,
}
TUNED_K: dict[str, int] = {
    "embeddinggemma": 10,
    "jina-v5": 10,
    "voyage-4-nano": 20,
    "arctic2": 10,
}
ABOUT_TEMPLATE: str = "This recommendation is about {label}."
VARIANTS: tuple[str, ...] = ("strip_leading_formulaic", "lowercase")
DETAIL: dict[str, str] = {"strip_leading_formulaic": "strip", "lowercase": "lower"}

cells: list[EmbedCell | LabelEmbedCell | EvalCell] = []
for detail in SEMIFINALISTS:
    embedder = MODELS[detail]
    for variant in VARIANTS:
        short = DETAIL[variant]
        prep = PreprocessConfig(variant=variant)
        cells.append(
            EmbedCell(
                label=f"upr-embed-{detail}-{short}",
                embed=embedder,
                kinds=("document", "query"),
                preprocess=prep,
            )
        )
        cells.append(
            LabelEmbedCell(
                label=f"upr-labelembed-{detail}-{short}-about",
                embed=embedder,
                label_texts=LabelTextsConfig(variant="template", template=ABOUT_TEMPLATE),
                preprocess=prep,
            )
        )
        cells.append(
            EvalCell(
                label=f"upr-eval-abl-{detail}-{short}",
                predictor="fusion",
                params={"a": 0.5, "b": 1.0, "c": TUNED_C[detail], "k": TUNED_K[detail]},
                k=2,
                embedding_run=f"upr-embed-{detail}-{short}",
                label_embedding_run=f"upr-labelembed-{detail}-{short}-about",
            )
        )

SWEEP = define_sweep(cells, dataset="upr", label="preprocess_ablation")
