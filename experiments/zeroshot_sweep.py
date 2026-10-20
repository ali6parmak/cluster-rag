"""UPR zero-shot sweep (Phase 4): label-text templates over the four semifinalists.

The zero-shot mechanism test after the reviewer's OK: `labelembed` builds one
label-text artifact per (model, variant) — `variant: name` (the bare label, no
catalog file exists for descriptions/descriptions-templates, PLAN.md section 7)
plus two `{label}` templates — and each `zeroshot` predictor cell scores the
fixed **val** split against that artifact (decision `top_k k=2`, matching every
Phase-4 number so far). The label-text templates ride the models'
card-verified `prefixes.label`; nothing else about the pool changes.

Cells (=run ids), 24 in total, section "## UPR — zeroshot":

- `upr-labelembed-<model>-<variant>`  — the label-text artifact (68 labels)
- `upr-eval-zeroshot-<model>-<variant>` — cosine(text, label text) on val

models: the four checkbox-C semifinalists only ("embed all, ablate few" — the
pool cut already excluded the rest). The fusion `c`-weight grid is a separate
sweep, authored once the winning variant is known (next change set).

    uv run cluster-rag sweep --file experiments/zeroshot_sweep.py
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import LabelTextsConfig
from cluster_rag.sweep import EvalCell, LabelEmbedCell, define_sweep

SEMIFINALISTS: tuple[str, ...] = ("embeddinggemma", "jina-v5", "voyage-4-nano", "arctic2")

VARIANTS: dict[str, LabelTextsConfig] = {
    "name": LabelTextsConfig(variant="name"),
    "about": LabelTextsConfig(variant="template", template="This recommendation is about {label}."),
    "topic": LabelTextsConfig(variant="template", template="Topic: {label}."),
}

cells: list[LabelEmbedCell | EvalCell] = []
for detail in SEMIFINALISTS:
    embedder = MODELS[detail]
    for variant, label_texts in VARIANTS.items():
        cells.append(
            LabelEmbedCell(
                label=f"upr-labelembed-{detail}-{variant}",
                embed=embedder,
                label_texts=label_texts,
            )
        )
        cells.append(
            EvalCell(
                label=f"upr-eval-zeroshot-{detail}-{variant}",
                predictor="zeroshot",
                k=2,
                embedding_run=f"upr-embed-{detail}",
                label_embedding_run=f"upr-labelembed-{detail}-{variant}",
            )
        )

SWEEP = define_sweep(cells, dataset="upr", label="zeroshot")
