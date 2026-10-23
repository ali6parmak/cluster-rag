"""Stage-0 analysis sweep: the OMCT pair-score artifact (CS-1).

`TEXT_PROCESSING_PLAN.md` section 3 (Stage 0.1) fixes the models of the
Stage-0 artifact: the two Stage-2 axis models (`embeddinggemma`, Ollama, and
`voyage-4-nano`, HF) plus `arctic2` (Ollama), whose full-text `strip_block`
scores produced the plan's indicative numbers — so the artifact re-derives
them on the same footing. All three are warm in the embedding cache: nothing
is re-encoded.

One cell per model: each cell is its own run (`omct-analyze-<model>`) with
its own artifact. The control set is model-independent and seeded, so the
three runs draw **exactly the same control cells** (the cell ids are in every
artifact, and a later stage re-uses them).

The model configs come from the shared pool (`experiments/embedder_pool.py`),
so the track's runs use the same model ids, prefixes and pinned revisions as
the main pipeline's sweeps.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.config import AnalysisConfig, AnalysisEmbedderConfig
from cluster_rag.sweep import AnalyzeOmctCell, define_sweep

# The Stage-0 models (TEXT_PROCESSING_PLAN.md section 3, Stage 0.1).
STAGE0_MODELS: tuple[str, ...] = ("embeddinggemma", "voyage-4-nano", "arctic2")

# The control size: a seeded, proportional sample of the unjudged
# same-document cells (the plan's default; the AI stratum is the binding
# constraint).
CONTROL_SIZE = 4000

cells = [
    AnalyzeOmctCell(
        label=f"analyze-{name}",
        analysis=AnalysisConfig(
            embedders=(AnalysisEmbedderConfig(name=name, embed=MODELS[name]),),
            control_size=CONTROL_SIZE,
        ),
    )
    for name in STAGE0_MODELS
]

SWEEP = define_sweep(cells, dataset="omct", label="analyze")
