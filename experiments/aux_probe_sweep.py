"""Phase 2.5 aux-probe sweep: the Phase 2 model grid under the final code.

Configs only (PLAN.md Phase 2.5 constraint): plain loops build typed cells;
execution, artifacts, frozen per-run configs and experiment-table rows belong
to the sweep command:

    uv run cluster-rag sweep --file experiments/aux_probe_sweep.py

Each label becomes the cell's run_id (the probe-aux command adds the ``aux-``
prefix itself), so this grid produces ``results/aux-probe2-<model>/`` folders.
Model ids/prefixes/revisions are copied verbatim from the Phase 2 configs
(``configs/experiments/probe_*.yaml``), where they were checked against the
model cards. ``strip_block`` is the schema default, so cells carry no probe
overrides; add ``{"probe": {"question_transforms": ["strip_block", "keep"]}}``
for an explicit furniture comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cluster_rag.sweep import SweepCell, define_sweep

BASE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def _hf(
    model_id: str, revision: str, prefixes: dict[str, str], batch_size: int = 64
) -> dict[str, Any]:
    """Embed overrides for a sentence-transformers backend."""
    return {
        "backend": "hf",
        "model_id": model_id,
        "revision": revision,
        "batch_size": batch_size,
        "device": None,
        "prefixes": prefixes,
    }


def _ollama(model_id: str, prefixes: dict[str, str]) -> dict[str, Any]:
    """Embed overrides for an Ollama backend (local models only)."""
    return {"backend": "ollama", "model_id": model_id, "batch_size": 64, "prefixes": prefixes}


NO_PREFIXES = {"query": "", "document": "", "label": ""}

HF_MODELS = [
    (
        "minilm",
        _hf(
            "sentence-transformers/all-MiniLM-L6-v2",
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            NO_PREFIXES,
        ),
    ),
    (
        "e5-base-v2",
        _hf(
            "intfloat/e5-base-v2",
            "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            {"query": "query: ", "document": "passage: ", "label": "query: "},
        ),
    ),
    (
        "bge-m3",
        _hf("BAAI/bge-m3", "9a0624b896d81da7492a910ffa53731274b6cf3d", NO_PREFIXES, batch_size=32),
    ),
]

OLLAMA_MODELS = [
    (
        "arctic2",
        _ollama(
            "snowflake-arctic-embed2:latest",
            {"query": "query: ", "document": "", "label": "query: "},
        ),
    ),
    (
        "embeddinggemma",
        _ollama(
            "embeddinggemma:latest",
            {
                "query": "task: search result | query: {content}",
                "document": "title: none | text: {content}",
                "label": "task: classification | query: {content}",
            },
        ),
    ),
    (
        "mxbai",
        _ollama(
            "mxbai-embed-large:latest",
            {
                "query": "Represent this sentence for searching relevant passages: ",
                "document": "",
                "label": "",
            },
        ),
    ),
]

CELLS = [
    SweepCell(label=f"probe2-{name}", command="probe-aux", overrides={"embed": embed_overrides})
    for name, embed_overrides in [*HF_MODELS, *OLLAMA_MODELS]
]

SWEEP = define_sweep(CELLS, base=BASE)
