"""OMCT probe sweep: the Phase 2 model grid (6 embedders) → `omct-probe2-*` runs.

Configs only (PLAN.md Phase 2.5 constraint): typed `ProbeCell`s built with
plain constructor calls; execution, artifacts, frozen per-run configs and
experiment-table rows belong to the sweep command:

    uv run cluster-rag sweep --file experiments/omct_probe_sweep.py

Cell labels are the bare run-id details — the `probe-aux` command adds the
`omct-` dataset prefix itself, so this grid produces `results/omct-probe2-<model>/`
folders and table sections "## OMCT — probe2" (+ "## OMCT — probe2 - groups").
Model ids/prefixes/revisions are copied verbatim from the Phase 2 configs
(`configs/experiments/probe_*.yaml`), where they were checked against the
model cards. `probe=None` keeps the schema defaults, so cells carry no probe
overrides; pass `probe=ProbeConfig(question_transforms=("strip_block", "keep"))`
for an explicit furniture comparison.
"""

from __future__ import annotations

from cluster_rag.config import EmbedderConfig, PrefixesConfig
from cluster_rag.sweep import ProbeCell, define_sweep

NO_PREFIXES = PrefixesConfig()  # every kind: no prefix


def _hf(
    model_id: str, revision: str, prefixes: PrefixesConfig, batch_size: int = 64
) -> EmbedderConfig:
    """Embedder config for a sentence-transformers backend (revision pinned)."""
    return EmbedderConfig(
        backend="hf",
        model_id=model_id,
        revision=revision,
        batch_size=batch_size,
        device=None,
        prefixes=prefixes,
    )


def _ollama(model_id: str, prefixes: PrefixesConfig) -> EmbedderConfig:
    """Embedder config for an Ollama backend (local models only)."""
    return EmbedderConfig(backend="ollama", model_id=model_id, batch_size=64, prefixes=prefixes)


MODELS: dict[str, EmbedderConfig] = {
    "minilm": _hf(
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        NO_PREFIXES,
    ),
    "e5-base-v2": _hf(
        "intfloat/e5-base-v2",
        "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        PrefixesConfig(query="query: ", document="passage: ", label="query: "),
    ),
    "bge-m3": _hf(
        "BAAI/bge-m3",
        "9a0624b896d81da7492a910ffa53731274b6cf3d",
        NO_PREFIXES,
        batch_size=32,
    ),
    "arctic2": _ollama(
        "snowflake-arctic-embed2:latest",
        PrefixesConfig(query="query: ", document="", label="query: "),
    ),
    "embeddinggemma": _ollama(
        "embeddinggemma:latest",
        PrefixesConfig(
            query="task: search result | query: {content}",
            document="title: none | text: {content}",
            label="task: classification | query: {content}",
        ),
    ),
    "mxbai": _ollama(
        "mxbai-embed-large:latest",
        PrefixesConfig(
            query="Represent this sentence for searching relevant passages: ",
            document="",
            label="",
        ),
    ),
}

cells = [ProbeCell(label=f"probe2-{name}", embed=embedder) for name, embedder in MODELS.items()]

SWEEP = define_sweep(cells, dataset="omct", label="probe2")
