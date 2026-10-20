"""Phase 3.5 migration sweep (UPR): pre-3.5 runs re-emitted under the new grammar.

Re-runs `stats` (was `base`) and the two full-dataset `embed` passes (were
`embed-e5-base-v2` / `embed-arctic2`) under the `upr-` run-id grammar, with
the vectors warm in the disk cache — nothing is re-encoded. The frozen
configs and artifacts are identical to the pre-3.5 ones apart from the
run_id and wall-clock timings; the verified comparison is recorded in the
Phase 3.5 migration note in docs/experiments.md. The two embed rows land in
docs/experiments_table_upr.md under "## UPR — migration".

    uv run cluster-rag sweep --file experiments/upr_migration_sweep.py
"""

from __future__ import annotations

from cluster_rag.config import EmbedderConfig, PrefixesConfig
from cluster_rag.sweep import EmbedCell, SweepCell, define_sweep

# Model ids/prefixes/revisions copied verbatim from configs/experiments/embed_*.yaml
# (checked against the model cards in Phase 2); preprocessing stays the base
# config's raw variant.
cells = [
    SweepCell(label="upr-stats-base", command="stats"),
    EmbedCell(
        label="upr-embed-e5-base-v2",
        embed=EmbedderConfig(
            backend="hf",
            model_id="intfloat/e5-base-v2",
            revision="f52bf8ec8c7124536f0efb74aca902b2995e5bcd",  # pin the cache snapshot
            batch_size=128,
            device=None,
            prefixes=PrefixesConfig(query="query: ", document="passage: ", label="query: "),
        ),
    ),
    EmbedCell(
        label="upr-embed-arctic2",
        embed=EmbedderConfig(
            backend="ollama",
            model_id="snowflake-arctic-embed2:latest",
            batch_size=64,
            prefixes=PrefixesConfig(query="query: ", document="", label="query: "),
        ),
    ),
]

SWEEP = define_sweep(cells, dataset="upr", label="migration")
