"""Phase 3.5 migration sweep (OMCT): the pairwise stats re-emitted under the new grammar.

Re-runs `stats --pairs` (was run `aux-base`) under the omct- naming — the
command adds the `omct-` dataset prefix itself, so the cell label is the
bare `stats-base` and artifacts land in `results/omct-stats-base/`. The
corpus is unchanged; the verified comparison against the pre-3.5 artifact is
recorded in the Phase 3.5 migration note in docs/experiments.md. `stats`
runs append no comparison-table rows.

    uv run cluster-rag sweep --file experiments/omct_stats_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import SweepCell, define_sweep

cells = [SweepCell(label="stats-base", command="stats-pairs")]

SWEEP = define_sweep(cells, dataset="omct", label="migration")
