"""Migration sweep (OMCT): the pairwise stats re-emitted under the omct- command names.

Re-runs `stats --omct` (the command was `stats --pairs` before Phase 3.6;
the run was `aux-base` before Phase 3.5) — the command adds the `omct-`
dataset prefix itself, so the cell label is the bare `stats-base` and
artifacts land in `results/omct-stats-base/` (the stats.json file was
pairs_stats.json before Phase 3.6). The corpus is unchanged; the verified
comparisons against the earlier artifacts are recorded in
docs/experiments.md (Phase 3.5 migration + Phase 3.6 notes). `stats` runs
append no comparison-table rows.

    uv run cluster-rag sweep --file experiments/omct_stats_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import SweepCell, define_sweep

cells = [SweepCell(label="stats-base", command="stats-omct")]

SWEEP = define_sweep(cells, dataset="omct", label="migration")
