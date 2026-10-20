"""OMCT probe sweep: the Phase 4 14-model pool → `omct-probe2-*` runs.

Configs only (PLAN.md Phase 2.5 constraint): typed `ProbeCell`s built with
plain constructor calls; execution, artifacts, frozen per-run configs and
experiment-table rows belong to the sweep command:

    uv run cluster-rag sweep --file experiments/omct_probe_sweep.py

Cell labels are the bare run-id details — the `probe-omct` command adds the
`omct-` dataset prefix itself, so this grid produces `results/omct-probe2-<model>/`
folders and table sections "## OMCT — probe2" (+ "## OMCT — probe2 - groups").
Model ids/prefixes/revisions live in `experiments/embedder_pool.py`, the table
shared with the UPR embedding sweep (verified against the model cards per pool
entry; HF revisions pin the snapshot hashes, Ollama digests are recorded by the
backend at run time). `probe=None` keeps the schema defaults, so cells carry
no probe overrides; pass `probe=ProbeConfig(question_transforms=("strip_block",
"keep"))` for an explicit furniture comparison.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.sweep import ProbeCell, define_sweep

cells = [ProbeCell(label=f"probe2-{name}", embed=embedder) for name, embedder in MODELS.items()]

SWEEP = define_sweep(cells, dataset="omct", label="probe2")
