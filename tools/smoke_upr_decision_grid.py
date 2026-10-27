"""A throwaway smoke script for the CS-9b code path (NOT a harness run).

It runs the `decide-upr` command with the sweep's own first cell, but with a
tiny scope (5 texts) and a tiny fit side (the first 5 train records), so the
wiring (the safe-negative rule, the label-blind counterpart, the shortlist
rerank) is exercised in seconds. It writes no table row and its results
folder is removed afterwards — it is a feasibility check, not a measurement.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from unittest.mock import patch

import cluster_rag.cli as cli
from cluster_rag.cli import _cmd_decide_upr, _load_sweep_runs

RUN_ID = "upr-decide-smoke"


def main() -> int:
    """Run the smoke test."""
    runs = _load_sweep_runs(Path("experiments_text_processing/upr_decision_grid_sweep.py"))
    config = runs[0].config
    config.run_id = RUN_ID
    config.upr_analysis.decision.scope = "sample"
    config.upr_analysis.decision.sample_size = 5
    # no table row: the smoke run is not a measurement
    config.paths.experiment_table_text_processing_upr = None

    original = cli._prepare_evaluation

    def patched(cfg, split):  # type: ignore[no-untyped-def]
        setup = original(cfg, split)
        return dataclasses.replace(setup, train_records=setup.train_records[:5])

    with patch.object(cli, "_prepare_evaluation", patched):
        code = _cmd_decide_upr(config)
    shutil.rmtree(Path("results") / RUN_ID, ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
