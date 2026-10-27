"""A throwaway smoke script for the **listwise** shape (NOT a harness run).

The 20-option `choice` question has never been exercised against the server
(every track `choice` arm is two-option), so the shape is smoke-tested on
**three texts** before the real run: the script runs the `decide-upr` command
with the listwise sweep's own first cell, but with a three-text scope, and
prints the listwise arm's numbers. It writes no table row and its results
folder is removed afterwards -- a feasibility check, not a measurement.

Usage:

    uv run python tools/smoke_upr_listwise.py
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import cluster_rag.cli as cli
from cluster_rag.cli import _cmd_decide_upr, _load_sweep_runs

RUN_ID = "upr-decide-smoke-listwise"
SWEEP = "experiments_text_processing/upr_listwise_sweep.py"


def main() -> int:
    """Run the smoke test and print the listwise arm's numbers."""
    runs = _load_sweep_runs(Path(SWEEP))
    config = runs[0].config
    config.run_id = RUN_ID
    config.upr_analysis.decision.scope = "sample"
    config.upr_analysis.decision.sample_size = 3
    # no table row: the smoke run is not a measurement
    config.paths.experiment_table_text_processing_upr = None

    original = cli._prepare_evaluation

    def patched(cfg, split):  # type: ignore[no-untyped-def]
        setup = original(cfg, split)
        return dataclasses.replace(setup, train_records=setup.train_records[:3])

    with patch.object(cli, "_prepare_evaluation", patched):
        code = _cmd_decide_upr(config)
    artifact = Path("results") / RUN_ID / "decisions.json"
    if artifact.is_file():
        payload = json.loads(artifact.read_text("utf-8"))
        for name, block in payload["arms"].items():
            if "listwise" not in name:
                continue
            measurement = block["measurement"]
            print(
                f"  {name}: micro-F1 {measurement['decision']['micro_f1']:.4f}, "
                f"nDCG@5 {measurement['ranking'].get('ndcg@5', 0.0):.4f}, "
                f"calls {block.get('n_calls_planned')}, "
                f"calibration {measurement['calibration'].get('applicable')}"
            )
    shutil.rmtree(Path("results") / RUN_ID, ignore_errors=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
