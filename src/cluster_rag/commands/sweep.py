"""The ``sweep`` command: run a Python sweep file's resolved cells.

The sweep file only produces configs; this command executes each cell
through the same command functions the CLI uses, one run folder per cell.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from cluster_rag.commands.embed import _cmd_embed, _cmd_labelembed
from cluster_rag.commands.omct import (
    _cmd_analyze_omct,
    _cmd_decide_omct,
    _cmd_probe_omct,
)
from cluster_rag.commands.stats import _cmd_split, _cmd_stats
from cluster_rag.commands.upr import (
    _cmd_analyze_upr,
    _cmd_decide_upr,
    _cmd_evaluate,
    _cmd_hitmap_upr,
    _cmd_rerank_upr,
)
from cluster_rag.sweep import COMMANDS as SWEEP_COMMANDS
from cluster_rag.sweep import SHARED_EXPERIMENTS_DIR

logger = logging.getLogger(__name__)


def _load_sweep_runs(path: Path) -> list[Any]:
    """Import a sweep file and return its resolved SWEEP list (validated).

    The file is a Python module executed once at load; its ``SWEEP`` must be a
    non-empty list of :class:`~cluster_rag.sweep.SweepRun` — i.e. the product
    of ``define_sweep(cells, dataset=..., label=...)`` over ``configs/base.yaml``.

    While the file executes, two directories are importable: the sweep file's
    own directory (sibling modules) and the shared ``experiments/``
    directory, so ``embedder_pool.MODELS`` stays shared no matter where the
    sweep file lives (the text-processing track's sweeps live in
    ``experiments_text_processing/``). Both are removed again afterwards.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("cluster_rag_sweep_file", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load sweep file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # The sweep file's own directory first (its priority is the higher one),
    # then the shared experiments directory.
    candidates = [str(path.parent.resolve()), str(SHARED_EXPERIMENTS_DIR)]
    added: list[str] = []
    for entry in candidates:
        if entry in added or entry in sys.path or not Path(entry).is_dir():
            continue
        added.append(entry)
    for entry in added:
        sys.path.insert(0, entry)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import error aborts with its cause
        raise ValueError(f"loading sweep file {path} failed: {exc}") from exc
    finally:
        for entry in added:
            sys.path.remove(entry)
    sweep = getattr(module, "SWEEP", None)
    if not isinstance(sweep, list) or not sweep:
        raise ValueError(
            f"sweep file {path} must expose a non-empty SWEEP list "
            "(define_sweep(cells, dataset=..., label=...))"
        )
    from cluster_rag.sweep import SweepRun

    if not all(isinstance(run, SweepRun) for run in sweep):
        raise ValueError(
            f"sweep file {path}: SWEEP must contain SweepRun entries "
            "(from define_sweep), got "
            f"{[type(entry).__name__ for entry in sweep[:3]]}"
        )
    return sweep


def _cmd_sweep(path: Path) -> int:
    """Run the sweep command: execute each resolved cell as its own single run."""
    from cluster_rag.sweep import SweepRun

    runs = _load_sweep_runs(path)
    assert all(isinstance(run, SweepRun) for run in runs)  # typing only; _load_sweep_runs checks
    total = len(runs)
    started = time.perf_counter()
    print(f"Sweep {path}: {total} cell(s)")
    for index, run in enumerate(runs, start=1):
        print(f"  [{index}/{total}] {run.command} label={run.label} (run_id={run.config.run_id})")
        started_cell = time.perf_counter()
        if run.command == "embed":  # kinds come from the cell's resolved config
            code = _cmd_embed(run.config, kinds=run.config.embed_kinds)
        elif run.command == "labelembed":  # the zero-shot label-text artifact
            code = _cmd_labelembed(run.config)
        elif run.command == "evaluate":  # split comes from the cell's eval.split
            code = _cmd_evaluate(run.config)
        elif run.command == "stats":  # plain UPR dataset statistics
            code = _cmd_stats(run.config)
        elif run.command == "stats-omct":  # the omct pairwise-data stats (stats --omct)
            code = _cmd_stats(run.config, omct=True)
        elif run.command == "split":  # sweep cells never re-split silently (--force only via CLI)
            code = _cmd_split(run.config, force=False)
        elif run.command == "probe-omct":
            code = _cmd_probe_omct(run.config)
        elif run.command == "analyze-omct":  # the text-processing track's Stage 0
            code = _cmd_analyze_omct(run.config)
        elif run.command == "decide-omct":  # the text-processing track's Stage 3.5
            code = _cmd_decide_omct(run.config)
        elif run.command == "analyze-upr":  # the text-processing track's UPR side
            code = _cmd_analyze_upr(run.config)
        elif run.command == "decide-upr":  # the track's UPR decision probe (CS-9)
            code = _cmd_decide_upr(run.config)
        elif run.command == "rerank-upr":  # the track's UPR reranker probe (CS-15)
            code = _cmd_rerank_upr(run.config)
        elif run.command == "hitmap-upr":  # the track's hit map (CS-18)
            code = _cmd_hitmap_upr(run.config)
        else:  # a command the CLI does not implement: fail loudly, never guess
            raise ValueError(
                f"sweep cell {run.label!r} runs command {run.command!r}, which this CLI "
                "does not implement (known commands: "
                f"{', '.join(sorted(SWEEP_COMMANDS))})"
            )
        if code != 0:
            logger.error(
                "Sweep cell %d/%d (%s, label=%s) exited with code %d — aborting the sweep",
                index,
                total,
                run.command,
                run.label,
                code,
            )
            return code
        print(f"  [{index}/{total}] done in {time.perf_counter() - started_cell:.1f}s")
    print(f"Sweep finished: {total} run(s), {time.perf_counter() - started:.1f}s total")
    return 0
