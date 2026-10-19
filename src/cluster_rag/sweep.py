"""Experiment sweeps as Python (Phase 2.5, PLAN.md).

A sweep file (e.g. ``experiments/aux_probe_sweep.py``) declares typed cells
(:class:`SweepCell`: a command plus config overrides) built with plain Python
loops, and resolves them with :func:`define_sweep` into one validated
:class:`~cluster_rag.config.AppConfig` per cell. Sweeps only *produce*
configs: the ``sweep`` CLI command executes the resolved cells through the
existing command functions (``embed`` / ``probe-aux``), so run folders, frozen
configs, logs and experiment-table rows are exactly what single runs produce.

Guards: the label doubles as the run_id (unique across the sweep,
filesystem-safe), and ``run_id`` / ``extends`` may not be overridden — the
label defines the run id and the base file owns the ``extends`` chain.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cluster_rag.config import AppConfig, load_config, merge_configs

COMMANDS: tuple[str, ...] = ("embed", "probe-aux")


class SweepCell(BaseModel):
    """One sweep cell: a command plus the config overrides that define its run.

    ``label`` becomes the cell's run_id (aux- commands add the aux- prefix
    themselves at run time), so labels should read like run ids.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    command: str
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_is_run_id_safe(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(
                f"sweep cell label {value!r} must be filesystem-safe ([A-Za-z0-9._-]+)"
            )
        return value

    @field_validator("command")
    @classmethod
    def _command_is_known(cls, value: str) -> str:
        if value not in COMMANDS:
            raise ValueError(f"Unknown sweep command {value!r}; expected one of {COMMANDS}")
        return value


@dataclass(frozen=True)
class SweepRun:
    """One resolved sweep cell: label (= run_id), command and frozen config."""

    label: str
    command: str
    config: AppConfig


def define_sweep(cells: Sequence[SweepCell], *, base: str | Path) -> list[SweepRun]:
    """Resolve every cell over the base config into a validated :class:`SweepRun`.

    The base (a YAML path whose ``extends`` chain is resolved by
    :func:`~cluster_rag.config.load_config`) is merged with the cell's
    overrides (cell wins on every key, recursion included) and the run_id.
    Pydantic re-validation covers every override value, so an invalid cell
    fails here — before any run starts.
    """
    base_config = load_config(Path(base))
    runs: list[SweepRun] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.label in seen:
            raise ValueError(
                f"duplicate sweep cell label {cell.label!r}: labels double as run_ids "
                "and must be unique within a sweep"
            )
        seen.add(cell.label)
        forbidden = [key for key in cell.overrides if key in ("run_id", "extends")]
        if forbidden:
            raise ValueError(
                f"sweep cell {cell.label!r} overrides {forbidden}: run_id comes from "
                "the label and extends belongs to the base config file"
            )
        payload = merge_configs(
            base_config.model_dump(mode="python"), {"run_id": cell.label, **cell.overrides}
        )
        runs.append(
            SweepRun(
                label=cell.label, command=cell.command, config=AppConfig.model_validate(payload)
            )
        )
    return runs
