"""Experiment sweeps as Python (Phase 2.5/3.5): typed cells -> resolved configs.

A sweep file under ``experiments/`` declares typed cells built with plain
Python loops — :class:`EvalCell`, :class:`EmbedCell`, :class:`LabelEmbedCell`,
:class:`ProbeCell`, :class:`AnalyzeOmctCell` and :class:`DecideOmctCell`
(validated fields, no dict/YAML wiring), plus the generic :class:`SweepCell`
(label + command + config overrides) for the commands without a dedicated cell
(``stats``/``stats-omct``/``split``). :func:`define_sweep` resolves every
cell over ``configs/base.yaml`` into one validated
:class:`~cluster_rag.config.AppConfig` per run. Sweeps only *produce*
configs: the ``sweep`` CLI command executes the resolved cells through the
existing command functions (``embed`` / ``labelembed`` / ``evaluate`` /
``probe-omct`` / ``analyze-omct`` / ``decide-omct`` / ``stats`` / ``split``), so
run folders, frozen configs, logs and experiment-table rows are exactly what
single runs produce.

A sweep is authored for exactly **one dataset** (:func:`define_sweep`'s
``dataset``, ``upr`` or ``omct``): it validates each cell's command against
that dataset and takes the sweep ``label`` that names the comparison-table
section — injected into every resolved config as ``table.sweep_label``.
Cell labels are mandatory, explicit, filesystem-safe run ids: ``upr`` cells
must start with ``upr-``; ``omct`` cells carry no dataset prefix (the
command adds ``omct-`` itself). Guards: labels are unique within a sweep,
and ``run_id`` / ``extends`` / ``table.sweep_label`` may not be overridden —
the label defines the run id, the base file owns the ``extends`` chain, and
the sweep label defines the table section.

Sweep files may live in their own directory: the ``sweep`` command's loader
puts the file's directory **and** the shared ``experiments/`` directory on
``sys.path`` while the file executes (see :data:`SHARED_EXPERIMENTS_DIR`), so
``embedder_pool.MODELS`` stays shared across the main pipeline's sweeps and
the text-processing track's sweeps in ``experiments_text_processing/``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cluster_rag.config import (
    AnalysisConfig,
    AppConfig,
    EmbedderConfig,
    EvalDecisionConfig,
    EvalMetricsConfig,
    LabelTextsConfig,
    PreprocessConfig,
    ProbeConfig,
    UprAnalysisConfig,
    load_config,
    merge_configs,
    validate_embedding_run_id,
    validate_label_embedding_run_id,
)
from cluster_rag.embed.base import KINDS

# ``stats-omct`` is the sweep command for `stats --omct` (separate name, so
# the dataset decides nothing at execution time); ``stats`` is the plain UPR
# dataset statistics run.
UPR_COMMANDS: tuple[str, ...] = (
    "stats",
    "split",
    "embed",
    "labelembed",
    "evaluate",
    "analyze-upr",
)
OMCT_COMMANDS: tuple[str, ...] = (
    "stats-omct",
    "probe-omct",
    "analyze-omct",
    "decide-omct",
)
DATASET_COMMANDS: dict[str, tuple[str, ...]] = {"upr": UPR_COMMANDS, "omct": OMCT_COMMANDS}
COMMANDS: tuple[str, ...] = (
    "stats",
    "stats-omct",
    "split",
    "embed",
    "labelembed",
    "evaluate",
    "probe-omct",
    "analyze-omct",
    "decide-omct",
    "analyze-upr",
)

# The one base config every sweep resolves over (PLAN.md section 5); relative
# to the working directory the CLI is run from (the repository root).
BASE_CONFIG_PATH = Path("configs/base.yaml")

# The shared experiment modules (``experiments/embedder_pool.py``: the
# card-verified model configs of the whole pool) stay importable from every
# sweep file, wherever it lives: the ``sweep`` command's loader puts this
# directory on ``sys.path`` next to the sweep file's own directory, so the
# text-processing track's sweeps (in ``experiments_text_processing/``) can
# still ``from embedder_pool import MODELS``.
SHARED_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

_SWEEP_LABEL_ERROR = "sweep labels must be filesystem-safe ([A-Za-z0-9._-]+)"


def _validate_label(value: str) -> str:
    """Shared validation for cell labels (run ids) and the sweep label."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"labels must be filesystem-safe ([A-Za-z0-9._-]+), got {value!r}")
    return value


class _LabelledCell(BaseModel):
    """Base for typed sweep cells: a mandatory, explicit label (= run id)."""

    model_config = ConfigDict(extra="forbid")

    label: str

    @field_validator("label")
    @classmethod
    def _label_is_run_id_safe(cls, value: str) -> str:
        return _validate_label(value)


class SweepCell(_LabelledCell):
    """One generic sweep cell: a command plus the config overrides that define its run.

    Used for commands without a dedicated typed cell (``stats``, ``stats-omct``,
    ``split``); ``evaluate`` / ``embed`` / ``probe-omct`` have :class:`EvalCell` /
    :class:`EmbedCell` / :class:`ProbeCell`. ``label`` becomes the cell's
    run_id (the omct- dataset commands add the ``omct-`` prefix themselves at
    run time), so labels should read like run ids.
    """

    command: str
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("command")
    @classmethod
    def _command_is_known(cls, value: str) -> str:
        if value not in COMMANDS:
            raise ValueError(f"Unknown sweep command {value!r}; expected one of {COMMANDS}")
        return value


class EvalCell(_LabelledCell):
    """One ``evaluate`` cell: a predictor plus its decision rule and split.

    ``k`` is the shorthand for a ``top_k`` decision (the Phase 3 grid); a full
    ``decision`` (:class:`~cluster_rag.config.EvalDecisionConfig`) is the
    alternative for the other rules — exactly one of the two. ``metrics=None``
    keeps the schema default grids; ``split`` is restricted to the fixed
    evaluation splits. ``embedding_run`` (Phase 4 opening D) names the prior
    ``upr-embed-*`` run whose vectors embedding-dependent predictors consume
    (knn, prototype, max-sim); embedding-free predictors ignore it (warn) and
    such predictors without it fail with the wiring message.
    """

    predictor: str = Field(min_length=1)  # name from the predictors registry
    params: dict[str, Any] = Field(default_factory=dict)
    decision: EvalDecisionConfig | None = None
    k: int | None = Field(default=None, ge=1)
    split: Literal["val", "test"] = "val"
    metrics: EvalMetricsConfig | None = None
    embedding_run: str | None = None
    # Zero-shot label vectors (Phase 4): a prior `upr-labelembed-*` run; needed
    # by predictor `zeroshot` and by `fusion` with c > 0 only.
    label_embedding_run: str | None = None

    @field_validator("embedding_run")
    @classmethod
    def _embedding_run_is_valid(cls, value: str | None) -> str | None:
        """Syntactic check at authoring time (the loader checks existence later)."""
        if value is None:
            return None
        return validate_embedding_run_id(value)

    @field_validator("label_embedding_run")
    @classmethod
    def _label_embedding_run_is_valid(cls, value: str | None) -> str | None:
        """Syntactic check at authoring time (the loader checks existence later)."""
        if value is None:
            return None
        return validate_label_embedding_run_id(value)

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "evaluate"

    @model_validator(mode="after")
    def _decision_xor_k(self) -> EvalCell:
        """``decision`` and the ``k`` shorthand are exactly one thing."""
        if self.decision is not None and self.k is not None:
            raise ValueError(
                f"EvalCell {self.label!r}: give decision= or the top_k shorthand k=, not both"
            )
        if self.decision is None and self.k is None:
            raise ValueError(
                f"EvalCell {self.label!r}: needs a decision rule — pass the top_k "
                "shorthand k= or a full decision="
            )
        return self

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides: the complete ``eval`` section."""
        decision = self.decision or EvalDecisionConfig(rule="top_k", k=self.k)
        payload: dict[str, Any] = {
            "eval": {
                "predictor": {"name": self.predictor, "params": self.params},
                "decision": decision.model_dump(mode="python"),
                "split": self.split,
                "embedding_run_id": self.embedding_run,
                "label_embedding_run_id": self.label_embedding_run,
            }
        }
        if self.metrics is not None:
            payload["eval"]["metrics"] = self.metrics.model_dump(mode="python")
        return payload


class EmbedCell(_LabelledCell):
    """One ``embed`` cell: backend/model config + the encode kinds for the run.

    ``kinds`` (validated against the ``KINDS`` encode kinds) names the matrices
    the run writes, one per kind (Phase 4 kind sweeps use ``("document",
    "query")``); the UPR dataset text is embedded under each kind's configured
    prefix. ``preprocess=None`` keeps the base config's variant (usually
    ``raw``).
    """

    embed: EmbedderConfig
    kinds: tuple[str, ...] = ("document",)
    preprocess: PreprocessConfig | None = None

    @field_validator("kinds")
    @classmethod
    def _kinds_are_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject unknown/empty kind lists before any run resolves."""
        if not value:
            raise ValueError("EmbedCell kinds must name at least one encode kind")
        unknown = [kind for kind in value if kind not in KINDS]
        if unknown:
            raise ValueError(
                f"Unknown encode kind(s) {unknown} in EmbedCell; expected values in {KINDS}"
            )
        return value

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "embed"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (``embed``/``embed_kinds`` + optional preprocessing)."""
        payload: dict[str, Any] = {
            "embed": self.embed.model_dump(mode="python"),
            "embed_kinds": list(self.kinds),
        }
        if self.preprocess is not None:
            payload["preprocess"] = self.preprocess.model_dump(mode="python")
        return payload


class LabelEmbedCell(_LabelledCell):
    """One ``labelembed`` cell: label texts + embedder (the zero-shot artifact).

    ``label_texts`` picks what stands in for each dataset label (variant
    ``name`` or a ``{label}`` template, :class:`~cluster_rag.config.LabelTextsConfig`);
    the artifacts it produces are consumed through ``eval.label_embedding_run_id``
    by the ``zeroshot`` predictor and by ``fusion`` with ``c > 0``.
    ``preprocess=None`` keeps the base config's variant (usually ``raw``).
    """

    embed: EmbedderConfig
    label_texts: LabelTextsConfig
    preprocess: PreprocessConfig | None = None

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "labelembed"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (``embed`` + ``label_texts`` + preprocess)."""
        payload: dict[str, Any] = {
            "embed": self.embed.model_dump(mode="python"),
            "label_texts": self.label_texts.model_dump(mode="python"),
        }
        if self.preprocess is not None:
            payload["preprocess"] = self.preprocess.model_dump(mode="python")
        return payload


class ProbeCell(_LabelledCell):
    """One ``probe-omct`` cell: embedder + probe options on the pairwise dataset.

    ``probe=None`` keeps the schema defaults (``strip_block`` transforms; add
    an explicit :class:`~cluster_rag.config.ProbeConfig` for a ``keep``
    comparison). The command adds the ``omct-`` dataset prefix at run time.
    """

    embed: EmbedderConfig
    probe: ProbeConfig | None = None
    preprocess: PreprocessConfig | None = None

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "probe-omct"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (``embed`` plus optional probe/preprocess)."""
        payload: dict[str, Any] = {"embed": self.embed.model_dump(mode="python")}
        if self.probe is not None:
            payload["probe"] = self.probe.model_dump(mode="python")
        if self.preprocess is not None:
            payload["preprocess"] = self.preprocess.model_dump(mode="python")
        return payload


class AnalyzeOmctCell(_LabelledCell):
    """One ``analyze-omct`` cell: the models to score with + the control design.

    The text-processing track's Stage-0 artifact
    (TEXT_PROCESSING_PLAN.md section 3, Stage 0.1): the per-pair scores of
    the judged pairs, the control set (a seeded, proportional sample of the
    *unjudged* same-document cells), the per-document strata and the
    negative-enrichment measurement. ``analysis`` is the whole
    :class:`~cluster_rag.config.AnalysisConfig` — the models (each with its
    card-verified embedder config), the control size, the sampling rule, the
    question transform and the chunker. The command adds the ``omct-``
    dataset prefix at run time.
    """

    analysis: AnalysisConfig

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "analyze-omct"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (the complete ``analysis`` section)."""
        return {"analysis": self.analysis.model_dump(mode="python")}


class DecideOmctCell(_LabelledCell):
    """One ``decide-omct`` cell: the decision models and their prompt grid.

    The text-processing track's Stage-3.5 probe
    (TEXT_PROCESSING_PLAN.md section 3, Stage 3.5.1): the decision models
    score the reviewed OMCT pairs, and the artifact carries the measurement
    blocks beside the synthetic competence control. ``analysis`` is the whole
    :class:`~cluster_rag.config.AnalysisConfig` — the decision section (the
    models with their pinned digests, the prompt, the scope and the
    competence control) plus the control design the probe shares with the
    Stage-0 runs. The command adds the ``omct-`` dataset prefix at run time.
    """

    analysis: AnalysisConfig

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "decide-omct"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (the complete ``analysis`` section)."""
        return {"analysis": self.analysis.model_dump(mode="python")}


class AnalyzeUprCell(_LabelledCell):
    """One ``analyze-upr`` cell: the reference predictor + the track's arms.

    The text-processing track's UPR artifact
    (TEXT_PROCESSING_PLAN.md section 7, CS-8a): the whole-text **reference
    arm** (the main pipeline's own scoring of the split, which must reproduce
    the matching ``evaluate`` run field by field) beside the track's chunked
    arms, with the UPR-side diagnostics (the non-gold labels as the
    background, the rank percentiles, the hit@k, the level and rank shift).

    The reference predictor is the **same** wiring the ``evaluate`` command
    uses (``eval.predictor`` and the two embedding runs), so the reference arm
    is the pipeline's own numbers and not a re-implementation of them.
    ``embed`` is the embedder the chunked arms encode their chunks with (it
    must live in the same space as the embedding run's vectors); it is
    optional, because a run with no chunked arm needs no encoder.
    """

    predictor: str = Field(min_length=1)  # name from the predictors registry
    params: dict[str, Any] = Field(default_factory=dict)
    decision: EvalDecisionConfig | None = None
    k: int | None = Field(default=None, ge=1)
    split: Literal["val", "test"] = "val"
    metrics: EvalMetricsConfig | None = None
    embedding_run: str
    label_embedding_run: str | None = None
    analysis: UprAnalysisConfig
    embed: EmbedderConfig | None = None

    @field_validator("embedding_run")
    @classmethod
    def _embedding_run_is_valid(cls, value: str) -> str:
        """Syntactic check at authoring time (the loader checks existence later)."""
        return validate_embedding_run_id(value)

    @field_validator("label_embedding_run")
    @classmethod
    def _label_embedding_run_is_valid(cls, value: str | None) -> str | None:
        """Syntactic check at authoring time (the loader checks existence later)."""
        if value is None:
            return None
        return validate_label_embedding_run_id(value)

    @model_validator(mode="after")
    def _decision_xor_k(self) -> AnalyzeUprCell:
        """``decision`` and the ``k`` shorthand are exactly one thing."""
        if self.decision is not None and self.k is not None:
            raise ValueError(
                f"AnalyzeUprCell {self.label!r}: give decision= or the top_k shorthand k=, not both"
            )
        if self.decision is None and self.k is None:
            raise ValueError(
                f"AnalyzeUprCell {self.label!r}: needs a decision rule — pass the "
                "top_k shorthand k= or a full decision="
            )
        return self

    @property
    def command(self) -> str:
        """The command this cell resolves to (constant for typed cells)."""
        return "analyze-upr"

    @property
    def overrides(self) -> dict[str, Any]:
        """The cell's config overrides (the ``eval`` section + ``upr_analysis``)."""
        decision = self.decision or EvalDecisionConfig(rule="top_k", k=self.k)
        payload: dict[str, Any] = {
            "eval": {
                "predictor": {"name": self.predictor, "params": self.params},
                "decision": decision.model_dump(mode="python"),
                "split": self.split,
                "embedding_run_id": self.embedding_run,
                "label_embedding_run_id": self.label_embedding_run,
            },
            "upr_analysis": self.analysis.model_dump(mode="python"),
        }
        if self.metrics is not None:
            payload["eval"]["metrics"] = self.metrics.model_dump(mode="python")
        if self.embed is not None:
            payload["embed"] = self.embed.model_dump(mode="python")
        return payload


Cell = (
    SweepCell
    | EvalCell
    | EmbedCell
    | LabelEmbedCell
    | ProbeCell
    | AnalyzeOmctCell
    | DecideOmctCell
    | AnalyzeUprCell
)


@dataclass(frozen=True)
class SweepRun:
    """One resolved sweep cell: label (= run_id), command and frozen config."""

    label: str
    command: str
    config: AppConfig


def _validate_sweep_label(label: str) -> str:
    """The sweep label names the table section: filesystem-safe, no prefix rules."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(_SWEEP_LABEL_ERROR)
    return label


def _label_runs_allowed(label: str, dataset: str) -> None:
    """Run-id prefix rules per sweep dataset (AGENTS.md run-id grammar)."""
    if dataset == "upr":
        if not label.startswith("upr-"):
            raise ValueError(f"upr sweep cell label {label!r} must start with 'upr-'")
        return
    if label.startswith(("omct-", "upr-")):
        raise ValueError(
            f"omct sweep cell label {label!r} must not carry a dataset prefix: the "
            f"command adds 'omct-' itself (run id omct-{label})"
        )


def _cell_parts(cell: Cell) -> tuple[str, str, dict[str, Any]]:
    """Normalize any typed cell into (label, command, overrides)."""
    return cell.label, cell.command, dict(cell.overrides)


def define_sweep(
    cells: Sequence[Cell], *, dataset: Literal["upr", "omct"], label: str
) -> list[SweepRun]:
    """Resolve every cell over ``configs/base.yaml`` into a validated :class:`SweepRun`.

    The base (``configs/base.yaml``, whose ``extends`` chain is resolved by
    :func:`~cluster_rag.config.load_config`) is merged with the cell's
    overrides (cell wins on every key, recursion included) plus the run_id and
    the injected ``table.sweep_label``. Pydantic re-validation covers every
    override value, so an invalid cell fails here — before any run starts.

    Constraints: ``dataset`` selects the allowed commands (upr:
    stats/split/embed/evaluate; omct: stats-omct/probe-omct/analyze-omct) and
    the cell-label prefix rules (upr cells start with ``upr-``; omct cells are
    unprefixed); ``label`` (filesystem-safe, free-form) names the sweep's
    table section.
    """
    if dataset not in DATASET_COMMANDS:
        raise ValueError(
            f"Unknown sweep dataset {dataset!r}; expected one of {sorted(DATASET_COMMANDS)}"
        )
    _validate_sweep_label(label)
    allowed = DATASET_COMMANDS[dataset]
    base_config = load_config(BASE_CONFIG_PATH)

    runs: list[SweepRun] = []
    seen: set[str] = set()
    for cell in cells:
        cell_label, command, overrides = _cell_parts(cell)
        if cell_label in seen:
            raise ValueError(
                f"duplicate sweep cell label {cell_label!r}: labels double as run_ids "
                "and must be unique within a sweep"
            )
        seen.add(cell_label)
        if command not in allowed:
            raise ValueError(
                f"sweep cell {cell_label!r} runs {command!r}, which is not a "
                f"{dataset!r} sweep command (allowed: {allowed})"
            )
        _label_runs_allowed(cell_label, dataset)
        forbidden = [key for key in overrides if key in ("run_id", "extends")]
        if "table" in overrides and "sweep_label" in overrides.get("table", {}):
            forbidden.append("table.sweep_label")
        if forbidden:
            raise ValueError(
                f"sweep cell {cell_label!r} overrides {forbidden}: run_id comes from "
                "the label, table.sweep_label is injected by define_sweep, and "
                "extends belongs to the base config file"
            )
        payload = merge_configs(
            base_config.model_dump(mode="python"),
            {"run_id": cell_label, "table": {"sweep_label": label}, **overrides},
        )
        runs.append(
            SweepRun(label=cell_label, command=command, config=AppConfig.model_validate(payload))
        )
    return runs
