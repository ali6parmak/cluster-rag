"""The CLI's shared helpers: the run-id plumbing, the table append, the guard.

Every command needs the same few things: the run id (and the dataset
prefix rules), the resolved-config writer, the comparison-table append and
the whole-text guard. They live here so the command modules hold only
their own logic.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from cluster_rag.config import AppConfig
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.experiment_table import (
    append_table_rows,
    intro_for,
    section_heading,
    table_path,
    table_target,
)
from cluster_rag.label_catalog import LabelEntry, load_label_catalog

logger = logging.getLogger(__name__)


def _require_run_id(config: AppConfig) -> str:
    """Return the run_id from the config or raise, since artifacts need it."""
    if not config.run_id:
        raise ValueError(
            "run_id must be set in the config: commands write to results/<run_id>/ "
            "and the experiment log references the same run_id."
        )
    return config.run_id


def _append_command_rows(
    config: AppConfig,
    command: str,
    *,
    columns: Sequence[str],
    key_columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
    groups: bool = False,
) -> None:
    """Append one command's rows to its comparison table (command-derived).

    The command decides the ``(dataset, track)`` pair and therefore the file:
    the main pipeline's commands write to the per-dataset tables, the
    text-processing track's commands to the track's own pair of tables
    (AGENTS.md "Runs and where things live"). An unset path skips the append
    (throwaway configs in tests).
    """
    path = table_path(config.paths, command)
    if path is None:
        return
    target = table_target(command)
    append_table_rows(
        path,
        heading=section_heading(
            target.dataset,
            command=command,
            sweep_label=config.table.sweep_label,
            groups=groups,
        ),
        columns=columns,
        key_columns=key_columns,
        intro=intro_for(target.dataset, target.track),
        rows=rows,
    )


UPR_PREFIX = "upr-"  # UPR dataset runs (AGENTS.md "Runs and where things live")
OMCT_PREFIX = "omct-"  # auxiliary OMCT pairwise-data runs


def _warn_upr_prefix(run_id: str) -> None:
    """Warn (never fail) when a UPR command's run_id lacks the `upr-` prefix.

    UPR commands accept any run_id so existing configs keep working; the
    warning keeps the `<dataset>-<command>-<detail>[-<split>]` grammar visible.
    """
    if not run_id.startswith(UPR_PREFIX):
        logger.warning(
            "run_id %r does not start with the UPR dataset prefix %r "
            "(expected grammar: <dataset>-<command>-<detail>[-<split>] per AGENTS.md); "
            "the run proceeds without it.",
            run_id,
            UPR_PREFIX,
        )


def _omct_run_id(run_id: str) -> str:
    """Return the run_id with the `omct-` dataset prefix added by the command.

    OMCT commands prefix the run id themselves (the config carries the bare
    detail), so a config that already carries `omct-`/`upr-` would double up —
    warned about, never fatal.
    """
    if run_id.startswith((OMCT_PREFIX, UPR_PREFIX)):
        logger.warning(
            "run_id %r already carries a dataset prefix; the command adds %r "
            "itself, so artifacts land under results/%s%s/ (off-grammar but not fatal).",
            run_id,
            OMCT_PREFIX,
            OMCT_PREFIX,
            run_id,
        )
    return f"{OMCT_PREFIX}{run_id}"


def _fmt(value: float | None, decimals: int) -> str | None:
    """Fixed-precision rendering for table cells (None passes through)."""
    if value is None:
        return None
    return f"{value:.{decimals}f}"


def _write_resolved_config(results_dir: Path, config: AppConfig) -> None:
    """Freeze the resolved config into results/<run_id>/config.yaml."""
    payload = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    (results_dir / "config.yaml").write_text(payload, encoding="utf-8")


def _cached_embedder(embedder: Any, config: AppConfig, version: str) -> Any:
    """The embedder of one side, cached under that side's version slot."""
    if not config.cache.enabled:
        return embedder
    return CachedEmbedder(embedder, config.paths.cache, preprocessing_version=version)


def _check_reference_arm(
    config: AppConfig,
    reference_run_id: str | None,
    reference_block: dict[str, Any],
) -> dict[str, Any] | None:
    """The guard: the whole-text arm must reproduce the ``evaluate`` run.

    The UPR analogue of ``tools/verify_analysis_baseline.py``: the track's
    whole-text reference arm is the main pipeline's own scoring, so it must
    match the matching ``evaluate`` run's report **field by field**. A
    mismatch means the run's numbers cannot be compared with the pipeline's.

    Args:
        config: The resolved app config (the results folder).
        reference_run_id: The ``evaluate`` run to compare with (``None``
            disables the check).
        reference_block: The reference arm's block (its ``metrics``).

    Returns:
        The guard's outcome (recorded in the artifact), or ``None`` when the
        check was not requested.

    Raises:
        ValueError: If the check fails — a mismatch is never a warning.
    """
    if not reference_run_id:
        return None
    path = config.paths.results / reference_run_id / "metrics.json"
    if not path.is_file():
        raise ValueError(
            f"the reference evaluate run {reference_run_id!r} has no metrics artifact "
            f"({path}); the guard cannot check the whole-text arm"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("report")
    actual = reference_block["metrics"]["overall"]
    differences: list[str] = []
    if expected is None:
        differences.append("the reference run has no report block")
    else:
        if expected.get("overall") != actual:
            differences.append("the overall report differs")
        if expected.get("per_bucket") != reference_block["metrics"]["per_bucket"]:
            differences.append("the per-bucket report differs")
    if payload.get("n_records") != actual.get("n_records"):
        differences.append("the record count differs")
    if payload.get("n_labels") != actual.get("n_labels"):
        differences.append("the label count differs")
    if differences:
        raise ValueError(
            f"the whole-text reference arm does not reproduce the evaluate run "
            f"{reference_run_id!r}: {'; '.join(differences)}"
        )
    return {
        "reference_run": reference_run_id,
        "checked": ["overall", "per_bucket", "n_records", "n_labels"],
        "matches": True,
    }


def _upr_label_catalog(config: AppConfig) -> dict[str, LabelEntry]:
    """The optional label catalog (``paths.label_catalog``).

    An absent catalog is not an error: the pack's two catalog-gated feature
    groups are then unavailable and the artifact records the gap.
    """
    path = config.paths.label_catalog
    if path is None:
        return {}
    catalog = load_label_catalog(path)
    logger.info("the label catalog %s carries %d entries", path, len(catalog))
    return catalog
