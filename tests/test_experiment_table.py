"""Tests for the auto-appended per-dataset experiment tables (Phase 3.5).

Contract: two table files (one per dataset); rows land under their sweep's
``## <UPR|OMCT> — <sweep-label>`` heading (else the standalone
``## <UPR|OMCT> — <command>`` heading); rows upsert by their key columns; a
missing file/heading is created; counts and "n/a" cells render next to
numbers. The text-processing track owns its own pair of files, chosen by the
command's (dataset, track) pair — never by a heuristic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster_rag.config import PathsConfig
from cluster_rag.experiment_table import (
    EMBED_COLUMNS,
    EMBED_KEY_COLUMNS,
    EVALUATE_COLUMNS,
    EVALUATE_KEY_COLUMNS,
    MAIN_TRACK,
    PROBE_OMCT_COLUMNS,
    PROBE_OMCT_KEY_COLUMNS,
    TEXT_PROCESSING_TRACK,
    TableTarget,
    _separator,
    append_table_rows,
    intro_for,
    section_heading,
    table_path,
    table_target,
)
from cluster_rag.sweep import COMMANDS


def _embed_row(run_id: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": run_id,
        "date": "2026-10-20",
        "backend": "hf",
        "model": "intfloat/e5-base-v2",
        "model_version": "f52bf8ec8c71",
        "dim": 768,
        "ctx": 512,
        "kinds": "document",
        "preprocess": "raw",
        "n_texts": 3000,
        "seconds": 4.937,
        "cache_hits": 0,
        "cache_misses": 3000,
    }
    return {**values, **overrides}


def test_section_headings() -> None:
    """Sweep label first; command-based fallback; groups suffixed; UPR/OMCT tags."""
    assert section_heading("upr", command="embed", sweep_label="baselines") == "UPR — baselines"
    assert section_heading("upr", command="evaluate") == "UPR — evaluate"  # standalone run
    assert section_heading("omct", command="probe-omct") == "OMCT — probe-omct"
    assert (
        section_heading("omct", command="probe-omct", sweep_label="probe2", groups=True)
        == "OMCT — probe2 - groups"
    )
    assert (
        section_heading("omct", command="probe-omct", groups=True) == "OMCT — probe-omct - groups"
    )


def test_intro_per_dataset() -> None:
    assert intro_for("upr").startswith("# UPR experiment tables")
    assert intro_for("omct").startswith("# OMCT experiment tables")
    assert "docs/experiments.md" in intro_for("omct")


def test_append_creates_file_heading_and_rows(tmp_path: Path) -> None:
    """A missing file gets the intro plus the heading, header and data rows."""
    path = tmp_path / "docs" / "experiments_table_upr.md"
    append_table_rows(
        path,
        heading=section_heading("upr", command="embed"),
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        intro=intro_for("upr"),
        rows=[
            _embed_row("upr-embed-a"),
            _embed_row("upr-embed-b", cache_hits=3000, cache_misses=0),
        ],
    )
    text = path.read_text("utf-8")
    assert text.startswith("# UPR experiment tables")
    assert "## UPR — embed" in text
    lines = text.splitlines()
    assert lines[lines.index("## UPR — embed") + 2] == "| " + " | ".join(EMBED_COLUMNS) + " |"
    assert "| upr-embed-a |" in text
    assert "| upr-embed-b |" in text and "3000" in text
    # trailing newline, one heading only
    assert text.endswith("\n")
    assert text.count("## UPR — embed") == 1


def test_append_upserts_by_key_columns(tmp_path: Path) -> None:
    """Re-running a run_id replaces its row; a new run_id appends after it."""
    path = tmp_path / "tables.md"
    append_table_rows(
        path,
        heading="UPR — embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[_embed_row("upr-embed-a", seconds=4.937)],
    )
    append_table_rows(
        path,
        heading="UPR — embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[
            _embed_row("upr-embed-a", seconds=3.453, cache_hits=3000, cache_misses=0),
            _embed_row("upr-embed-c"),
        ],
    )
    lines = path.read_text("utf-8").splitlines()
    data_rows = [line for line in lines if line.startswith("| upr-")]
    assert len(data_rows) == 2  # one replaced row + the new one appended after it
    assert "3.453" in data_rows[0] and "cache_misses" not in data_rows[0]
    assert data_rows[0].startswith("| upr-embed-a |") and data_rows[1].startswith("| upr-embed-c")
    assert lines.index("| " + " | ".join(EMBED_COLUMNS) + " |") < lines.index(data_rows[1])


def test_multiple_headings_are_independent(tmp_path: Path) -> None:
    """A second table gets its own section; existing tables stay untouched."""
    path = tmp_path / "tables.md"
    columns = ("run_id", "transform", "n_pairs", "pair_auc")
    append_table_rows(
        path,
        heading="OMCT — probe-omct",
        columns=columns,
        key_columns=("run_id",),
        rows=[
            {"run_id": "omct-x", "transform": "strip_block", "n_pairs": 1520, "pair_auc": 0.5793}
        ],
    )
    append_table_rows(
        path,
        heading="OMCT — probe-omct - groups",
        columns=columns,
        key_columns=("run_id", "transform"),
        rows=[{"run_id": "omct-x", "transform": "strip_block", "n_pairs": 845, "pair_auc": 0.4821}],
    )
    text = path.read_text("utf-8")
    assert text.count("## OMCT —") == 2
    assert "0.5793" in text and "845" in text
    # the first table's row is intact and not inside the second section
    lines = text.splitlines()
    assert lines.index("## OMCT — probe-omct") < lines.index("## OMCT — probe-omct - groups")
    assert "| omct-x | strip_block | 1520 | 0.5793 |" in text


def test_missing_and_none_cells_render(tmp_path: Path) -> None:
    """None/absent values render as n/a; a pipe in a value cannot break rows."""
    path = tmp_path / "tables.md"
    columns = ("run_id", "value")
    append_table_rows(
        path,
        heading="h",
        columns=columns,
        key_columns=("run_id",),
        rows=[
            {"run_id": "r-1", "value": None},
            {"run_id": "r-2"},  # value missing entirely
            {"run_id": "r-3", "value": "a|b"},
        ],
    )
    lines = path.read_text("utf-8").splitlines()
    rendered = [row for row in lines[lines.index("## h") :] if row.startswith("| r-")]
    assert "| r-1 | n/a |" in rendered
    assert "| r-2 | n/a |" in rendered
    assert "| r-3 | a/b |" in rendered  # pipe escaped as "/"


def test_key_column_outside_columns_raises(tmp_path: Path) -> None:
    """Key columns must be part of the table columns."""
    with pytest.raises(ValueError, match="key columns"):
        append_table_rows(
            tmp_path / "t.md",
            heading="h",
            columns=("run_id",),
            key_columns=("run_id", "bogus"),
            rows=[],
        )


def test_column_groupings_match_the_phase35_contract() -> None:
    """Sanity: the regrouped evaluate table (f1 decision first, ranking second)."""
    assert EVALUATE_COLUMNS[:5] == ("run_id", "date", "split", "predictor", "embedding")
    assert EVALUATE_COLUMNS[5:10] == (
        "f1 decision",
        "micro-F1",
        "macro-F1",
        "sample-F1",
        "coverage",
    )
    assert EVALUATE_COLUMNS[10:] == (
        "P@5",
        "R@10",
        "nDCG@5",
        "shortlist R@50",
        "n_texts",
        "seconds",
    )
    assert EVALUATE_KEY_COLUMNS == ("run_id",)
    assert "cache_misses" in EMBED_COLUMNS
    assert "run_id" in PROBE_OMCT_COLUMNS and "pair_auc" in PROBE_OMCT_COLUMNS
    assert PROBE_OMCT_KEY_COLUMNS == ("run_id", "transform")


def test_separator_width() -> None:
    assert _separator(3) == "|---|---|---|"


def _paths(**overrides: object) -> PathsConfig:
    """A paths config with every table file set (overrides win)."""
    values: dict[str, object] = {
        "data_raw": Path("data/raw/x.jsonl"),
        "data_processed": Path("data/processed"),
        "data_splits": Path("data/splits"),
        "cache": Path("cache"),
        "results": Path("results"),
        "experiment_table_upr": Path("docs/upr.md"),
        "experiment_table_omct": Path("docs/omct.md"),
        "experiment_table_text_processing_omct": Path("docs/tp_omct.md"),
        "experiment_table_text_processing_upr": Path("docs/tp_upr.md"),
    }
    values.update(overrides)
    return PathsConfig(**values)  # type: ignore[arg-type]


def test_command_targets_map_dataset_and_track() -> None:
    """Every command names its (dataset, track) pair; a typo fails loudly."""
    assert table_target("embed") == TableTarget("upr", MAIN_TRACK)
    assert table_target("evaluate") == TableTarget("upr", MAIN_TRACK)
    assert table_target("probe-omct") == TableTarget("omct", MAIN_TRACK)
    assert table_target("analyze-omct") == TableTarget("omct", TEXT_PROCESSING_TRACK)
    assert table_target("chunk-omct") == TableTarget("omct", TEXT_PROCESSING_TRACK)
    assert table_target("chunk-upr") == TableTarget("upr", TEXT_PROCESSING_TRACK)
    with pytest.raises(ValueError, match="Unknown command"):
        table_target("not-a-command")


def test_every_sweep_command_has_a_table_target() -> None:
    """No sweep command may lack a table (the mapping is command-derived)."""
    for command in COMMANDS:
        assert table_target(command).dataset in ("upr", "omct")


def test_table_path_is_command_derived() -> None:
    """The track's commands write to the track's files, never to the main ones."""
    paths = _paths()
    assert table_path(paths, "embed") == Path("docs/upr.md")
    assert table_path(paths, "probe-omct") == Path("docs/omct.md")
    assert table_path(paths, "analyze-omct") == Path("docs/tp_omct.md")
    assert table_path(paths, "chunk-omct") == Path("docs/tp_omct.md")
    assert table_path(paths, "chunk-upr") == Path("docs/tp_upr.md")


def test_table_path_none_skips_the_append() -> None:
    """An unset table file means "no table" (the append is skipped)."""
    paths = _paths(experiment_table_text_processing_omct=None)
    assert table_path(paths, "analyze-omct") is None
    assert table_path(paths, "embed") == Path("docs/upr.md")


def test_track_intro_names_the_track_log() -> None:
    """The track's table files point at the track's own narrative log."""
    intro = intro_for("omct", TEXT_PROCESSING_TRACK)
    assert intro.startswith("# OMCT text-processing experiment tables")
    assert "docs/experiments_text_processing.md" in intro
    assert "docs/experiments.md" not in intro
