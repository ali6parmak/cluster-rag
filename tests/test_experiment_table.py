"""Tests for the auto-appended experiment tables (Phase 2.5).

Contract: one markdown table per (phase, command) heading; rows upsert by
their key columns; a missing file/heading is created; counts and "n/a" cells
render next to numbers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster_rag.experiment_table import (
    EMBED_COLUMNS,
    EMBED_KEY_COLUMNS,
    PROBE_AUX_COLUMNS,
    PROBE_AUX_KEY_COLUMNS,
    _separator,
    append_table_rows,
)


def _embed_row(run_id: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": run_id,
        "date": "2026-10-19",
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


def test_append_creates_file_heading_and_rows(tmp_path: Path) -> None:
    """A missing file gets the intro plus the heading, header and data rows."""
    path = tmp_path / "docs" / "experiments_table.md"
    append_table_rows(
        path,
        heading="Phase 2 — embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[_embed_row("embed-a"), _embed_row("embed-b", cache_hits=3000, cache_misses=0)],
    )
    text = path.read_text("utf-8")
    assert text.startswith("# Experiment tables")
    assert "## Phase 2 — embed" in text
    lines = text.splitlines()
    assert lines[lines.index("## Phase 2 — embed") + 2] == "| " + " | ".join(EMBED_COLUMNS) + " |"
    assert "| embed-a |" in text
    assert "| embed-b |" in text and "3000" in text
    # trailing newline, one heading only
    assert text.endswith("\n")
    assert text.count("## Phase 2 — embed") == 1


def test_append_upserts_by_key_columns(tmp_path: Path) -> None:
    """Re-running a run_id replaces its row; a new run_id appends after it."""
    path = tmp_path / "tables.md"
    append_table_rows(
        path,
        heading="Phase 2 — embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[_embed_row("embed-a", seconds=4.937)],
    )
    append_table_rows(
        path,
        heading="Phase 2 — embed",
        columns=EMBED_COLUMNS,
        key_columns=EMBED_KEY_COLUMNS,
        rows=[_embed_row("embed-a", seconds=3.453, cache_hits=3000, cache_misses=0)],
    )
    lines = path.read_text("utf-8").splitlines()
    data_rows = [line for line in lines if line.startswith("| embed-")]
    assert len(data_rows) == 1  # replaced in place, not duplicated
    assert "3.453" in data_rows[0] and "cache_hits" not in data_rows[0]


def test_multiple_headings_are_independent(tmp_path: Path) -> None:
    """A second table gets its own section; existing tables stay untouched."""
    path = tmp_path / "tables.md"
    columns = ("run_id", "transform", "n_pairs", "pair_auc")
    append_table_rows(
        path,
        heading="Phase 2 — probe-aux",
        columns=columns,
        key_columns=("run_id",),
        rows=[{"run_id": "aux-x", "transform": "strip_block", "n_pairs": 1520, "pair_auc": 0.5793}],
    )
    append_table_rows(
        path,
        heading="Phase 2 — probe-aux groups",
        columns=columns,
        key_columns=("run_id", "transform"),
        rows=[{"run_id": "aux-x", "transform": "strip_block", "n_pairs": 845, "pair_auc": 0.4821}],
    )
    text = path.read_text("utf-8")
    assert text.count("## Phase 2 —") == 2
    assert "0.5793" in text and "845" in text
    # the first table's row is intact and not inside the second section
    lines = text.splitlines()
    assert lines.index("## Phase 2 — probe-aux") < lines.index("## Phase 2 — probe-aux groups")
    assert "| aux-x | strip_block | 1520 | 0.5793 |" in text


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


def test_probe_row_shape_matches_probe_columns() -> None:
    """Sanity: every probe column is a string, so the CLI mapping stays honest."""
    assert "run_id" in PROBE_AUX_COLUMNS and "pair_auc" in PROBE_AUX_COLUMNS
    assert "transform" in PROBE_AUX_COLUMNS and PROBE_AUX_KEY_COLUMNS == (
        "run_id",
        "transform",
    )
    assert "cache_misses" in EMBED_COLUMNS


def test_separator_width() -> None:
    assert _separator(3) == "|---|---|---|"
