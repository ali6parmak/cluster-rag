"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_jsonl() -> Path:
    """Path to the tiny fixture dataset (~20 rows, see AGENTS.md testing rules)."""
    return FIXTURE_DIR / "sample.jsonl"


@pytest.fixture()
def sample_pairs() -> Path:
    """Path to the tiny pairwise fixture (6 rows, mirrors data/external/omct_pairs.json)."""
    return FIXTURE_DIR / "sample_pairs.json"


@pytest.fixture()
def write_config(tmp_path: Path):
    """Factory fixture: write a minimal YAML config pointing at a dataset.

    The config uses absolute paths under ``tmp_path`` so tests never touch the
    real ``data/`` or ``results/`` directories. Pass ``extra`` to append
    additional YAML sections.
    """

    def _write_config(
        data_raw: Path,
        run_id: str | None = "cli-test",
        extra: str = "",
        data_external: Path | None = None,
        experiment_table: Path | None = None,
    ) -> Path:
        config_file = tmp_path / "cfg.yaml"
        run_id_line = f"run_id: {run_id}\n" if run_id else ""
        external_line = f"  data_external: {data_external.as_posix()}\n" if data_external else ""
        table_line = (
            f"  experiment_table: {experiment_table.as_posix()}\n" if experiment_table else ""
        )
        config_file.write_text(
            run_id_line + "paths:\n"
            f"  data_raw: {data_raw.as_posix()}\n"
            f"  data_processed: {(tmp_path / 'processed').as_posix()}\n"
            f"  data_splits: {(tmp_path / 'splits').as_posix()}\n"
            + external_line
            + table_line
            + f"  cache: {(tmp_path / 'cache').as_posix()}\n"
            f"  results: {(tmp_path / 'results').as_posix()}\n" + extra,
            encoding="utf-8",
        )
        return config_file

    return _write_config
