"""CLI smoke tests: fast, no network, no models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from cluster_rag.cli import main
from cluster_rag.commands._common import _append_command_rows


def test_help_exits_zero() -> None:
    """Phase 0 exit criterion: the CLI help works."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_stats_command_on_fixture(sample_jsonl, write_config, tmp_path, capsys) -> None:
    """`stats` runs end-to-end on the fixture and writes the run artifacts."""
    config_file = write_config(sample_jsonl)
    assert main(["stats", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "Records: 20 valid, 0 malformed line(s) dropped" in out
    assert "Unique labels" in out
    assert "Frequency buckets" in out
    assert "Children" in out  # a frequent fixture label appears in the top list

    results_dir = tmp_path / "results" / "cli-test"
    stats_path = results_dir / "stats.json"
    assert (results_dir / "config.yaml").is_file()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "cli-test"
    assert payload["source"].endswith("sample.jsonl")
    assert payload["n_valid_records"] == 20
    assert payload["n_malformed_lines"] == 0
    assert payload["dataset_stats"]["overview"]["n_records"] == 20


def test_stats_command_warns_on_unprefixed_run_id(
    sample_jsonl, write_config, tmp_path, caplog
) -> None:
    """UPR commands only warn (never fail) when the run_id lacks the upr- prefix."""
    config_file = write_config(sample_jsonl)  # run_id "cli-test" carries no upr- prefix
    with caplog.at_level(logging.WARNING):
        assert main(["stats", "--config", str(config_file)]) == 0
    assert any("upr-" in record.message for record in caplog.records)
    assert (tmp_path / "results" / "cli-test" / "stats.json").is_file()


def test_stats_command_requires_run_id(sample_jsonl, write_config, caplog) -> None:
    """Without run_id the command refuses to run: artifacts need a folder name."""
    config_file = write_config(sample_jsonl, run_id=None)
    with caplog.at_level(logging.ERROR):
        assert main(["stats", "--config", str(config_file)]) == 1
    assert any("run_id" in record.message for record in caplog.records)


def test_stats_command_missing_config_returns_error() -> None:
    assert main(["stats", "--config", "does-not-exist.yaml"]) == 1


def test_append_command_rows_uses_the_command_derived_file(tmp_path: Path) -> None:
    """The command decides the table file; the track's file stays separate.

    The text-processing track owns its own pair of comparison tables, so a
    track command's rows must land there and never in the main pipeline's
    per-dataset files.
    """
    from cluster_rag.config import AppConfig

    upr_table = tmp_path / "docs" / "experiments_table_upr.md"
    track_table = tmp_path / "docs" / "experiments_table_text_processing_omct.md"
    config = AppConfig.model_validate(
        {
            "run_id": "x",
            "paths": {
                "data_raw": "data/raw/x.jsonl",
                "data_processed": "data/processed",
                "data_splits": "data/splits",
                "cache": "cache",
                "results": str(tmp_path / "results"),
                "experiment_table_upr": str(upr_table),
                "experiment_table_text_processing_omct": str(track_table),
            },
        }
    )
    columns = ("run_id", "value")
    _append_command_rows(
        config, "evaluate", columns=columns, key_columns=("run_id",), rows=[{"run_id": "r-1"}]
    )
    _append_command_rows(
        config,
        "analyze-omct",
        columns=columns,
        key_columns=("run_id",),
        rows=[{"run_id": "r-2"}],
    )
    assert "| r-1 |" in upr_table.read_text(encoding="utf-8")
    assert "| r-2 |" in track_table.read_text(encoding="utf-8")
    assert "r-2" not in upr_table.read_text(encoding="utf-8")
