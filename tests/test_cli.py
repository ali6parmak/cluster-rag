"""CLI smoke tests: fast, no network, no models."""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster_rag.cli import main


def _write_config(tmp_path: Path, data_raw: Path) -> Path:
    """Point a minimal config at a dataset path (absolute, as written in the YAML)."""
    config_file = tmp_path / "cli.yaml"
    config_file.write_text(
        "paths:\n"
        f"  data_raw: {data_raw.as_posix()}\n"
        "  data_processed: processed\n"
        "  data_splits: splits\n"
        "  cache: cache\n"
        "  results: results\n",
        encoding="utf-8",
    )
    return config_file


def test_help_exits_zero() -> None:
    """Phase 0 exit criterion: the CLI help works."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_stats_command_on_fixture(sample_jsonl, tmp_path, capsys) -> None:
    """`stats` runs end-to-end on the fixture dataset and prints a summary."""
    config_file = _write_config(tmp_path, sample_jsonl)
    assert main(["stats", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "Records: 20 valid, 0 malformed line(s) dropped" in out
    assert "Unique labels" in out
    assert "Children" in out  # a frequent fixture label appears in the top list


def test_stats_command_missing_config_returns_error() -> None:
    assert main(["stats", "--config", "does-not-exist.yaml"]) == 1
