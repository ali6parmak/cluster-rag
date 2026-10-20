"""CLI smoke tests: fast, no network, no models."""

from __future__ import annotations

import json
import logging

import pytest

from cluster_rag.cli import main


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
