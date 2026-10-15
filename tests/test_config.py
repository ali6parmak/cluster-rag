"""Tests for the YAML config system."""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster_rag.config import load_config

CONFIG_YAML = """
paths:
  data_raw: data/raw/upr.jsonl
  data_processed: data/processed
  data_splits: data/splits
  cache: cache
  results: results
"""


def test_load_config_parses_values(tmp_path) -> None:
    """A valid config validates and keeps relative paths untouched."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nseed: 7\nrun_id: test-run\n", encoding="utf-8")

    config = load_config(config_file)
    assert config.seed == 7
    assert config.run_id == "test-run"
    assert config.paths.data_raw == Path("data/raw/upr.jsonl")
    assert config.paths.label_catalog is None


def test_load_config_defaults(tmp_path) -> None:
    """run_id and seed have defaults; only paths are required."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")

    config = load_config(config_file)
    assert config.seed == 42
    assert config.run_id is None


def test_load_config_rejects_missing_paths(tmp_path) -> None:
    """A required path missing from the YAML is a validation error."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("paths:\n  data_raw: data/raw/upr.jsonl\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_file)


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    """Typos in a config file (unknown keys) fail immediately."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nembeddr: typo-section\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_file)


def test_load_config_rejects_non_mapping(tmp_path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_file)
