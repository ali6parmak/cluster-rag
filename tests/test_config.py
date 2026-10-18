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


def test_load_config_stats_section(tmp_path) -> None:
    """The stats section validates, overrides the schema defaults and rejects typos."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nstats:\n  frequency_head_min: 77\n", encoding="utf-8")

    config = load_config(config_file)
    assert config.stats.frequency_head_min == 77
    assert config.stats.frequency_torso_min == 40  # untouched default

    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG_YAML + "\nstats:\n  frequency_head_minm: 77\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_embed_section(tmp_path) -> None:
    """The embed section validates backends/prefixes for both backends."""
    hf = CONFIG_YAML + (
        "\nembed:"
        "\n  backend: hf"
        "\n  model_id: sentence-transformers/all-MiniLM-L6-v2"
        "\n  revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        "\n  device: cuda"
        "\n  prefixes:"
        "\n    query: 'query: '"
        "\n    document: 'passage: '"
    )
    config = load_config_str(hf, tmp_path)
    assert config.embed.backend == "hf"
    assert config.embed.prefixes.query == "query: " and config.embed.prefixes.label == ""
    assert config.embed.batch_size == 64  # untouched default
    assert config.preprocess.variant == "raw" and config.cache.enabled

    ollama = hf.replace("backend: hf", "backend: ollama").replace(
        "  revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf", "  host: http://h:1234"
    )
    config = load_config_str(ollama, tmp_path)
    assert config.embed.host == "http://h:1234" and config.embed.timeout == 120.0

    bad = hf + "\n  batch_size: 0\n"
    with pytest.raises(ValueError):
        load_config_str(bad, tmp_path)


def test_load_config_extends_merges_parent(tmp_path) -> None:
    """extends merges a parent tree underneath; child wins in deep merge."""
    parent = tmp_path / "base.yaml"
    parent.write_text(CONFIG_YAML + "\nseed: 7\nrun_id: parent\n", encoding="utf-8")
    child = tmp_path / "exp.yaml"
    child.write_text(
        "extends: base.yaml\nseed: 99\npaths:\n  cache: other-cache\n", encoding="utf-8"
    )
    config = load_config(child)
    assert config.seed == 99  # child override
    assert config.run_id == "parent"  # parent value kept
    assert config.paths.cache == Path("other-cache")  # deep merge into paths
    assert config.paths.data_raw == Path("data/raw/upr.jsonl")  # parent paths kept


def test_load_config_extends_supports_chains_and_rejects_cycles(tmp_path) -> None:
    """Grandparents merge too; a extends cycle fails instead of hanging."""
    grand = tmp_path / "grand.yaml"
    grand.write_text(CONFIG_YAML + "\nrun_id: grand\n", encoding="utf-8")
    parent = tmp_path / "parent.yaml"
    parent.write_text("extends: grand.yaml\nseed: 7\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\n", encoding="utf-8")
    config = load_config(child)
    assert (config.seed, config.run_id) == (7, "grand")


def test_extends_cycle_direct(tmp_path) -> None:
    """A config extending itself is rejected."""
    child = tmp_path / "selfref.yaml"
    child.write_text(CONFIG_YAML + "\nextends: selfref.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle|deep"):
        load_config(child)


def load_config_str(text: str, directory: Path):
    """Helper: write a yaml string under directory and load it."""
    config_file = directory / "cfg.yaml"
    config_file.write_text(text, encoding="utf-8")
    return load_config(config_file)
