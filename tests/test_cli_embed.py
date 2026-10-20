"""Tests for the `embed` CLI command (Phase 2), with a fake embedder only.

Covers: artifact layout (embeddings.npz + meta.json), the cache-hit contract
end-to-end (second run encodes nothing), and the missing-embed-section error.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from cluster_rag import cli
from cluster_rag.cli import main
from fake_embedder import FakeEmbedder

EMBED_CONFIG_EXTRA = """
embed:
  backend: fake
  model_id: fake-embedder
  batch_size: 3
  prefixes:
    query: "query: "
    document: "passage: "
    label: ""
"""


@pytest.fixture()
def embed_run(sample_jsonl, write_config, monkeypatch, tmp_path):
    """Factory around the embed command with a per-run fake embedder."""
    created: list[FakeEmbedder] = []

    def factory(config):
        fake = FakeEmbedder(
            model_id="fake-embedder", dimension=6, prefixes={"document": "passage: "}
        )
        created.append(fake)
        return fake

    monkeypatch.setattr(cli, "build_embedder", factory)

    def run(command_args: list[str] | None = None, run_id: str = "embed-test") -> int:
        config_file = write_config(
            sample_jsonl,
            run_id=run_id,
            extra=EMBED_CONFIG_EXTRA + "preprocess:\n  variant: raw\n",
        )
        argv = ["embed", "--config", str(config_file)]
        if command_args:
            argv.extend(command_args)
        code = main(argv)
        results_dir = tmp_path / "results" / run_id
        meta = json.loads((results_dir / "meta.json").read_text("utf-8"))
        npz = np.load(results_dir / "embeddings.npz")
        return code, meta, npz, created

    return run


def test_embed_command_cold_run(embed_run, capsys) -> None:
    """First run: everything encoded from scratch, matrix written in id order."""
    code, meta, npz, created = embed_run()
    assert code == 0
    assert len(created) == 1
    n = meta["n_records"]
    assert n == 20
    kind = meta["kind_stats"]["document"]
    assert kind["n_vectors"] == 20 and kind["dimension"] == 6
    assert kind["cache_misses"] == n and kind["cache_hits"] == 0
    assert meta["cache"]["enabled"] is True and meta["cache"]["n_stored"] == n
    assert meta["embedder"]["backend"] == "fake"
    assert meta["preprocess"]["n_changed"] == 0
    ids = npz["ids"].tolist()
    document = npz["vectors_document"]
    assert document.shape == (n, 6) and document.dtype == np.float32
    assert len(ids) == document.shape[0]
    assert np.allclose(np.linalg.norm(document, axis=1), 1.0, atol=1e-6)
    # stdout summary mentions the cache accounting
    out = capsys.readouterr().out
    assert f"cache: 0 hit(s), {n} miss(es), {n} stored" in out


def test_embed_command_second_run_all_cache_hits(embed_run) -> None:
    """Rerun with the same config: the embedder is never asked for anything."""
    embed_run()
    code, meta, npz, created = embed_run()
    assert code == 0 and len(created) == 2
    assert created[1].call_count == 0  # fake embedder of the second run never encodes
    n = meta["n_records"]
    assert meta["cache"]["cache_hits"] == n and meta["cache"]["n_stored"] == 0
    assert npz["vectors_document"].shape == (n, 6)


def test_embed_command_cache_disabled(sample_jsonl, write_config, monkeypatch, tmp_path) -> None:
    """cache.enabled: false: the raw backend runs and counters stay at zero."""
    created: list[FakeEmbedder] = []

    def factory(config):
        fake = FakeEmbedder(
            model_id="fake-embedder", dimension=6, prefixes={"document": "passage: "}
        )
        created.append(fake)
        return fake

    monkeypatch.setattr(cli, "build_embedder", factory)
    config_file = write_config(
        sample_jsonl,
        run_id="embed-nocache",
        extra=EMBED_CONFIG_EXTRA + "\ncache:\n  enabled: false\npreprocess:\n  variant: raw\n",
    )
    assert main(["embed", "--config", str(config_file)]) == 0
    meta = json.loads((tmp_path / "results" / "embed-nocache" / "meta.json").read_text("utf-8"))
    assert meta["cache"]["enabled"] is False
    assert meta["cache"]["cache_hits"] == 0 and meta["cache"]["cache_misses"] == 0
    assert created[0].call_count == 1  # encoded exactly once (single kind)


def test_embed_appends_experiment_table_row(
    sample_jsonl, write_config, monkeypatch, tmp_path
) -> None:
    """paths.experiment_table_upr: one row per run, upserted by run_id."""

    def factory(config):
        return FakeEmbedder(
            model_id="fake-embedder", dimension=6, prefixes={"document": "passage: "}
        )

    monkeypatch.setattr(cli, "build_embedder", factory)
    table = tmp_path / "experiments_table.md"
    config_file = write_config(
        sample_jsonl,
        run_id="embed-table",
        experiment_table_upr=table,
        extra=EMBED_CONFIG_EXTRA + "preprocess:\n  variant: raw\n",
    )
    assert main(["embed", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## UPR — embed" in text
    data_rows = [line for line in text.splitlines() if line.startswith("| embed-table")]
    assert len(data_rows) == 1 and "| 20 |" in data_rows[0]

    assert main(["embed", "--config", str(config_file)]) == 0  # warm rerun
    text = table.read_text("utf-8")
    data_rows = [line for line in text.splitlines() if line.startswith("| embed-table")]
    assert len(data_rows) == 1  # replaced, not duplicated
    # the row now reports the cache-hit rerun counters
    cells = [cell.strip() for cell in data_rows[0].strip("|").split("|")]
    assert cells[-2:] == ["20", "0"]  # cache_hits / cache_misses
    assert "## UPR — embed" in text  # standalone run: the command names the section


def test_embed_table_row_under_sweep_heading(
    sample_jsonl, write_config, monkeypatch, tmp_path
) -> None:
    """A run with table.sweep_label lands under '## UPR — <sweep-label>'."""

    def factory(config):
        return FakeEmbedder(
            model_id="fake-embedder", dimension=6, prefixes={"document": "passage: "}
        )

    monkeypatch.setattr(cli, "build_embedder", factory)
    table = tmp_path / "experiments_table.md"
    config_file = write_config(
        sample_jsonl,
        run_id="upr-embed-swept",
        experiment_table_upr=table,
        extra=EMBED_CONFIG_EXTRA + "preprocess:\n  variant: raw\ntable:\n  sweep_label: my-sweep\n",
    )
    assert main(["embed", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## UPR — my-sweep" in text and "| upr-embed-swept |" in text


def test_embed_command_requires_embed_section(sample_jsonl, write_config, caplog) -> None:
    """Without an embed: section the command fails with a clear message."""
    config_file = write_config(sample_jsonl, run_id="no-embed")
    with caplog.at_level(logging.ERROR):
        assert main(["embed", "--config", str(config_file)]) == 1
    assert any("embed: section" in record.message for record in caplog.records)


def test_embed_command_kinds_flag(
    sample_jsonl, write_config, monkeypatch, tmp_path, caplog
) -> None:
    """Multiple kinds write one matrix per kind, cache keys separated by prefix."""
    created: list[FakeEmbedder] = []

    def factory(config):
        fake = FakeEmbedder(
            model_id="fake-embedder",
            dimension=6,
            prefixes={"query": "query: ", "document": "passage: "},
        )
        created.append(fake)
        return fake

    monkeypatch.setattr(cli, "build_embedder", factory)
    config_file = write_config(sample_jsonl, run_id="embed-multi", extra=EMBED_CONFIG_EXTRA)
    assert main(["embed", "--config", str(config_file), "--kinds", "document,query"]) == 0
    npz = np.load(tmp_path / "results" / "embed-multi" / "embeddings.npz")
    assert set(npz.files) == {"ids", "vectors_document", "vectors_query"}
    # both kinds encoded in one run; the cache separates them via the prefix key
    assert len(created) == 1 and created[0].call_count == 2
    assert created[0].encode_calls[0] == ("document", created[0].encode_calls[0][1])
    assert created[0].encode_calls[1][0] == "query"

    with caplog.at_level(logging.ERROR):
        assert main(["embed", "--config", str(config_file), "--kinds", "documents"]) == 1
    assert any("expected one of" in record.message for record in caplog.records)
