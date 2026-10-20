"""Tests for the Python sweep infrastructure (Phase 2.5).

The contract under test: cells resolve into validated AppConfigs (pydantic
covers every override), labels double as run_ids and must be unique, and the
`sweep` CLI command executes the resolved cells through the existing command
functions — in order, aborting on the first failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cluster_rag import cli
from cluster_rag.cli import main
from cluster_rag.sweep import COMMANDS, SweepCell, SweepRun, define_sweep

BASE_YAML = """\
run_id: base
seed: 7
paths:
  data_raw: data/raw/x.jsonl
  data_processed: processed
  data_splits: splits
  cache: cache
  results: results
"""


@pytest.fixture()
def base_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(BASE_YAML, encoding="utf-8")
    return path


def test_cell_validation_rejects_bad_command_and_label() -> None:
    """Unknown commands and unsafe labels fail at cell construction time."""
    with pytest.raises(ValueError, match="Unknown sweep command"):
        SweepCell(label="x", command="magic")
    with pytest.raises(ValueError, match="filesystem-safe"):
        SweepCell(label="sub/dir", command="embed")
    with pytest.raises(ValueError, match="Extra inputs"):
        SweepCell(label="x", command="embed", run_id="y")  # type: ignore[call-arg]
    assert COMMANDS == ("embed", "evaluate", "probe-aux")


def test_define_sweep_resolves_cells_over_base(base_yaml: Path) -> None:
    """Cells merge over the base (defaults flow through, overrides win)."""
    cells = [
        SweepCell(
            label="probe2-a",
            command="probe-aux",
            overrides={
                "embed": {"backend": "hf", "model_id": "model-a"},
                "probe": {"recall_ks": [1, 5]},
            },
        ),
        SweepCell(
            label="embed-b",
            command="embed",
            overrides={"embed": {"backend": "hf", "model_id": "model-b"}},
        ),
    ]
    runs = define_sweep(cells, base=base_yaml)
    assert isinstance(runs[0], SweepRun)
    assert [run.label for run in runs] == ["probe2-a", "embed-b"]
    assert [run.command for run in runs] == ["probe-aux", "embed"]
    assert [run.config.run_id for run in runs] == ["probe2-a", "embed-b"]
    # overrides applied
    assert runs[0].config.embed is not None and runs[0].config.embed.model_id == "model-a"
    assert runs[1].config.embed is not None and runs[1].config.embed.model_id == "model-b"
    assert runs[0].config.probe.recall_ks == (1, 5)
    # base values flow through untouched
    assert all(run.config.seed == 7 for run in runs)
    assert runs[0].config.preprocess.variant == "raw"
    assert runs[0].config.probe.question_transforms == ("strip_block",)


def test_define_sweep_rejects_duplicates_and_forbidden_overrides(base_yaml: Path) -> None:
    """Duplicate labels collide on run_ids; run_id/extends overrides are banned."""
    cells = [
        SweepCell(
            label="same", command="embed", overrides={"embed": {"backend": "hf", "model_id": "a"}}
        ),
        SweepCell(
            label="same",
            command="probe-aux",
            overrides={"embed": {"backend": "hf", "model_id": "b"}},
        ),
    ]
    with pytest.raises(ValueError, match="duplicate sweep cell label"):
        define_sweep(cells, base=base_yaml)

    cells_single = [SweepCell(label="x", command="embed", overrides={"run_id": "y"})]
    with pytest.raises(ValueError, match="run_id comes from the label"):
        define_sweep(cells_single, base=base_yaml)

    cells_extends = [SweepCell(label="x", command="embed", overrides={"extends": "other.yaml"})]
    with pytest.raises(ValueError, match="extends belongs to the base"):
        define_sweep(cells_extends, base=base_yaml)


def test_define_sweep_validates_override_values(base_yaml: Path) -> None:
    """An invalid override fails resolution (pydantic) before any run starts."""
    cells = [
        SweepCell(
            label="x", command="probe-aux", overrides={"probe": {"question_transforms": ["wrong"]}}
        )
    ]
    with pytest.raises(ValueError, match="Unknown question transform"):
        define_sweep(cells, base=base_yaml)


def _write_sweep_file(tmp_path: Path, base_yaml: Path, cells_literal: str) -> Path:
    """A temporary sweep file (module) exposing SWEEP built from `cells_literal`."""
    sweep_file = tmp_path / "sweep_file.py"
    sweep_file.write_text(
        "from cluster_rag.sweep import SweepCell, define_sweep\n"
        "CELLS = " + cells_literal + "\n"
        f"SWEEP = define_sweep(CELLS, base={str(base_yaml)!r})\n",
        encoding="utf-8",
    )
    return sweep_file


TWO_CELLS = (
    "[SweepCell(label='embed-a', command='embed', overrides=({'embed': {'backend': 'hf', "
    "'model_id': 'a'}})), SweepCell(label='probe-b', command='probe-aux', overrides="
    "({'embed': {'backend': 'hf', 'model_id': 'b'}}))]"
)


def test_sweep_command_executes_cells_in_order(
    tmp_path: Path, base_yaml: Path, monkeypatch, capsys
) -> None:
    """Cells run through the existing command functions, in order, with summaries."""
    calls: list[tuple[str, str]] = []

    def fake_embed(config, kinds):
        calls.append(("embed", config.run_id))
        return 0

    def fake_probe(config):
        calls.append(("probe-aux", config.run_id))
        return 0

    monkeypatch.setattr(cli, "_cmd_embed", fake_embed)
    monkeypatch.setattr(cli, "_cmd_probe_aux", fake_probe)
    sweep_file = _write_sweep_file(tmp_path, base_yaml, TWO_CELLS)
    assert main(["sweep", "--file", str(sweep_file)]) == 0
    out = capsys.readouterr().out
    assert calls == [("embed", "embed-a"), ("probe-aux", "probe-b")]
    assert "Sweep" in out and "2 cell(s)" in out and "Sweep finished" in out


def test_sweep_command_aborts_on_first_failure(
    tmp_path: Path, base_yaml: Path, monkeypatch, caplog
) -> None:
    """A failing cell aborts the sweep with its exit code; later cells do not run."""
    calls: list[str] = []

    monkeypatch.setattr(
        cli, "_cmd_embed", lambda config, kinds: (calls.append(config.run_id), 1)[1]
    )
    monkeypatch.setattr(cli, "_cmd_probe_aux", lambda config: (calls.append(config.run_id), 0)[1])
    sweep_file = _write_sweep_file(tmp_path, base_yaml, TWO_CELLS)
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(sweep_file)]) == 1
    assert calls == ["embed-a"]  # probe-b never executed
    assert any("aborting the sweep" in record.message for record in caplog.records)


def test_sweep_command_rejects_bad_files(tmp_path: Path, base_yaml: Path, caplog) -> None:
    """Missing SWEEP / broken module / missing file fail with clear messages."""
    bare = tmp_path / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(bare)]) == 1
    assert any("SWEEP" in record.message for record in caplog.records)

    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(broken)]) == 1
    assert any("boom" in record.message for record in caplog.records)

    assert main(["sweep", "--file", str(tmp_path / "missing.py")]) == 1
