"""Tests for the typed Python sweep infrastructure (Phases 2.5 + 3.5).

Contracts under test: every cell resolves over configs/base.yaml into a
validated AppConfig (pydantic covers every override value), labels are
explicit filesystem-safe run ids with dataset prefix rules, a sweep is
exactly one dataset (its allowed command set), the sweep label is injected
as ``table.sweep_label``, and the `sweep` CLI command executes the resolved
cells through the existing command functions — in order, aborting on the
first failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from cluster_rag.cli import main
from cluster_rag.commands import sweep as sweep_commands
from cluster_rag.config import EmbedderConfig, EvalDecisionConfig
from cluster_rag.sweep import (
    COMMANDS,
    DATASET_COMMANDS,
    SHARED_EXPERIMENTS_DIR,
    AnalyzeUprCell,
    EmbedCell,
    EvalCell,
    LabelEmbedCell,
    ProbeCell,
    SweepCell,
    SweepRun,
    define_sweep,
)

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
def repo_root(tmp_path: Path, monkeypatch) -> Path:
    """A fixture repo root: define_sweep resolves the literal configs/base.yaml
    path against the CWD, so the test builds its own controlled base file."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_sweep_cell_validation() -> None:
    """Generic cells: known commands only, filesystem-safe labels, no extras."""
    with pytest.raises(ValueError, match="Unknown sweep command"):
        SweepCell(label="x", command="magic")
    with pytest.raises(ValueError, match="filesystem-safe"):
        SweepCell(label="sub/dir", command="embed")
    with pytest.raises(ValueError, match="Extra inputs"):
        SweepCell(label="x", command="embed", run_id="y")  # type: ignore[call-arg]
    assert COMMANDS == (
        "stats",
        "stats-omct",
        "split",
        "embed",
        "labelembed",
        "evaluate",
        "probe-omct",
        "analyze-omct",
        "decide-omct",
        "analyze-upr",
        "decide-upr",
        "rerank-upr",
    )
    assert DATASET_COMMANDS == {
        "upr": (
            "stats",
            "split",
            "embed",
            "labelembed",
            "evaluate",
            "analyze-upr",
            "decide-upr",
            "rerank-upr",
        ),
        "omct": ("stats-omct", "probe-omct", "analyze-omct", "decide-omct"),
    }


def test_eval_cell_requires_decision_xor_k() -> None:
    """`decision` and the `k` shorthand are exactly one thing."""
    with pytest.raises(ValueError, match="not both"):
        EvalCell(label="upr-eval-x", predictor="random", k=2, decision=EvalDecisionConfig(k=2))
    with pytest.raises(ValueError, match="needs a decision rule"):
        EvalCell(label="upr-eval-x", predictor="random")
    # k is validated with the field first (a top_k shorthand must be positive)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        EvalCell(label="upr-eval-x", predictor="random", k=0)
    assert EvalCell(label="upr-eval-x", predictor="random", k=2).decision is None


def test_typed_cells_reject_unknown_fields() -> None:
    """Typed cells forbid extra fields and validate their values up front."""
    with pytest.raises(ValueError, match="Extra inputs"):
        EvalCell(label="upr-eval-x", predictor="random", k=2, run_id="y")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Extra inputs"):
        EmbedCell(label="upr-embed-x", embed=EmbedderConfig(backend="hf", model_id="m"), k=2)
    with pytest.raises(ValueError, match="Input should be 'val' or 'test'"):
        EvalCell(label="upr-eval-x", predictor="random", k=2, split="train")
    with pytest.raises(ValueError, match="filesystem-safe"):
        EvalCell(label="sub/dir", predictor="random", k=2)


def test_embed_cell_kinds_field() -> None:
    """EmbedCell.kinds validates (known, non-empty) and lands in the overrides."""
    default = EmbedCell(label="upr-embed-a", embed=EmbedderConfig(backend="hf", model_id="m"))
    assert default.kinds == ("document",)
    assert default.overrides["embed_kinds"] == ["document"]

    both = EmbedCell(
        label="upr-embed-b",
        embed=EmbedderConfig(backend="hf", model_id="m"),
        kinds=("document", "query"),
    )
    assert both.overrides["embed_kinds"] == ["document", "query"]

    with pytest.raises(ValueError, match="Unknown encode kind"):
        EmbedCell(
            label="upr-embed-c",
            embed=EmbedderConfig(backend="hf", model_id="m"),
            kinds=("vectors",),
        )
    with pytest.raises(ValueError, match="must name at least one encode kind"):
        EmbedCell(label="upr-embed-d", embed=EmbedderConfig(backend="hf", model_id="m"), kinds=())


def test_define_sweep_resolves_embed_kinds(repo_root: Path) -> None:
    """An EmbedCell's kinds become the resolved config's embed_kinds."""
    cells = [
        EmbedCell(
            label="upr-embed-a",
            embed=EmbedderConfig(backend="hf", model_id="m"),
            kinds=("document", "query"),
        )
    ]
    runs = define_sweep(cells, dataset="upr", label="embed_pool")
    assert runs[0].config.embed_kinds == ("document", "query")


def test_sweep_file_imports_sibling_module(repo_root: Path) -> None:
    """The sweep loader exposes the file's directory for sibling imports.

    Sweep files share the embedder pool (experiments/embedder_pool.py); the
    loader must therefore make the file's directory importable while it
    executes and restore sys.path afterwards. The files live in a subdirectory
    of the fixture root, so the import works only via the loader, never via
    an ambient sys.path entry.
    """
    subdir = repo_root / "pool_sweeps"
    subdir.mkdir()
    (subdir / "mini_pool.py").write_text("MODEL_IDS = ['mini-a', 'mini-b']\n", encoding="utf-8")
    (subdir / "mini_sweep.py").write_text(
        "from mini_pool import MODEL_IDS\n"
        "from cluster_rag.sweep import SweepCell, define_sweep\n"
        "cells = [SweepCell(label='upr-stats-' + m, command='stats') for m in MODEL_IDS]\n"
        "SWEEP = define_sweep(cells, dataset='upr', label='mini')\n",
        encoding="utf-8",
    )
    subdir_resolved = str(subdir.resolve())
    assert subdir_resolved not in sys.path
    runs = sweep_commands._load_sweep_runs(subdir / "mini_sweep.py")
    assert [run.label for run in runs] == ["upr-stats-mini-a", "upr-stats-mini-b"]
    assert subdir_resolved not in sys.path  # removed again after the load


def test_sweep_file_imports_the_shared_pool_from_another_directory(repo_root: Path) -> None:
    """The shared embedder pool stays importable from any sweep directory.

    The text-processing track's sweeps live in `experiments_text_processing/`
    (parallel to `experiments/`), but they must use the same pool as the main
    pipeline: the loader keeps `experiments/` on sys.path while the sweep file
    executes, so `from embedder_pool import MODELS` works there too.
    """
    subdir = repo_root / "experiments_text_processing"
    subdir.mkdir()
    (subdir / "track_sweep.py").write_text(
        "from embedder_pool import MODELS\n"
        "from cluster_rag.sweep import SweepCell, define_sweep\n"
        "model = sorted(MODELS)[0].replace('/', '_')\n"
        "cells = [SweepCell(label='upr-stats-' + model, command='stats')]\n"
        "SWEEP = define_sweep(cells, dataset='upr', label='track')\n",
        encoding="utf-8",
    )
    # The import must go through the loader, not through a cached module.
    sys.modules.pop("embedder_pool", None)
    shared_resolved = str(SHARED_EXPERIMENTS_DIR.resolve())
    runs = sweep_commands._load_sweep_runs(subdir / "track_sweep.py")
    pool = sys.modules.get("embedder_pool")
    assert pool is not None, "the sweep file did not import the shared embedder pool"
    assert sorted(pool.MODELS)[0].replace("/", "_") in runs[0].label
    assert shared_resolved not in sys.path  # removed again after the load


def test_the_random_floor_sweep_resolves_its_grid(monkeypatch) -> None:
    """The CS-6d grid: 3 models x 3 group sets, the random-vector floor.

    The sweep file is a real artifact of the track, so its grid is a contract:
    every Stage-3.3 arm gets its random-vector floor, and the floor's knob is
    on in every cell.
    """
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    runs = sweep_commands._load_sweep_runs(
        repo / "experiments_text_processing" / "stage3_random_floor_sweep.py"
    )
    assert len(runs) == 9
    assert sorted(run.label for run in runs) == sorted(
        f"analyze-{model}-stage3-rand-{group}"
        for model in ("embeddinggemma", "voyage-4-nano", "arctic2")
        for group in ("base", "full", "profile")
    )
    for run in runs:
        assert run.command == "analyze-omct"
        assert run.config.table.sweep_label == "stage3_random_floor"
        assert run.config.analysis is not None
        assert run.config.analysis.stage3 is not None
        # the floor's knob is on, and the combiner is the logistic regression
        # (CS-6b: the learner is not the lever)
        assert run.config.analysis.stage3.embeddings == "random"
        assert run.config.analysis.stage3.combiner.name == "logistic"


def test_sweep_command_fails_loudly_on_an_unimplemented_command(repo_root: Path) -> None:
    """A cell whose command the CLI does not implement aborts, never guesses."""
    (repo_root / "bogus_sweep.py").write_text(
        "from cluster_rag.sweep import SweepCell, SweepRun, define_sweep\n"
        "runs = define_sweep([SweepCell(label='upr-stats-x', command='stats')],"
        " dataset='upr', label='bogus')\n"
        "SWEEP = [SweepRun(label=runs[0].label, command='not-a-command',"
        " config=runs[0].config)]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not implement"):
        sweep_commands._cmd_sweep(repo_root / "bogus_sweep.py")


def test_define_sweep_resolves_typed_cells(repo_root: Path) -> None:
    """Typed cells merge over the fixture base (defaults flow, overrides win)."""
    cells = [
        EvalCell(label="upr-eval-a", predictor="random", k=2, split="val"),
        EvalCell(
            label="upr-eval-b",
            predictor="most_frequent",
            decision=EvalDecisionConfig(rule="top_k_min_score", k=3, min_score=0.1),
            metrics=None,
        ),
        SweepCell(label="upr-stats-base", command="stats"),
    ]
    runs = define_sweep(cells, dataset="upr", label="baselines")
    assert isinstance(runs[0], SweepRun)
    assert [run.label for run in runs] == ["upr-eval-a", "upr-eval-b", "upr-stats-base"]
    assert [run.command for run in runs] == ["evaluate", "evaluate", "stats"]
    assert [run.config.run_id for run in runs] == ["upr-eval-a", "upr-eval-b", "upr-stats-base"]
    # the sweep label is injected into every resolved config
    assert all(run.config.table.sweep_label == "baselines" for run in runs)
    # the EvalCell shorthand becomes a full top_k decision
    assert runs[0].config.eval.predictor.name == "random"
    assert runs[0].config.eval.decision.rule == "top_k" and runs[0].config.eval.decision.k == 2
    assert runs[0].config.eval.split == "val"
    # the full decision passes through unchanged
    assert runs[1].config.eval.decision.rule == "top_k_min_score"
    assert runs[1].config.eval.decision.min_score == 0.1
    # base values flow through untouched
    assert all(run.config.seed == 7 for run in runs)
    assert runs[2].config.preprocess.variant == "raw"


def test_define_sweep_rejects_unknown_dataset_and_command(repo_root: Path) -> None:
    """dataset is closed over {upr, omct}; cells must run that dataset's commands."""
    with pytest.raises(ValueError, match="Unknown sweep dataset"):
        define_sweep(
            [EvalCell(label="upr-eval-x", predictor="random", k=2)], dataset="unknown", label="x"
        )
    with pytest.raises(ValueError, match="not a 'omct' sweep command"):
        define_sweep([EvalCell(label="x", predictor="random", k=2)], dataset="omct", label="x")
    with pytest.raises(ValueError, match="not a 'upr' sweep command"):
        define_sweep([SweepCell(label="upr-x", command="stats-omct")], dataset="upr", label="x")
    with pytest.raises(ValueError, match="not a 'omct' sweep command"):
        define_sweep([SweepCell(label="x", command="split")], dataset="omct", label="x")


def test_analyze_upr_cell_resolves_the_reference_arm(repo_root: Path) -> None:
    """The analyze-upr cell: the eval wiring + the upr_analysis section."""
    from cluster_rag.config import UprAnalysisConfig

    cell = AnalyzeUprCell(
        label="upr-analyze-x",
        predictor="fusion",
        params={"a": 0.5, "b": 1.0, "c": 1.0, "k": 10},
        k=2,
        embedding_run="upr-embed-fix",
        label_embedding_run="upr-labelembed-fix",
        analysis=UprAnalysisConfig(length_buckets=(100, 200), reference_eval_run="upr-eval-ref"),
    )
    assert cell.command == "analyze-upr"
    overrides = cell.overrides
    assert overrides["eval"]["predictor"]["name"] == "fusion"
    assert overrides["eval"]["embedding_run_id"] == "upr-embed-fix"
    assert overrides["eval"]["label_embedding_run_id"] == "upr-labelembed-fix"
    assert overrides["eval"]["decision"] == {
        "rule": "top_k",
        "k": 2,
        "threshold": None,
        "min_score": None,
    }
    assert overrides["upr_analysis"]["reference_eval_run"] == "upr-eval-ref"
    assert overrides["upr_analysis"]["length_buckets"] == (100, 200)
    runs = define_sweep([cell], dataset="upr", label="upr_analysis")
    assert runs[0].config.run_id == "upr-analyze-x"
    assert runs[0].config.upr_analysis is not None
    assert runs[0].config.upr_analysis.reference_eval_run == "upr-eval-ref"


def test_analyze_upr_cell_needs_a_decision_and_an_upr_prefix(repo_root: Path) -> None:
    """The cell's decision xor k rule and the upr- label prefix."""
    from cluster_rag.config import UprAnalysisConfig

    with pytest.raises(ValueError, match="not both"):
        AnalyzeUprCell(
            label="upr-analyze-x",
            predictor="fusion",
            k=2,
            decision=EvalDecisionConfig(rule="top_k", k=3),
            embedding_run="upr-embed-fix",
            analysis=UprAnalysisConfig(),
        )
    with pytest.raises(ValueError, match="needs a decision rule"):
        AnalyzeUprCell(
            label="upr-analyze-x",
            predictor="fusion",
            embedding_run="upr-embed-fix",
            analysis=UprAnalysisConfig(),
        )
    with pytest.raises(ValueError, match="must start with 'upr-'"):
        define_sweep(
            [
                AnalyzeUprCell(
                    label="analyze-x",
                    predictor="fusion",
                    k=2,
                    embedding_run="upr-embed-fix",
                    analysis=UprAnalysisConfig(),
                )
            ],
            dataset="upr",
            label="upr_analysis",
        )


def test_define_sweep_label_prefix_rules(repo_root: Path) -> None:
    """upr cell labels carry upr-; omct cell labels are unprefixed."""
    with pytest.raises(ValueError, match="must start with 'upr-'"):
        define_sweep([EvalCell(label="eval-x", predictor="random", k=2)], dataset="upr", label="x")
    with pytest.raises(ValueError, match="must not carry a dataset prefix"):
        define_sweep(
            [ProbeCell(label="omct-probe-x", embed=EmbedderConfig(backend="hf", model_id="m"))],
            dataset="omct",
            label="x",
        )
    with pytest.raises(ValueError, match="must not carry a dataset prefix"):
        define_sweep(
            [ProbeCell(label="upr-probe-x", embed=EmbedderConfig(backend="hf", model_id="m"))],
            dataset="omct",
            label="x",
        )
    # the sweep label itself is free-form (heading only), filesystem-safe only
    runs = define_sweep(
        [ProbeCell(label="probe-x", embed=EmbedderConfig(backend="hf", model_id="m"))],
        dataset="omct",
        label="probe2",
    )
    assert runs[0].config.table.sweep_label == "probe2"
    with pytest.raises(ValueError, match="filesystem-safe"):
        define_sweep(
            [ProbeCell(label="probe-x", embed=EmbedderConfig(backend="hf", model_id="m"))],
            dataset="omct",
            label="pro be",
        )


def test_define_sweep_rejects_duplicates_and_forbidden_overrides(repo_root: Path) -> None:
    """Duplicate labels collide on run_ids; run_id/extends/table.sweep_label are banned."""
    cells = [
        EvalCell(label="upr-eval-x", predictor="random", k=2),
        EvalCell(label="upr-eval-x", predictor="most_frequent", k=2),
    ]
    with pytest.raises(ValueError, match="duplicate sweep cell label"):
        define_sweep(cells, dataset="upr", label="x")

    with pytest.raises(ValueError, match="run_id comes from the label"):
        define_sweep(
            [SweepCell(label="upr-x", command="embed", overrides={"run_id": "y"})],
            dataset="upr",
            label="x",
        )
    with pytest.raises(ValueError, match="extends belongs to the base"):
        define_sweep(
            [SweepCell(label="upr-x", command="embed", overrides={"extends": "other.yaml"})],
            dataset="upr",
            label="x",
        )
    with pytest.raises(ValueError, match="table.sweep_label is injected"):
        define_sweep(
            [SweepCell(label="upr-x", command="embed", overrides={"table": {"sweep_label": "y"}})],
            dataset="upr",
            label="x",
        )


def test_define_sweep_validates_override_values(repo_root: Path) -> None:
    """An invalid override fails resolution (pydantic) before any run starts."""
    cells = [
        ProbeCell(
            label="probe-x",
            embed=EmbedderConfig(backend="hf", model_id="m"),
            probe=None,
            preprocess=None,
        )
    ]
    # a valid typed cell resolves fine (probe defaults keep the schema default)
    runs = define_sweep(cells, dataset="omct", label="probe")
    assert runs[0].config.probe.question_transforms == ("strip_block",)

    bad = [
        SweepCell(
            label="probe-y",
            command="probe-omct",
            overrides={"probe": {"question_transforms": ["wrong"]}},
        )
    ]
    with pytest.raises(ValueError, match="Unknown question transform"):
        define_sweep(bad, dataset="omct", label="probe")


def _write_sweep_file(tmp_path: Path, cells_literal: str) -> Path:
    """A temporary sweep file (module) exposing SWEEP built from `cells_literal`."""
    sweep_file = tmp_path / "sweep_file.py"
    sweep_file.write_text(
        "from cluster_rag.config import EmbedderConfig\n"
        "from cluster_rag.sweep import EvalCell, EmbedCell, ProbeCell, SweepCell, define_sweep\n"
        "CELLS = " + cells_literal + "\n"
        'SWEEP = define_sweep(CELLS, dataset="upr", label="sweep-label")\n',
        encoding="utf-8",
    )
    return sweep_file


TWO_CELLS = (
    "[EmbedCell(label='upr-embed-a', embed=EmbedderConfig(backend='hf', model_id='a')), "
    "EvalCell(label='upr-eval-b', predictor='random', k=2, split='val')]"
)


def test_sweep_command_executes_cells_in_order(repo_root: Path, monkeypatch, capsys) -> None:
    """Cells run through the existing command functions, in order, with summaries."""
    calls: list[tuple[str, str]] = []

    def fake_embed(config, kinds):
        calls.append(("embed", config.run_id))
        return 0

    def fake_evaluate(config):
        calls.append(("evaluate", config.run_id))
        return 0

    monkeypatch.setattr(sweep_commands, "_cmd_embed", fake_embed)
    monkeypatch.setattr(sweep_commands, "_cmd_evaluate", fake_evaluate)
    sweep_file = _write_sweep_file(repo_root, TWO_CELLS)
    assert main(["sweep", "--file", str(sweep_file)]) == 0
    out = capsys.readouterr().out
    assert calls == [("embed", "upr-embed-a"), ("evaluate", "upr-eval-b")]
    assert "Sweep" in out and "2 cell(s)" in out and "Sweep finished" in out


def test_sweep_command_runs_stats_by_command_name(repo_root: Path, monkeypatch) -> None:
    """`stats` and `stats-omct` cells dispatch to _cmd_stats with the omct flag."""
    calls: list[tuple[str, str, str]] = []

    def fake_stats(config, omct=False):
        calls.append(("stats", config.run_id, str(omct)))
        return 0

    monkeypatch.setattr(sweep_commands, "_cmd_stats", fake_stats)
    # One sweep per dataset (upr + omct in one file via SWEEP concatenation):
    sweep_file = repo_root / "stats_sweep.py"
    sweep_file.write_text(
        "from cluster_rag.sweep import SweepCell, define_sweep\n"
        "SWEEP = (\n"
        "    define_sweep([SweepCell(label='upr-stats-base', command='stats')],\n"
        "                 dataset='upr', label='s')\n"
        "    + define_sweep([SweepCell(label='stats-base', command='stats-omct')],\n"
        "                   dataset='omct', label='s')\n"
        ")\n",
        encoding="utf-8",
    )
    assert main(["sweep", "--file", str(sweep_file)]) == 0
    # the omct cell's config.run_id is the bare label; the command adds omct-
    assert calls == [
        ("stats", "upr-stats-base", "False"),
        ("stats", "stats-base", "True"),
    ]


def test_sweep_command_aborts_on_first_failure(repo_root: Path, monkeypatch, caplog) -> None:
    """A failing cell aborts the sweep with its exit code; later cells do not run."""
    calls: list[str] = []

    monkeypatch.setattr(
        sweep_commands, "_cmd_embed", lambda config, kinds: (calls.append(config.run_id), 1)[1]
    )
    monkeypatch.setattr(
        sweep_commands, "_cmd_evaluate", lambda config: (calls.append(config.run_id), 0)[1]
    )
    sweep_file = _write_sweep_file(repo_root, TWO_CELLS)
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(sweep_file)]) == 1
    assert calls == ["upr-embed-a"]  # upr-eval-b never executed
    assert any("aborting the sweep" in record.message for record in caplog.records)


def test_sweep_command_rejects_bad_files(repo_root: Path, caplog) -> None:
    """Missing SWEEP / broken module / missing file fail with clear messages."""
    bare = repo_root / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(bare)]) == 1
    assert any("SWEEP" in record.message for record in caplog.records)

    broken = repo_root / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        assert main(["sweep", "--file", str(broken)]) == 1
    assert any("boom" in record.message for record in caplog.records)

    assert main(["sweep", "--file", str(repo_root / "missing.py")]) == 1


def test_eval_cell_embedding_run_field() -> None:
    """EvalCell.embedding_run validates and lands as eval.embedding_run_id."""
    cell = EvalCell(
        label="upr-eval-knn-x",
        predictor="knn",
        params={"k": 3},
        k=2,
        embedding_run="upr-embed-e5-base-v2",
    )
    assert cell.overrides["eval"]["embedding_run_id"] == "upr-embed-e5-base-v2"
    default = EvalCell(label="upr-eval-x", predictor="random", k=2)
    assert default.overrides["eval"]["embedding_run_id"] is None
    # Only a UPR embed run may feed an eval: omct- or path-like values fail here.
    with pytest.raises(ValueError, match="upr-embed-"):
        EvalCell(label="upr-eval-x", predictor="knn", k=2, embedding_run="omct-probe2-x")
    with pytest.raises(ValueError, match="filesystem-safe"):
        EvalCell(label="upr-eval-x", predictor="knn", k=2, embedding_run="results/nope")


def test_define_sweep_resolves_embedding_run(repo_root: Path) -> None:
    """An EvalCell's embedding_run becomes the resolved config's eval key."""
    cells = [
        EvalCell(
            label="upr-eval-knn-a",
            predictor="knn",
            params={"k": 3},
            k=2,
            embedding_run="upr-embed-jina-v5",
        )
    ]
    runs = define_sweep(cells, dataset="upr", label="pool_sweep")
    assert runs[0].config.eval.embedding_run_id == "upr-embed-jina-v5"
    assert runs[0].config.eval.predictor.params == {"k": 3}


def test_eval_cell_label_embedding_run_field() -> None:
    """EvalCell.label_embedding_run validates and lands as eval.label_embedding_run_id."""
    cell = EvalCell(
        label="upr-eval-zeroshot-x",
        predictor="zeroshot",
        k=2,
        embedding_run="upr-embed-e5-base-v2",
        label_embedding_run="upr-labelembed-e5-base-v2-name",
    )
    assert cell.overrides["eval"]["label_embedding_run_id"] == "upr-labelembed-e5-base-v2-name"
    with pytest.raises(ValueError, match="upr-labelembed-"):
        EvalCell(label="upr-eval-x", predictor="zeroshot", k=2, label_embedding_run="upr-embed-x")


def test_label_embed_cell_field() -> None:
    """LabelEmbedCell carries the embedder + label_texts into its overrides."""
    from cluster_rag.config import LabelTextsConfig

    cell = LabelEmbedCell(
        label="upr-labelembed-jina-v5-about",
        embed=EmbedderConfig(backend="hf", model_id="m"),
        label_texts=LabelTextsConfig(variant="template", template="About {label}."),
    )
    assert cell.command == "labelembed"
    assert cell.overrides["label_texts"] == {"variant": "template", "template": "About {label}."}
    with pytest.raises(ValueError, match="Extra inputs"):
        LabelEmbedCell(
            label="upr-labelembed-x",
            embed=EmbedderConfig(backend="hf", model_id="m"),
            label_texts=LabelTextsConfig(),
            kinds=("label",),  # type: ignore[call-arg]
        )


def test_define_sweep_resolves_label_cells(repo_root: Path) -> None:
    """LabelEmbedCell and zeroshot EvalCells resolve over the base config."""
    from cluster_rag.config import LabelTextsConfig

    cells = [
        LabelEmbedCell(
            label="upr-labelembed-jina-v5-about",
            embed=EmbedderConfig(backend="hf", model_id="m"),
            label_texts=LabelTextsConfig(variant="template", template="About {label}."),
        ),
        EvalCell(
            label="upr-eval-zeroshot-jina-v5-about",
            predictor="zeroshot",
            k=2,
            embedding_run="upr-embed-jina-v5",
            label_embedding_run="upr-labelembed-jina-v5-about",
        ),
    ]
    runs = define_sweep(cells, dataset="upr", label="zeroshot")
    assert runs[0].command == "labelembed"
    assert runs[0].config.label_texts.render_text("General") == "About General."
    assert runs[1].config.eval.label_embedding_run_id == "upr-labelembed-jina-v5-about"
    assert runs[1].config.eval.embedding_run_id == "upr-embed-jina-v5"
