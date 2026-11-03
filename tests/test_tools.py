"""Tests for the repo tools over the run artifacts.

The text-processing track's tools (``tools/verify_analysis_baseline.py``,
``tools/compare_stage3_arms.py``) are verification scripts over the run
artifacts. They are not part of the package, so they are loaded here from the
repository's ``tools/`` folder.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name: str) -> Any:
    """Import a tool module from the repository's tools folder."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(scores: list[float]) -> dict[str, Any]:
    """A minimal analysis artifact with one model's out-of-fold scores."""
    judged = []
    for index in range(len(scores)):
        judged.append(
            {
                "pair_id": f"pair-{index}",
                "document": "Doc A" if index < 4 else "Doc B",
                "label": 1 if index % 4 in (0, 1) else 0,
                "score": 0.0,
                "rank_percentile": 0.0,
            }
        )
    return {
        "models": [
            {
                "name": "fake-model",
                "judged_pairs": judged,
                "stage3": {"out_of_fold_scores": {"judged_pairs": list(scores)}},
            }
        ]
    }


def test_compare_stage3_arms_reports_a_paired_difference(tmp_path: Path) -> None:
    """The tool pairs two arms' per-pair scores and reports the difference."""
    module = _load("compare_stage3_arms")
    # two documents, two positives and two negatives each; the arm's scores
    # are the reference's with the ranking flipped inside every document
    reference_scores = [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1]
    flipped = [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9]
    (tmp_path / "ref").mkdir()
    (tmp_path / "arm").mkdir()
    (tmp_path / "ref" / "analysis.json").write_text(
        json.dumps(_artifact(reference_scores)), encoding="utf-8"
    )
    (tmp_path / "arm" / "analysis.json").write_text(
        json.dumps(_artifact(flipped)), encoding="utf-8"
    )
    assert module.main.__module__  # the tool is importable
    result = module.compare(
        module.model_scores(_artifact(reference_scores), "fake-model"),
        module.model_scores(_artifact(flipped), "fake-model"),
        resamples=50,
        seed=1,
    )
    # the flipped ranking is much worse inside every document
    assert result["auc_reference"] > 0.9
    assert result["auc_arm"] < 0.1
    assert result["delta_auc"] < -0.8
    assert result["ci_low"] < 0.0


def test_compare_stage3_arms_reports_a_null_comparison(tmp_path: Path) -> None:
    """The same arm against itself gives exactly zero."""
    module = _load("compare_stage3_arms")
    scores = [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1]
    result = module.compare(
        module.model_scores(_artifact(scores), "fake-model"),
        module.model_scores(_artifact(scores), "fake-model"),
        resamples=20,
        seed=1,
    )
    assert result["delta_auc"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_compare_stage3_arms_needs_the_out_of_fold_scores() -> None:
    """A run from before the Stage-3.3 change set is refused, not guessed at."""
    module = _load("compare_stage3_arms")
    artifact = _artifact([0.1, 0.2])
    artifact["models"][0]["stage3"] = {}
    with pytest.raises(KeyError, match="out_of_fold_scores"):
        module.model_scores(artifact, "fake-model")


def _upr_artifacts(tmp_path: Path, report: dict[str, Any]) -> Path:
    """Write a matching pair of artifacts (the evaluate run + the analyze run)."""
    results = tmp_path / "results"
    (results / "upr-eval-ref").mkdir(parents=True)
    (results / "upr-analyze-x").mkdir(parents=True)
    (results / "upr-eval-ref" / "metrics.json").write_text(
        json.dumps({"report": report, "n_records": 2, "n_labels": 3}), encoding="utf-8"
    )
    (results / "upr-analyze-x" / "analysis.json").write_text(
        json.dumps(
            {
                "guard": {"reference_run": "upr-eval-ref", "matches": True},
                "arms": {
                    "full_text": {
                        "metrics": {
                            "overall": report["overall"],
                            "per_bucket": report["per_bucket"],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return results


def test_verify_upr_baseline_accepts_a_matching_reference(tmp_path: Path) -> None:
    """The guard tool: a reference arm that reproduces the evaluate run passes."""
    module = _load("verify_upr_baseline")
    report = {"overall": {"n_records": 2, "n_labels": 3}, "per_bucket": {"tail": {"n": 1}}}
    results = _upr_artifacts(tmp_path, report)
    assert module.check_run(results, "upr-analyze-x") == []


def test_verify_upr_baseline_reports_a_mismatch(tmp_path: Path) -> None:
    """The guard tool: a differing report is a problem, not a pass."""
    module = _load("verify_upr_baseline")
    report = {"overall": {"n_records": 2, "n_labels": 3}, "per_bucket": {"tail": {"n": 1}}}
    results = _upr_artifacts(tmp_path, report)
    artifact_path = results / "upr-analyze-x" / "analysis.json"
    artifact = json.loads(artifact_path.read_text("utf-8"))
    artifact["arms"]["full_text"]["metrics"]["overall"]["n_records"] = 99
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    problems = module.check_run(results, "upr-analyze-x")
    assert problems and "the overall report differs" in problems[0]


def test_verify_upr_baseline_reports_an_unchecked_run(tmp_path: Path) -> None:
    """A run without a guard block is reported as unchecked, never as passing."""
    module = _load("verify_upr_baseline")
    results = tmp_path / "results"
    (results / "upr-analyze-x").mkdir(parents=True)
    (results / "upr-analyze-x" / "analysis.json").write_text(
        json.dumps({"arms": {}}), encoding="utf-8"
    )
    problems = module.check_run(results, "upr-analyze-x")
    assert problems and "no guard block" in problems[0]


def _contrastive_setup(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny corpus map and a split for the contrastive measurement tool.

    Returns:
        The config path and the results folder.
    """
    data = tmp_path / "data.jsonl"
    records = [
        {"id": "r1", "text": "alpha beta", "language": "en", "labels": ["Alpha"]},
        {"id": "r2", "text": "alpha gamma", "language": "en", "labels": ["Alpha"]},
        {
            "id": "r3",
            "text": "alpha delta",
            "language": "en",
            "labels": ["Alpha", "Beta"],
        },
    ]
    data.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train_ids.json").write_text(
        json.dumps({"split": "train", "ids": ["r1", "r2"]}), encoding="utf-8"
    )
    (splits / "val_ids.json").write_text(
        json.dumps({"split": "val", "ids": ["r3"]}), encoding="utf-8"
    )
    results = tmp_path / "results"
    (results / "upr-hitmap-corpus").mkdir(parents=True)

    def word(name: str, log_odds: float, z_score: float, name_word: bool) -> dict[str, Any]:
        return {
            "word": name,
            "n_with": 2,
            "df": 3,
            "lift": 1.0,
            "log_odds": log_odds,
            "z_score": z_score,
            "word_class": "distinctive",
            "is_name_word": name_word,
        }

    (results / "upr-hitmap-corpus" / "hitmap.json").write_text(
        json.dumps(
            {
                "run_id": "upr-hitmap-corpus",
                "corpus_map": {
                    "corpus": {"split": "train", "n_texts": 2, "n_labels": 3},
                    "per_label": [
                        {
                            "label": "Alpha",
                            "n_corpus_texts": 2,
                            "n_words": 2,
                            "words": [
                                word("alpha", 3.0, 2.0, True),
                                word("gamma", 1.0, 1.0, False),
                            ],
                        },
                        {
                            "label": "Beta",
                            "n_corpus_texts": 1,
                            "n_words": 2,
                            "words": [
                                word("alpha", -1.0, 0.5, True),
                                word("beta", 2.0, 1.5, True),
                            ],
                        },
                        {"label": "Gamma", "n_corpus_texts": 0, "n_words": 0, "words": []},
                    ],
                    "per_word": [
                        {
                            "word": "alpha",
                            "df": 3,
                            "word_class": "distinctive",
                            "labels": [
                                {"label": "Alpha", "n_with": 2, "lift": 1.0, "log_odds": 3.0},
                                {"label": "Beta", "n_with": 2, "lift": 1.0, "log_odds": -1.0},
                            ],
                        },
                        {
                            "word": "beta",
                            "df": 1,
                            "word_class": "distinctive",
                            "labels": [
                                {"label": "Beta", "n_with": 1, "lift": 1.0, "log_odds": 2.0},
                            ],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "run_id: upr-contrastive-check\n"
        "paths:\n"
        f"  data_raw: {data.as_posix()}\n"
        f"  data_processed: {(tmp_path / 'processed').as_posix()}\n"
        f"  data_splits: {splits.as_posix()}\n"
        f"  cache: {(tmp_path / 'cache').as_posix()}\n"
        f"  results: {results.as_posix()}\n",
        encoding="utf-8",
    )
    return config, results


def test_contrastive_map_check_measures_the_novelty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tool reads the map's artifact and measures the block's novelty."""
    module = _load("contrastive_map_check")
    config, results = _contrastive_setup(tmp_path)
    assert module.main(["--config", str(config), "--results", str(results)]) == 0
    out = capsys.readouterr().out
    assert "the registered bars" in out
    payload = json.loads(
        (results / "upr-hitmap-contrastive" / "contrastive.json").read_text("utf-8")
    )
    pooled = payload["summary"]["pooled"]
    # one val text, one alternative pair (Gamma has no table, so it never wins)
    assert pooled["n_pairs"] == 1
    # the text's word "alpha" is in both tables with opposite signs
    assert pooled["share_with_a_sign_disagreeing_text_row"] == pytest.approx(1.0)
    assert pooled["share_with_a_text_row"] == pytest.approx(1.0)
    # the corpus's words "gamma" and "beta" are not on the screen for the pair
    # (the focus label's table shows "gamma", but never against the other one)
    assert pooled["mean_novelty_share"] == pytest.approx(2 / 3)
    # the map's own artifact is the source, and its hash is recorded
    assert payload["source"]["corpus_map_sha256"]


def test_contrastive_map_check_needs_the_map_artifact(tmp_path: Path) -> None:
    """A missing map's artifact is an error, not an empty measurement."""
    module = _load("contrastive_map_check")
    config, results = _contrastive_setup(tmp_path)
    (results / "upr-hitmap-corpus" / "hitmap.json").unlink()
    assert module.main(["--config", str(config), "--results", str(results)]) == 1
