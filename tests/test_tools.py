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
