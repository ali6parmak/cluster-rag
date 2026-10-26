"""Tests for the UPR decision probe (the text-processing track's CS-9).

Contracts under test:

- the **cells**: one cell per (text, label), the label-blind counterpart's
  deterministic shift, and the cell's own gold flag;
- the **measurement**: the UPR harness's own metrics (the decision view, the
  ranking view, the gold side, the background side), the calibration and the
  **probability compression**;
- the **sample**: a seeded draw of the split's texts, the same texts for
  every arm (so the arms are paired);
- the config's contract: the scope, the sample size and the pinned models.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import DecisionModelConfig, UprAnalysisConfig, UprDecisionConfig
from cluster_rag.text_processors.decision.upr_probe import (
    build_upr_cells,
    measure_upr_arm,
    probability_compression,
    score_matrix,
    upr_question_text,
)
from cluster_rag.upr_analysis import build_upr_view
from cluster_rag.upr_decision import UprDecisionSample, sample_texts

DECISION = {"rule": "top_k", "k": 1}


def _sample() -> UprDecisionSample:
    """A two-text, two-label sample with one gold cell."""
    return UprDecisionSample(
        texts=("Ratify the convention on enforced disappearances", "Ensure housing for all"),
        labels=("Detention", "Poverty"),
        gold=np.asarray([[True, False], [False, True]]),
        rows=(0, 1),
    )


def test_the_question_text_is_the_label_name() -> None:
    """A UPR label is a taxonomy term: the question is the topic itself."""
    assert upr_question_text("Detention") == "Detention"


def test_the_cells_cover_the_whole_grid() -> None:
    """One cell per (text, label), each carrying its gold flag."""
    cells = build_upr_cells(_sample().texts, _sample().labels, _sample().gold)
    assert len(cells) == 4
    assert [cell.label for cell in cells] == [1, 0, 0, 1]
    assert cells[0].paragraph_text.startswith("Ratify")
    assert cells[0].question_text == "Detention"


def test_the_label_blind_counterpart_shifts_the_questions() -> None:
    """The counterpart keeps every cell but replaces the question."""
    sample = _sample()
    cells = build_upr_cells(sample.texts, sample.labels, sample.gold, mismatched=True)
    assert [cell.question_text for cell in cells] == [
        "Poverty",
        "Detention",
        "Poverty",
        "Detention",
    ]
    # the cells themselves (and their gold flags) are unchanged
    assert [cell.label for cell in cells] == [1, 0, 0, 1]


def test_the_score_matrix_follows_the_cells() -> None:
    """The scores line up with the gold matrix cell by cell."""
    sample = _sample()
    cells = build_upr_cells(sample.texts, sample.labels, sample.gold)
    matrix = score_matrix(cells, [0.9, 0.1, 0.2, 0.8])
    assert matrix.shape == sample.gold.shape
    assert matrix[0, 0] == pytest.approx(0.9)
    assert matrix[1, 1] == pytest.approx(0.8)


def test_the_measurement_reports_the_harness_metrics() -> None:
    """The decision view, the ranking view, the gold side and the background."""
    sample = _sample()
    cells = build_upr_cells(sample.texts, sample.labels, sample.gold)
    matrix = score_matrix(cells, [0.9, 0.1, 0.2, 0.8])
    measurement = measure_upr_arm(
        matrix,
        sample.gold,
        decision=DECISION,
        ks=(1,),
        shortlist_ms=(1,),
        bins=5,
    )
    assert measurement["decision"]["micro_f1"] == pytest.approx(1.0)
    assert measurement["gold"]["rank_p50"] == pytest.approx(0.0)
    assert measurement["gold"]["hit@1"] == pytest.approx(1.0)
    assert measurement["background"]["n_cells"] == 2
    assert measurement["calibration"]["n_bins"] == 5


def test_the_probability_compression_is_measured() -> None:
    """The compression: the gold cells' mean probability against the background's."""
    scores = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    gold = np.asarray([[True, False], [False, True]])
    compression = probability_compression(scores, gold)
    assert compression["gold_mean"] == pytest.approx(0.85)
    assert compression["background_mean"] == pytest.approx(0.15)
    assert compression["share_gold_above_0.5"] == pytest.approx(1.0)
    assert compression["share_background_above_0.5"] == pytest.approx(0.0)


def test_the_sample_is_seeded_and_paired() -> None:
    """The same seed gives the same texts (so the arms are paired)."""
    from cluster_rag.data import Record

    records = tuple(
        Record(id=str(index), text=f"text {index}", language="en", labels=("Detention",))
        for index in range(20)
    )
    view = build_upr_view(records, ("Detention",), {"Detention": "head"}, (10, 20))
    first = sample_texts(view, size=5, seed=42)
    second = sample_texts(view, size=5, seed=42)
    other = sample_texts(view, size=5, seed=7)
    assert first.rows == second.rows
    assert first.rows != other.rows
    assert first.n_texts == 5
    assert first.gold.shape == (5, 1)


def test_the_decision_config_is_off_by_default() -> None:
    """No model = no probe (the earlier runs' artifacts keep their content)."""
    decision = UprDecisionConfig()
    assert decision.models == ()
    assert decision.scope == "sample"
    assert UprAnalysisConfig().decision.models == ()


def test_the_decision_config_rejects_an_empty_sample() -> None:
    """A sample of zero texts is a config error, not a silent empty run."""
    with pytest.raises(ValueError, match="sample_size"):
        UprDecisionConfig(sample_size=0)


def test_the_decision_model_must_pin_its_digest() -> None:
    """The model is pinned by id **and** digest (the track's hygiene)."""
    model = DecisionModelConfig(
        name="tev1",
        model_id="tev1:latest",
        digest="7954dce22bc6c29c3b119656207681d12cb34434fed7caa086bfbc919bef0740",
        instructions="Is the text about the topic in the question?",
    )
    assert model.digest
    with pytest.raises(ValueError):
        DecisionModelConfig(
            name="tev1",
            model_id="tev1:latest",
            digest="",
            instructions="Is the text about the topic in the question?",
        )
