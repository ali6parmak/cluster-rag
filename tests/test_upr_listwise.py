"""Tests for the UPR listwise second stage (the text-processing track's CS-12).

Contracts under test:

- the **question spec**: one choice question over the candidates, with the
  options' criteria, and the question's wording in the instructions;
- the **call plan**: one call per text, each with its own options, and the
  input budget counting the options and the criteria (a 20-option call is
  much larger than the pointwise one);
- the **candidates**: the round-1 top-N columns per text;
- the **rank fusion**: the round-2 order over the candidates, then the
  round-1 order for the rest, with the score matrix as the negative rank;
- the **measurement**: the calibration and the compression are not applicable
  to an arm whose scores are ranks.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import UprDecisionConfig, UprListwiseConfig
from cluster_rag.text_processors.decision.upr_listwise import (
    build_listwise_spec,
    fuse_rankings,
    generic_criteria,
    plan_listwise_calls,
    top_candidates,
)
from cluster_rag.text_processors.decision.upr_probe import measure_upr_arm


def test_the_spec_carries_the_options_and_the_criteria() -> None:
    """The second round is one choice question over the candidates."""
    spec = build_listwise_spec(
        ("Detention", "Poverty"),
        {"Detention": "custody and arrest", "Poverty": "living conditions"},
    )
    assert spec.type == "choice"
    assert spec.options == ("Detention", "Poverty")
    assert spec.criteria["Poverty"] == "living conditions"
    assert "Which of these topics" in spec.instructions


def test_the_spec_rejects_mismatched_criteria() -> None:
    """The criteria must describe exactly the options."""
    with pytest.raises(ValueError, match="criteria"):
        build_listwise_spec(("Detention", "Poverty"), {"Poverty": "x"})


def test_the_generic_criterion_names_the_label() -> None:
    """The name-only control's criterion says what the label is."""
    assert "Detention" in generic_criteria("Detention")


def test_the_plan_makes_one_call_per_text() -> None:
    """Each text gets its own call, with its own options."""
    calls = plan_listwise_calls(
        ("text one", "text two"),
        [("A", "B"), ("C", "D")],
        [{"A": "a", "B": "b"}, {"C": "c", "D": "d"}],
    )
    assert len(calls) == 2
    assert calls[0].specs[0].options == ("A", "B")
    assert calls[1].specs[0].options == ("C", "D")
    assert calls[0].state["paragraph"] == "text one"


def test_the_plan_rejects_mismatched_lengths() -> None:
    """The texts, the options and the criteria must line up."""
    with pytest.raises(ValueError, match="same"):
        plan_listwise_calls(
            ("text one", "text two"),
            [("A", "B")],
            [{"A": "a", "B": "b"}],
        )


def test_the_plan_counts_the_options_and_the_criteria_in_the_budget() -> None:
    """A 20-option call with 20 descriptions must fit the model's budget."""
    options = tuple(f"label {index}" for index in range(20))
    criteria = dict.fromkeys(options, "a fairly long description of the option " * 3)
    calls = plan_listwise_calls(
        ("x" * 20000,),
        [options],
        [criteria],
        max_input_tokens=2500,
    )
    # the state is cut to fit: the question's own text is never cut
    assert len(calls[0].state["paragraph"]) < 20000
    assert len(calls[0].specs[0].instructions) > 0
    assert len(calls[0].specs[0].options) == 20


def test_the_candidates_are_the_round_one_top_n() -> None:
    """The candidates of every text: its round-1 top-N columns, best first."""
    scores = np.asarray([[0.1, 0.9, 0.5], [0.7, 0.2, 0.4]])
    assert top_candidates(scores, 2) == [(1, 2), (0, 2)]


def test_the_fusion_keeps_the_round_two_order_over_the_candidates() -> None:
    """The round-2 order wins inside the candidates; the round-1 order below."""
    round_one = np.asarray([[0.9, 0.8, 0.1, 0.2]])
    round_two = np.asarray([[np.nan, 0.2, 0.9, np.nan]])
    fused = fuse_rankings(round_one, round_two)
    # the candidates are the columns 1 and 2; the round-2 order puts 2 first
    assert fused[0, 2] > fused[0, 1]
    # the non-candidates keep the round-1 order below them
    assert fused[0, 0] > fused[0, 3]
    assert fused[0, 3] == pytest.approx(-3.0)
    assert fused[0, 2] == pytest.approx(0.0)


def test_the_fusion_rejects_mismatched_shapes() -> None:
    """The two rounds must share one shape."""
    with pytest.raises(ValueError, match="share one shape"):
        fuse_rankings(np.zeros((2, 3)), np.zeros((2, 4)))


def test_the_measurement_flags_the_calibration_as_not_applicable() -> None:
    """A rank-based arm has no calibration and no probability compression."""
    scores = np.asarray([[-0.0, -1.0, -2.0]])
    gold = np.asarray([[True, False, False]])
    measurement = measure_upr_arm(
        scores,
        gold,
        decision={"rule": "top_k", "k": 1},
        ks=(1,),
        shortlist_ms=(1,),
        probability_scores=False,
    )
    assert measurement["calibration"]["applicable"] is False
    assert measurement["compression"]["applicable"] is False
    assert measurement["decision"]["micro_f1"] == pytest.approx(1.0)


def test_the_listwise_is_off_by_default() -> None:
    """The earlier runs' artifacts keep their exact content (the CS-9 precedent)."""
    assert UprDecisionConfig().listwise.enabled is False
    assert UprListwiseConfig().enabled is False
    assert UprListwiseConfig().candidates == 20
    assert UprListwiseConfig().criteria == "catalog"


def test_the_listwise_config_rejects_too_many_candidates() -> None:
    """The API's choice question takes at most 26 options."""
    with pytest.raises(ValueError, match="candidates"):
        UprListwiseConfig(candidates=30)
