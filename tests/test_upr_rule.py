"""Tests for the UPR safe-negative rule and the shortlist rerank (CS-9b).

Contracts under test:

- the **margins**: every cell's distance below its own text's best label;
- the **rule**: the three forms (the fixed cut, the fit-side threshold, the
  margin rule), the fit side's own choices, the coverage / harm pair and the
  **label-blind counterpart** beside each form;
- the **shortlist rerank**: the champion's shortlist, the reranked top-1, the
  shortlist's recall as the ceiling and the two references beside it;
- the config's contract: the rule is **off** by default.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import UprDecisionConfig, UprDecisionRuleConfig
from cluster_rag.text_processors.decision.upr_rule import (
    RULE_FIT_THRESHOLD,
    RULE_FIXED_CUT,
    RULE_MARGIN,
    upr_margins,
    upr_rule_block,
    upr_shortlist_rerank_block,
)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """A 3-text, 2-label grid: the gold cells are the high ones."""
    scores = np.asarray(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.60, 0.40],
        ]
    )
    gold = np.asarray(
        [
            [True, False],
            [False, True],
            [True, False],
        ]
    )
    return scores, gold


def test_the_margin_is_the_distance_below_the_text_s_best() -> None:
    """Every cell's margin: how far below its own text's best label it sits."""
    margins = upr_margins(np.asarray([[0.9, 0.1], [0.2, 0.8]]))
    assert margins[0, 0] == pytest.approx(0.0)
    assert margins[0, 1] == pytest.approx(0.8)
    assert margins[1, 1] == pytest.approx(0.0)
    assert margins[1, 0] == pytest.approx(0.6)


def test_the_margins_reject_a_one_dimensional_matrix() -> None:
    """A flat score vector is not a grid."""
    with pytest.raises(ValueError, match="2-dimensional"):
        upr_margins(np.asarray([0.1, 0.2]))


def test_the_rule_reports_the_three_forms() -> None:
    """The block carries the fixed cut, the fit-side threshold and the margin rule."""
    scores, gold = _grid()
    block = upr_rule_block(scores, gold, fit_scores=scores, fit_gold=gold)
    assert set(block["forms"]) == {RULE_FIXED_CUT, RULE_FIT_THRESHOLD, RULE_MARGIN}
    assert block["space"]["n_gold_cells"] == 3
    assert block["space"]["n_background_cells"] == 3
    assert block["fit_side"]["n_texts"] == 3


def test_the_fixed_cut_clears_the_low_cells() -> None:
    """The fixed 0.5 cut: the cells below 0.5 are cleared, the gold ones are not."""
    scores, gold = _grid()
    block = upr_rule_block(scores, gold, fit_scores=scores, fit_gold=gold, fixed_cut=0.5)
    fixed = block["forms"][RULE_FIXED_CUT]["eval_side"]
    # every background cell is below 0.5 (0.10, 0.20, 0.40)
    assert fixed["n_unjudged_cleared"] == 3
    assert fixed["n_unjudged_cells"] == 3
    assert fixed["coverage"] == pytest.approx(1.0)
    # no gold cell is below 0.5
    assert fixed["n_positives_discarded"] == 0
    assert fixed["harm"] == pytest.approx(0.0)


def test_the_fit_side_never_leaks_into_the_evaluated_split() -> None:
    """The threshold is chosen on the fit side, and the artifact says so."""
    scores, gold = _grid()
    fit_scores = np.asarray([[0.95, 0.05], [0.05, 0.90], [0.55, 0.45]])
    block = upr_rule_block(scores, gold, fit_scores=fit_scores, fit_gold=gold)
    form = block["forms"][RULE_FIT_THRESHOLD]
    assert form["fit_side"]["criterion"] == "youden"
    assert "FIT side" in form["fit_side"]["note"]
    assert form["fit_side"]["n_candidates"] > 0


def test_the_margin_rule_is_chosen_on_the_fit_side() -> None:
    """The margin rule's margin comes from the fit side's own trade-off."""
    scores, gold = _grid()
    block = upr_rule_block(scores, gold, fit_scores=scores, fit_gold=gold)
    form = block["forms"][RULE_MARGIN]
    assert form["fit_side"]["criterion"] == "coverage - harm"
    assert form["fit_side"]["margin"] >= 0.0
    assert form["eval_side"]["n_unjudged_cells"] == 3


def test_the_label_blind_counterpart_sits_beside_every_form() -> None:
    """The counterpart is measured at the same operating point."""
    scores, gold = _grid()
    blind = np.asarray([[0.05, 0.90], [0.80, 0.20], [0.40, 0.60]])
    block = upr_rule_block(
        scores,
        gold,
        fit_scores=scores,
        fit_gold=gold,
        blind_scores=blind,
    )
    for form in block["forms"].values():
        counterpart = form["label_blind_counterpart"]
        assert counterpart["n_draws"] == 1
    threshold_form = block["forms"][RULE_FIT_THRESHOLD]["label_blind_counterpart"]
    assert "harm_at_matched_coverage" in threshold_form
    # the counterpart gets the SAME operating point (the CS-7b precedent)
    assert threshold_form["eval_side"] is not None


def test_the_rule_rejects_a_counterpart_with_another_shape() -> None:
    """The counterpart's scores must have the real ones' shape."""
    scores, gold = _grid()
    with pytest.raises(ValueError, match="must match"):
        upr_rule_block(scores, gold, fit_scores=scores, fit_gold=gold, blind_scores=scores[:2])


def test_the_rule_rejects_mismatched_shapes() -> None:
    """The scores and the gold must have the same shape."""
    scores, gold = _grid()
    with pytest.raises(ValueError, match="must match"):
        upr_rule_block(scores, gold[:2], fit_scores=scores, fit_gold=gold)


def test_the_shortlist_rerank_reorders_the_champions_shortlist() -> None:
    """The reranked top-1 is the shortlist member the decision model ranks first."""
    scores = np.asarray([[0.90, 0.10], [0.20, 0.80]])
    gold = np.asarray([[True, False], [False, True]])
    # the champion's order is the opposite of the decision model's
    reference = np.asarray([[0.10, 0.90], [0.80, 0.20]])
    block = upr_shortlist_rerank_block(scores, gold, reference_scores=reference, ks=(1, 2))
    per_k = block["per_k"]
    # with K=1 the shortlist is the champion's own first label: the rerank cannot
    # move it, so the reranked hit@1 is the champion's own
    assert per_k["1"]["champion_hit@1"] == pytest.approx(0.0)
    assert per_k["1"]["rerank_hit@1"] == pytest.approx(0.0)
    # the decision model's own top-1 is right for both texts
    assert per_k["1"]["decision_hit@1"] == pytest.approx(1.0)
    # with K=2 the shortlist holds both labels and the rerank finds the gold one
    assert per_k["2"]["rerank_hit@1"] == pytest.approx(1.0)
    assert per_k["2"]["shortlist_recall"] == pytest.approx(1.0)


def test_the_shortlist_recall_is_the_ceiling() -> None:
    """The shortlist's own recall bounds the reranked hit@1."""
    scores = np.asarray([[0.90, 0.10, 0.05]])
    gold = np.asarray([[True, False, False]])
    reference = np.asarray([[0.05, 0.10, 0.90]])
    block = upr_shortlist_rerank_block(scores, gold, reference_scores=reference, ks=(1, 3))
    per_k = block["per_k"]
    # the champion's top-1 is the third label, so the gold cell is not in the K=1
    assert per_k["1"]["shortlist_recall"] == pytest.approx(0.0)
    assert per_k["1"]["rerank_hit@1"] == pytest.approx(0.0)
    # with K=3 the shortlist holds it and the decision model ranks it first
    assert per_k["3"]["shortlist_recall"] == pytest.approx(1.0)
    assert per_k["3"]["rerank_hit@1"] == pytest.approx(1.0)
    assert per_k["3"]["rerank_hit@1"] <= per_k["3"]["shortlist_recall"]


def test_the_rerank_rejects_mismatched_shapes() -> None:
    """The reference scores must have the decision scores' shape."""
    scores, gold = _grid()
    with pytest.raises(ValueError, match="must match"):
        upr_shortlist_rerank_block(scores, gold, reference_scores=scores[:2], ks=(1,))


def test_the_rule_is_off_by_default() -> None:
    """The earlier runs' artifacts keep their exact content (the CS-9 precedent)."""
    assert UprDecisionConfig().rule.enabled is False
    assert UprDecisionRuleConfig().enabled is False
    assert UprDecisionRuleConfig().fixed_cut == pytest.approx(0.5)
    assert UprDecisionRuleConfig().rerank_ks == (5, 10, 20, 50)


def test_the_rule_config_rejects_a_non_positive_k() -> None:
    """A K of zero is a config error, not a silent empty rerank."""
    with pytest.raises(ValueError, match="rerank_ks"):
        UprDecisionRuleConfig(rerank_ks=(0,))
