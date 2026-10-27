"""Tests for the UPR sibling-max backoff (the text-processing track's CS-11).

Contracts under test:

- the **unsupported** labels: the labels with no fit-side support (zero gold
  cells on the fit side) — the backoff's targets, decided on the fit side;
- the **control** labels: the fit side's rarest *supported* labels;
- the **siblings**: a label's taxonomy siblings, restricted to the supported
  ones (a label without one is a measured null by construction);
- the **backoff** itself: the target's score is raised to the siblings' best
  (times the weight) when that is higher, and the other labels are untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.config import UprBackoffConfig, UprLearnedConfig
from cluster_rag.text_processors.features.upr_backoff import (
    control_labels,
    fit_side_counts,
    sibling_backoff,
    sibling_sets,
    unsupported_labels,
)

PARENTS = {
    "Detention": "Liberty",
    "Privacy": "Liberty",
    "Health": "Economic",
    "Food": "Economic",
    "General": "Catch-all",
}


def _fit_gold() -> np.ndarray:
    """A fit side where "General" has no support at all."""
    return np.asarray(
        [
            [True, False, True, False, False],
            [True, False, False, True, False],
            [False, False, True, False, False],
        ]
    )


def test_the_fit_side_counts_are_the_labels_support() -> None:
    """The fit side's gold-cell count per label."""
    counts = fit_side_counts(_fit_gold())
    assert list(counts) == [2, 0, 2, 1, 0]


def test_the_unsupported_labels_are_the_ones_without_support() -> None:
    """The backoff's targets: zero gold cells on the fit side."""
    labels = ("Detention", "Privacy", "Health", "Food", "General")
    assert unsupported_labels(_fit_gold(), labels) == ("Privacy", "General")


def test_the_control_labels_are_the_rarest_supported_ones() -> None:
    """The control's labels: the fit side's rarest supported labels."""
    labels = ("Detention", "Privacy", "Health", "Food", "General")
    assert control_labels(_fit_gold(), labels, n=2) == ("Food", "Detention")
    # a supported label is never the control's target
    assert "General" not in control_labels(_fit_gold(), labels, n=2)


def test_the_siblings_are_the_supported_ones_only() -> None:
    """A sibling without fit-side support cannot be inherited from."""
    labels = ("Detention", "Privacy", "Health", "Food", "General")
    siblings = sibling_sets(labels, PARENTS, supported=("Detention", "Health", "Food"))
    assert siblings["Detention"] == ()
    assert siblings["Privacy"] == ("Detention",)
    assert siblings["Health"] == ("Food",)
    assert siblings["General"] == ()


def test_the_backoff_raises_the_target_to_the_siblings_best() -> None:
    """The target's score becomes the siblings' best when that is higher."""
    labels = ("Detention", "Privacy", "Health", "Food", "General")
    scores = np.asarray(
        [
            [0.10, 0.20, 0.90, 0.30, 0.05],
            [0.80, 0.10, 0.20, 0.40, 0.05],
        ]
    )
    backoff, report = sibling_backoff(
        scores,
        labels,
        PARENTS,
        _fit_gold(),
        targets=("Privacy",),
        lam=1.0,
    )
    # "Privacy"'s only supported sibling is "Detention" (0.10 and 0.80)
    assert backoff[0, 1] == pytest.approx(0.20)  # the sibling is lower: unchanged
    assert backoff[1, 1] == pytest.approx(0.80)  # the sibling is higher: raised
    # the other labels are untouched
    assert backoff[0, 0] == pytest.approx(0.10)
    assert backoff[0, 4] == pytest.approx(0.05)
    assert report["per_target"]["Privacy"]["n_siblings"] == 1
    assert report["per_target"]["Privacy"]["share_changed"] == pytest.approx(0.5)


def test_the_backoff_never_lowers_a_score() -> None:
    """The backoff is a floor, not a replacement."""
    labels = ("Detention", "Privacy")
    scores = np.asarray([[0.90, 0.95]])
    backoff, _ = sibling_backoff(
        scores,
        labels,
        {"Detention": "Liberty", "Privacy": "Liberty"},
        np.asarray([[True, False]]),
        targets=("Privacy",),
        lam=1.0,
    )
    assert backoff[0, 1] == pytest.approx(0.95)


def test_the_backoff_reports_a_null_for_a_label_without_siblings() -> None:
    """A label with no supported sibling cannot be backed off."""
    labels = ("Detention", "Privacy")
    scores = np.asarray([[0.10, 0.20]])
    backoff, report = sibling_backoff(
        scores,
        labels,
        {"Detention": "Liberty", "Privacy": "Other"},
        np.asarray([[True, False]]),
        targets=("Privacy",),
        lam=1.0,
    )
    assert backoff[0, 1] == pytest.approx(0.20)
    assert report["per_target"]["Privacy"]["n_siblings"] == 0
    assert "null" in report["per_target"]["Privacy"]["note"]


def test_the_backoff_rejects_an_unknown_target() -> None:
    """A target outside the label space is a config error, not a silent no-op."""
    with pytest.raises(ValueError, match="not in the label space"):
        sibling_backoff(
            np.asarray([[0.1, 0.2]]),
            ("Detention", "Privacy"),
            PARENTS,
            np.asarray([[True, False]]),
            targets=("Nonexistent",),
            lam=1.0,
        )


def test_the_backoff_rejects_a_negative_weight() -> None:
    """A negative weight would lower a score, which the rule never does."""
    with pytest.raises(ValueError, match="weight"):
        sibling_backoff(
            np.asarray([[0.1, 0.2]]),
            ("Detention", "Privacy"),
            PARENTS,
            np.asarray([[True, False]]),
            targets=("Privacy",),
            lam=-1.0,
        )


def test_the_backoff_is_off_by_default() -> None:
    """The earlier runs' artifacts keep their exact content (the CS-8d precedent)."""
    assert UprLearnedConfig().backoff.enabled is False
    assert UprBackoffConfig().enabled is False
    assert UprBackoffConfig().lams == (0.5, 1.0)
    assert UprBackoffConfig().control is True


def test_the_backoff_config_rejects_a_negative_weight() -> None:
    """A negative weight is a config error."""
    with pytest.raises(ValueError, match="lams"):
        UprBackoffConfig(lams=(-0.5,))
