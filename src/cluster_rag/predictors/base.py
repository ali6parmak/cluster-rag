"""Predictor protocol (Phase 3): per-label score matrices for the harness contract.

A predictor maps texts to one score per label in the fixed label space. The
label space (the sorted taxonomy columns) and the train-only context are fixed
at construction, so `score` is a pure function of the input texts — trivial
baselines see train counts only, no val/test gold (leakage boundary).

Two views share one matrix: `score(...) >= threshold` rows (threshold view,
``eval.multilabel.apply_decision_rule``) and the descending argsort (top-k
view). Phase 4 retrieval predictors (kNN / prototype / zero-shot / fusion)
implement this same port. Pure module: no model clients, no I/O (AGENTS.md
dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cluster_rag.data import Record


class Predictor(Protocol):
    """The port every label scorer implements (baselines now, Phase 4 next)."""

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Score every text against every label.

        Returns:
            A ``(n_texts, n_labels)`` float matrix, higher = more likely,
            arbitrary scale (thresholds are relative to the score semantics).
            Column order is the construction-time label space.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-serializable lineage (name, params, versions) for run artifacts."""
        ...


@dataclass(frozen=True)
class PredictorContext:
    """Construction inputs shared by all predictors.

    Attributes:
        labels: The score-column convention: every label the predictor may
            output, including labels with zero train examples (they score
            according to the predictor's fall-back, e.g. 0 for frequencies).
        train_records: The train split only — the leakage boundary. No val/
            test record ever reaches a predictor at construction time.
    """

    labels: tuple[str, ...]
    train_records: tuple[Record, ...] = field(default=())


def check_no_extra_params(params: Mapping[str, Any], allowed: Mapping[str, Any], name: str) -> None:
    """Reject parameter keys a predictor does not support (fail before the run)."""
    unknown = [key for key in params if key not in allowed]
    if unknown:
        raise ValueError(
            f"predictor {name!r} does not support parameter(s) {unknown}; "
            f"supported: {sorted(allowed)}"
        )
