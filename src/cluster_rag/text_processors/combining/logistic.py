"""The logistic-regression combiner: the Stage-3.1 baseline.

The plan's Stage 3.1 puts a **logistic regression** behind the track's
``Combiner`` port: it fits a score over the feature pack's rows and scores
new rows. The gradient-boosted combiner (Stage 3.2) and the chunk-profile
features (Stage 3.3) come later; this file is deliberately the simplest
learned scorer, so the feature pack's contribution can be measured on its own.

The split handling stays with the caller (the driver): the combiner never
learns which document (or which split) a row came from. The caller hands it
the **fit side's** rows, and the ``sample_weight`` argument lets the caller
down-weight the fit side's unjudged control cells (presumed negatives whose
true positive rate is unmeasured).

**The scaler.** The features live on different scales (cosines in [-1, 1],
counts, one-hots), so the combiner standardizes them by default. The scaler
is fitted **inside** ``fit`` — on the fit side's rows only — and applied to
whatever is scored afterwards. That keeps the standardization out of the
holdout (a leakage rule of the plan's Stage 3).

Pure core in the AGENTS.md sense: no model client, no I/O, no global state.
The one heavy import is scikit-learn (already a core dependency of the main
pipeline; it downloads no model and touches no network).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# The config-facing name of this combiner (the registry key).
COMBINER_LOGISTIC = "logistic"


@dataclass(frozen=True)
class LogisticCombinerConfig:
    """The logistic regression's parameters.

    Attributes:
        C: The inverse regularization strength (sklearn's ``C``); smaller is
            more regularized.
        max_iter: The maximum number of solver iterations.
        standardize: Whether the features are standardized (the scaler is
            fitted on the fit side's rows only).
        class_weight: sklearn's ``class_weight`` (``None``, ``"balanced"`` or
            a mapping); ``None`` keeps the training weights as they are.
        solver: The sklearn solver.
        random_state: The seed of the solver's randomness.
    """

    C: float = 1.0
    max_iter: int = 1000
    standardize: bool = True
    class_weight: str | None = None
    solver: str = "lbfgs"
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject a degenerate parameter set.

        Raises:
            ValueError: If ``C`` is not positive, ``max_iter`` is not positive
                or ``class_weight`` is not one of sklearn's options.
        """
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.class_weight is not None and self.class_weight != "balanced":
            raise ValueError(f"class_weight must be None or 'balanced', got {self.class_weight!r}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (recorded in the artifact)."""
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "standardize": self.standardize,
            "class_weight": self.class_weight,
            "solver": self.solver,
            "random_state": self.random_state,
        }


class LogisticCombiner:
    """A logistic regression over the feature pack's rows.

    The combiner is a pure function of its rows: it holds the fitted model and
    the fitted scaler, and nothing else. ``score`` returns a **probability**
    (the model's ``predict_proba``), so the scale is the arm's own (the
    artifact says so: the level numbers of this arm are probabilities, the
    baseline arm's are cosines, and only the rank-based numbers are
    comparable between them).
    """

    name = COMBINER_LOGISTIC

    def __init__(self, config: LogisticCombinerConfig | None = None) -> None:
        self.config = config or LogisticCombinerConfig()
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._classes: np.ndarray = np.asarray([], dtype=np.int64)

    @property
    def is_fitted(self) -> bool:
        """Whether the combiner has been fitted."""
        return self._model is not None

    def fit(
        self,
        features: np.ndarray,
        labels: Sequence[int],
        sample_weight: Sequence[float] | None = None,
    ) -> LogisticCombiner:
        """Fit the regression on the fit side's rows.

        Args:
            features: The feature matrix of the fit side.
            labels: The 0/1 labels of the fit side's rows.
            sample_weight: An optional weight per row (the caller
                down-weights the unjudged control cells).

        Returns:
            The fitted combiner (``self``).

        Raises:
            ValueError: If the matrix and the labels disagree in length, a
                label is not 0/1, or the rows are empty.
        """
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if matrix.ndim != 2:
            raise ValueError(f"the feature matrix must be 2-D, got shape {matrix.shape}")
        if matrix.shape[0] != targets.size:
            raise ValueError(
                f"the feature matrix has {matrix.shape[0]} rows but {targets.size} labels"
            )
        if targets.size == 0:
            raise ValueError("the combiner cannot be fitted on zero rows")
        if not set(np.unique(targets).tolist()) <= {0, 1}:
            raise ValueError("the labels must be 0/1")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights is not None and weights.size != targets.size:
            raise ValueError("the sample weights must have one entry per row")

        if self.config.standardize:
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(matrix)
        else:
            self._scaler = None
            scaled = matrix
        model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            solver=self.config.solver,
            random_state=self.config.random_state,
        )
        model.fit(scaled, targets, sample_weight=weights)
        self._model = model
        self._classes = model.classes_
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """One score per row: the probability that the row is a positive.

        Args:
            features: The feature matrix to score.

        Returns:
            The float32 score vector (higher = more likely relevant).

        Raises:
            ValueError: If the combiner is not fitted.
        """
        if self._model is None:
            raise ValueError("the combiner is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"the feature matrix must be 2-D, got shape {matrix.shape}")
        if self._scaler is not None:
            matrix = self._scaler.transform(matrix)
        if self._classes.size < 2:
            # A degenerate fit (one class only): every row gets the same score.
            return np.zeros(matrix.shape[0], dtype=np.float32)
        probabilities = self._model.predict_proba(matrix)
        return probabilities[:, 1].astype(np.float32)

    def coefficients(self) -> dict[str, float]:
        """The fitted coefficients, keyed by the feature index.

        Raises:
            ValueError: If the combiner is not fitted.
        """
        if self._model is None:
            raise ValueError("the combiner is not fitted")
        values = self._model.coef_[0]
        return {str(index): float(value) for index, value in enumerate(values)}

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description of the combiner (recorded in the artifact)."""
        return {
            "name": self.name,
            "params": self.config.payload(),
            "fitted": self.is_fitted,
            "note": (
                "the combiner sees the fit side's rows only; the scaler (when "
                "enabled) is fitted inside fit(), so the holdout never "
                "contributes to the standardization"
            ),
        }
