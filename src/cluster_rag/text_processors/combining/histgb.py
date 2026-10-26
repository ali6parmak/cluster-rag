"""The gradient-boosted combiner: the Stage-3.2 learner.

The plan's Stage 3.2 puts a **gradient-boosted** scorer behind the track's
``Combiner`` port, next to the logistic regression of Stage 3.1
(:mod:`cluster_rag.text_processors.combining.logistic`). It is scikit-learn's
``HistGradientBoostingClassifier`` — a histogram-based gradient booster that
learns a *tree ensemble* instead of a linear score, so it can express
interactions between the feature pack's blocks (e.g. "the lexical overlap
matters, but only where the embedding cosine is low").

**Why the learner matters here at all.** The Stage-3.1b ablation measured the
feature pack's blocks one at a time and found that the learned arm's gain is
the **lexical** block's (a bag-of-words overlap), not the embedding geometry's
and not the question-type prior's. So a better learner has to earn something
over a *three-feature* logistic regression: a tree ensemble can only help if
the signal is not linear in the features, or if the blocks interact. The
comparison that matters is therefore HistGB on the **lexical** block against
HistGB on the **whole pack** — and against the logistic regression on the same
block (the track's Stage-3.1 / 3.1b runs, on the same data, control cells and
seeds).

**No standardization.** The logistic combiner standardizes its features (the
scaler is fitted inside ``fit()``, on the fit side's rows only). A tree
ensemble does not need it: the booster bins the features itself, so the scale
of a column cannot change the model's behaviour. There is therefore no scaler
here, and nothing that could leak the holdout into the fit.

**Early stopping is pinned off.** scikit-learn's default is ``"auto"``, which
switches early stopping on for large samples and carves an internal
validation split out of the training rows. The Stage-3 protocol wants the fit
side's rows to be the fit side's rows (and the same rows on every re-run), so
early stopping is explicitly disabled and the fit is deterministic.

**The feature usage (the importances capability).** A tree ensemble has no
coefficients: there is no single number per feature that says "how much the
model leans on it". What the trees *do* have is the record of their splits —
which feature each split node used, how much gain that split bought and how
deep it sat. :meth:`HistGbCombiner.feature_usage` reads that record and
returns it per feature, so the artifact can say **which features the trees
used**. It is a *usage* diagnostic (what the model splits on), not an
accuracy-based importance (what would improve the score if the feature were
removed) — the artifact says so, and the difference matters when the two
disagree.

Pure core in the AGENTS.md sense: no model client, no I/O, no global state.
The one heavy import is scikit-learn (already a core dependency of the main
pipeline; it downloads no model and touches no network).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# The config-facing name of this combiner (the registry key).
COMBINER_HISTGB = "histgb"


@dataclass(frozen=True)
class HistGbCombinerConfig:
    """The gradient booster's parameters.

    The defaults are scikit-learn's own, pinned explicitly so the artifact
    records the effective values and a re-run cannot drift with a library
    upgrade. They are **not** tuned on this data: the track has already
    measured that a fit-side selection did not transfer (the Stage-2 semantic
    threshold: the fit side's choice did not predict the holdout's), so a
    fit-side grid would buy a number that does not generalise.

    Attributes:
        max_iter: The number of boosting iterations (the trees' count).
        learning_rate: The shrinkage applied to each tree's contribution.
        max_leaf_nodes: The maximum number of leaves per tree.
        min_samples_leaf: The minimum number of rows in a leaf.
        l2_regularization: The L2 penalty on the leaves' values.
        max_depth: The maximum depth of a tree (``None`` = no limit).
        max_bins: The number of bins per feature.
        early_stopping: Whether the booster carves an internal validation
            split out of the training rows. Pinned off: the fit side's rows
            are the fit side's rows, and the fit stays deterministic.
        random_state: The seed of the booster's randomness.
    """

    max_iter: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_depth: int | None = None
    max_bins: int = 255
    early_stopping: bool = False
    random_state: int = 42

    def __post_init__(self) -> None:
        """Reject a degenerate parameter set.

        Raises:
            ValueError: If a parameter is outside its usable range.
        """
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.max_leaf_nodes < 2:
            raise ValueError(f"max_leaf_nodes must be >= 2, got {self.max_leaf_nodes}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}")
        if self.l2_regularization < 0:
            raise ValueError(f"l2_regularization must be >= 0, got {self.l2_regularization}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be None or >= 1, got {self.max_depth}")
        if self.max_bins < 2:
            raise ValueError(f"max_bins must be >= 2, got {self.max_bins}")

    def payload(self) -> dict[str, Any]:
        """JSON-friendly view of the parameters (recorded in the artifact)."""
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "max_depth": self.max_depth,
            "max_bins": self.max_bins,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }


def _iter_predictors(model: HistGradientBoostingClassifier) -> list[Any]:
    """The fitted trees, whatever shape scikit-learn stores them in.

    The booster's private ``_predictors`` attribute holds one entry per
    boosting iteration, and each entry is a list of trees (one per class for a
    multiclass fit; one for this binary task). The traversal flattens both
    levels, so a change in scikit-learn's internal layout cannot silently
    empty the diagnostic.
    """
    predictors = getattr(model, "_predictors", None)
    if not predictors:
        return []
    trees: list[Any] = []
    for entry in predictors:
        if isinstance(entry, (list, tuple)):
            trees.extend(entry)
        else:
            trees.append(entry)
    return trees


class HistGbCombiner:
    """A gradient-boosted tree ensemble over the feature pack's rows.

    The combiner is a pure function of its rows: it holds the fitted ensemble
    and nothing else. ``score`` returns a **probability** (the ensemble's
    ``predict_proba``), so the scale is the arm's own (the artifact says so:
    the level numbers of this arm are probabilities, the baseline arm's are
    cosines, and only the rank-based numbers are comparable between them) —
    the same convention as the logistic combiner, so the two learners' arms
    are read on the same footing.
    """

    name = COMBINER_HISTGB

    def __init__(self, config: HistGbCombinerConfig | None = None) -> None:
        self.config = config or HistGbCombinerConfig()
        self._model: HistGradientBoostingClassifier | None = None
        self._degenerate = False
        self._n_features = 0

    @property
    def is_fitted(self) -> bool:
        """Whether the combiner has been fitted."""
        return self._model is not None or self._degenerate

    def fit(
        self,
        features: np.ndarray,
        labels: Sequence[int],
        sample_weight: Sequence[float] | None = None,
    ) -> HistGbCombiner:
        """Fit the ensemble on the fit side's rows.

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

        self._n_features = int(matrix.shape[1])
        if np.unique(targets).size < 2:
            # A degenerate fit (one class only): the booster cannot be fitted,
            # so every row gets the same score.
            self._model = None
            self._degenerate = True
            return self

        model = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            max_depth=self.config.max_depth,
            max_bins=self.config.max_bins,
            early_stopping=self.config.early_stopping,
            random_state=self.config.random_state,
        )
        model.fit(matrix, targets, sample_weight=weights)
        self._model = model
        self._degenerate = False
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
        if not self.is_fitted:
            raise ValueError("the combiner is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"the feature matrix must be 2-D, got shape {matrix.shape}")
        if self._degenerate or self._model is None:
            # A degenerate fit (one class only): every row gets the same score.
            return np.zeros(matrix.shape[0], dtype=np.float32)
        probabilities = self._model.predict_proba(matrix)
        return probabilities[:, 1].astype(np.float32)

    def feature_usage(self) -> dict[str, dict[str, float]]:
        """Which features the fitted trees split on, keyed by the feature index.

        Each entry carries the number of split nodes that used the feature,
        the total gain those splits bought, the deepest level the feature was
        used at, and the two shares (of all the splits, of all the gain). A
        feature the trees never split on gets zeros — that is information, not
        a missing value.

        This is a **usage** diagnostic, not an accuracy-based importance: it
        says what the model splits on, not what would improve the score if
        the feature were removed.

        Returns:
            The per-feature usage, keyed by the feature index as a string
            (the driver names the features).
        """
        usage: dict[str, dict[str, float]] = {
            str(index): {
                "n_splits": 0.0,
                "total_gain": 0.0,
                "max_depth": 0.0,
                "split_share": 0.0,
                "gain_share": 0.0,
            }
            for index in range(self._n_features)
        }
        trees = _iter_predictors(self._model) if self._model is not None else []
        splits = 0
        gain = 0.0
        for tree in trees:
            nodes = getattr(tree, "nodes", None)
            if nodes is None:
                continue
            internal = nodes[~nodes["is_leaf"].astype(bool)]
            for feature, split_gain, depth in zip(
                internal["feature_idx"], internal["gain"], internal["depth"], strict=True
            ):
                index = int(feature)
                if index < 0 or index >= self._n_features:
                    continue
                entry = usage[str(index)]
                entry["n_splits"] += 1.0
                entry["total_gain"] += float(split_gain)
                entry["max_depth"] = max(entry["max_depth"], float(depth))
                splits += 1
                gain += float(split_gain)
        for entry in usage.values():
            entry["split_share"] = entry["n_splits"] / splits if splits else 0.0
            entry["gain_share"] = entry["total_gain"] / gain if gain else 0.0
        return usage

    def describe(self) -> dict[str, Any]:
        """A JSON-friendly description of the combiner (recorded in the artifact)."""
        return {
            "name": self.name,
            "params": self.config.payload(),
            "fitted": self.is_fitted,
            "note": (
                "the combiner sees the fit side's rows only. A tree ensemble "
                "needs no standardization (the booster bins the features "
                "itself), so there is no scaler that could leak the holdout "
                "into the fit, and early stopping is off, so no internal "
                "validation split is carved out of the fit side's rows."
            ),
        }
