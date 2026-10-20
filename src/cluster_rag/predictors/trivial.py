"""Trivial baselines (Phase 3): score-only references for the evaluation harness.

Both ignore the text being classified (or sample scores), so they calibrate
what "reasonable" means for every real metric: the most-frequent baseline
predicts the globally popular labels regardless of the query; the random
baseline samples a seeded uniform score per label. Both are pure numpy, see
train artifacts only, and reproduce exactly under a fixed seed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.predictors.base import PredictorContext, check_no_extra_params


class MostFrequentPredictor:
    """Scores every label by its train-set frequency — identical for every text.

    The "suggest whatever is globally most common" baseline: its ranking is
    constant across records, so recall@k and F1 depend only on how often the
    popular labels happen to cover a record's gold set. Labels with zero train
    examples (the unseen holdout) always rank last — the baseline must score
    0 on the zero-shot discovery question by construction.
    """

    REQUIRES_EMBEDDINGS: bool = False
    REQUIRED_KINDS: tuple[str, ...] = ()
    ALLOWED_PARAMS: frozenset[str] = frozenset()
    NAME: str = "most_frequent"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        counts: Counter[str] = Counter()
        for record in context.train_records:
            counts.update(record.labels)
        self._labels = context.labels
        self._scores = np.asarray(
            [counts.get(label, 0) for label in context.labels], dtype=np.float64
        )
        self._n_train = len(context.train_records)

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Tile the constant per-label score row: ``(n_texts, n_labels)``.

        ``query_vectors`` is ignored: this predictor is embedding-free.
        """
        return np.tile(self._scores, (len(texts), 1))

    def describe(self) -> dict[str, Any]:
        """Name, train size, and how many labels it has ever seen."""
        return {
            "name": self.NAME,
            "train_records": self._n_train,
            "n_known_labels": int((self._scores > 0).sum()),
        }


class RandomPredictor:
    """Scores labels with a seeded uniform draw — the random-reference baseline.

    Reproducible for a fixed seed: one row of iid uniforms per text; ties are
    measure-zero for continuous draws, and the stable-sort convention would
    rank them alphabetically if one ever occurred.
    """

    REQUIRES_EMBEDDINGS: bool = False
    REQUIRED_KINDS: tuple[str, ...] = ()
    ALLOWED_PARAMS: frozenset[str] = frozenset()
    NAME: str = "random"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        self._labels = context.labels
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._n_labels = len(context.labels)

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """One uniform draw per (text, label) cell: ``(n_texts, n_labels)``.

        ``query_vectors`` is ignored: this predictor is embedding-free.
        """
        return self._rng.uniform(size=(len(texts), self._n_labels))

    def describe(self) -> dict[str, Any]:
        """Name and the seed that reproduces this predictor bit-for-bit."""
        return {"name": self.NAME, "seed": self._seed}
