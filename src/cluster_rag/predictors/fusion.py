"""Fusion predictor (Phase 4): weighted mix of the retrieval signals.

PLAN Phase 4 "Fusion": ``score = a*knn + b*proto + c*zeroshot``, tuned on val.
This class implements the a- and b-terms over one embedding artifact (the same
context/geometry the component predictors use — knn votes from
:func:`cluster_rag.retrieve.knn.knn_vote_scores`, prototype cosines from
:func:`cluster_rag.retrieve.prototypes.prototype_scores`/``label_prototypes``);
the c-term joins once the zero-shot label-embedding artifact mechanism is
reviewer-approved (a second artifact kind feeds it; until then ``c`` is not an
accepted parameter, so a config that sets it fails loudly instead of silently
ignoring zero-shot).

Scale/ratio note: the components live on different native scales (kNN vote sums
in ~[0, k], prototype cosines in [-1, 1]). With a rank-based decision rule
(``top_k``) only the *ratio* a:b matters for any fixed (k, power) — scaling
both weights by a constant rescales the mixed sum and cannot change a ranking.
The weight sweep therefore fixes b = 1 and explores a as the ratio, with the
pure corners (a=1,b=0 -> knn; a=0,b=1 -> prototype) reproducing the component
predictors exactly — that equivalence is unit-tested.

Pure module: no model clients, no I/O; both components come from the same
train-only document matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params, check_vectors
from cluster_rag.retrieve.knn import knn_vote_scores
from cluster_rag.retrieve.prototypes import label_prototypes, prototype_scores


class FusionPredictor:
    """Scores labels by ``a * knn_votes(k, power) + b * prototype_cosines``."""

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("document", "query")
    ALLOWED_PARAMS: frozenset[str] = frozenset({"a", "b", "k", "power"})
    NAME: str = "fusion"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        for key in ("a", "b", "k"):
            if key not in params:
                raise ValueError(
                    f"predictor {self.NAME!r} requires params['{key}']: "
                    "'a' = kNN-vote weight, 'b' = prototype-cosine weight, "
                    "'k' = the kNN component's neighbor count (b = 1 in the sweep; "
                    "a explores the ratio)"
                )
        self._a = self._weight(params["a"], "a")
        self._b = self._weight(params["b"], "b")
        if self._a + self._b <= 0:
            raise ValueError("fusion weights must have a + b > 0 (at least one signal)")
        k = params["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"params['k'] must be a positive integer, got {k!r}")
        self._k = k
        power = params.get("power", 1.0)
        if not isinstance(power, (int, float)) or isinstance(power, bool):
            raise ValueError(f"params['power'] must be a number, got {power!r}")
        self._power = float(power)

        self._labels = context.labels
        self._label_matrix = build_label_matrix(context.train_records, context.labels)
        self._train_vectors = context.require_train_vectors()
        self._prototypes = label_prototypes(self._train_vectors, self._label_matrix)
        self._n_train = len(context.train_records)

    @staticmethod
    def _weight(value: Any, key: str) -> float:
        """Validate one non-negative finite weight."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"params['{key}'] must be a number, got {value!r}")
        weight = float(value)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"params['{key}'] must be a non-negative finite number, got {value!r}")
        return weight

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Mixed scores: ``(n_texts, n_labels)`` float32.

        ``query_vectors`` must carry one query-kind embedding per ``texts`` row
        (aligned by the evaluate command from ``eval.embedding_run_id``).
        """
        if query_vectors is None:
            raise ValueError(
                f"predictor {self.NAME!r} score() requires query_vectors: the "
                "query-kind embeddings of texts, aligned row-by-row (wired by the "
                "evaluate command from eval.embedding_run_id)"
            )
        query = check_vectors(query_vectors, name="query_vectors", n_rows=len(texts))
        knn_scores = knn_vote_scores(
            self._train_vectors, self._label_matrix, query, k=self._k, power=self._power
        )
        proto_scores = prototype_scores(self._prototypes, query)
        fused = self._a * knn_scores + self._b * proto_scores
        return fused.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        """Name, the fused formula's parameters, and the index facts."""
        return {
            "name": self.NAME,
            "a": self._a,
            "b": self._b,
            "k": self._k,
            "power": self._power,
            "n_train_examples": self._n_train,
            "dimension": int(self._train_vectors.shape[1]),
            "n_labels": len(self._labels),
        }
