"""kNN voting predictor (Phase 4): retrieval-augmented label scores.

The first non-trivial predictor port (PLAN Phase 4 checkbox "kNN voting"):
each query's label score is the similarity-weighted vote of its top-k nearest
**train** examples — the scoring math lives in
:func:`cluster_rag.retrieve.knn.knn_vote_scores`, this class adapts it to the
``Predictor`` port (constructor wiring, contract assertions, lineage).

Embeddings come from a prior ``upr-embed-*`` run (``eval.embedding_run_id``),
never encoded here: pure module, no model clients, no I/O. The train side uses
document-kind vectors (asymmetric models have a dedicated document prefix), the
eval side query-kind vectors — both aligned to records by id by the evaluate
command, which this class re-checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params, check_vectors
from cluster_rag.retrieve.knn import knn_vote_scores


class KNNVotingPredictor:
    """Scores labels by similarity-weighted votes of the k nearest train examples.

    Params (``eval.predictor.params``):
        k: required, positive int — the number of neighbors. Distinct from the
            decision rule's ``k`` (labels predicted); a sweep cell that uses
            both sets them as ``params={"k": 5}`` and the decision ``k=2``.
        power: optional, positive float — vote-weight exponent sim**power after
            clipping negative similarities to 0 (default 1.0).

    Deterministic: no seeding involved; ties break to the earlier train row
    (stable sort), so identical inputs reproduce identical scores.
    """

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("document", "query")
    ALLOWED_PARAMS: frozenset[str] = frozenset({"k", "power"})
    NAME: str = "knn"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        if "k" not in params:
            raise ValueError(
                f"predictor {self.NAME!r} requires params['k'] (positive int): the "
                "number of nearest train neighbors that vote. Distinct from the "
                "decision rule's top_k k (labels predicted)."
            )
        k = params["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"params['k'] must be a positive integer, got {k!r}")
        power = params.get("power", 1.0)
        if not isinstance(power, (int, float)) or isinstance(power, bool):
            raise ValueError(f"params['power'] must be a number, got {power!r}")
        self._k = k
        self._power = float(power)
        self._labels = context.labels
        self._label_matrix = build_label_matrix(context.train_records, context.labels)
        # Boundary check (reject, don't repair): float32 unit rows, one per record.
        self._train_vectors = context.require_train_vectors()

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Weighted top-k votes: ``(n_texts, n_labels)`` float32 scores.

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
        return knn_vote_scores(
            self._train_vectors, self._label_matrix, query, k=self._k, power=self._power
        )

    def describe(self) -> dict[str, Any]:
        """Name, parameters, and the index facts this predictor was built from."""
        return {
            "name": self.NAME,
            "k": self._k,
            "power": self._power,
            "n_train_examples": len(self._train_vectors),
            "dimension": int(self._train_vectors.shape[1]),
            "n_labels": len(self._labels),
        }
