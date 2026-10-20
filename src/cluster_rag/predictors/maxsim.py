"""Per-label max-similarity predictor (Phase 4): strongest single example match.

The third retrieval baseline (PLAN Phase 4 checkbox "Per-label max-similarity"):
a label scores by the **maximum** cosine between the query embedding and any
single train example carrying that label — over the same document-kind matrix
the prototype mean averages. Complements the mean for tail labels: a rare
label with one very close example stays visible where its average prototype
would wash out (and vice versa; the checkbox-C sweep measures which aggregation
carries which frequency bucket).

Same embedding wiring as knn/prototype (document-kind train vectors, query-kind
eval vectors, from a prior ``upr-embed-*`` run via ``eval.embedding_run_id``).
Labels with zero train examples score the cosine floor (-1.0) and can never
enter a top-k decision (zero-shot discovery is the zero-shot predictor's job).
Pure module: scoring math in
:func:`cluster_rag.retrieve.prototypes.max_sim_scores`, no model clients, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params, check_vectors
from cluster_rag.retrieve.prototypes import max_sim_scores


class MaxSimPredictor:
    """Scores labels by the best cosine against any single train example."""

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("document", "query")
    ALLOWED_PARAMS: frozenset[str] = frozenset()
    NAME: str = "maxsim"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        self._labels = context.labels
        self._label_matrix = build_label_matrix(context.train_records, context.labels)
        # Boundary check (reject, don't repair): float32 unit rows, one per record.
        self._train_vectors = context.require_train_vectors()
        self._n_train = len(context.train_records)

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Per-label max cosine: ``(n_texts, n_labels)`` float32 scores.

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
        return max_sim_scores(self._train_vectors, self._label_matrix, query)

    def describe(self) -> dict[str, Any]:
        """Name and the index facts this predictor was built from."""
        return {
            "name": self.NAME,
            "n_train_examples": self._n_train,
            "dimension": int(self._train_vectors.shape[1]),
            "n_labels": len(self._labels),
        }
