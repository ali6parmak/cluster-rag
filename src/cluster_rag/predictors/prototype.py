"""Label-prototype predictor (Phase 4): cosine to mean train-example embeddings.

The second retrieval baseline (PLAN Phase 4 checkbox "Label prototypes: mean of
the example embeddings per label"): a label scores by the cosine between the
query embedding and the label's renormalized mean of train example embeddings.
The math lives in :mod:`cluster_rag.retrieve.prototypes` (shared with the
upcoming per-label max-similarity variant and the fusion phase).

Same embedding wiring as the kNN predictor: document-kind vectors for the train
side, query-kind for the eval side, from a prior ``upr-embed-*`` run. Pure
module; labels with zero train examples get zero prototypes and always score 0
(zero-shot discovery is the zero-shot label-embedding predictor's job).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params, check_vectors
from cluster_rag.retrieve.prototypes import label_prototypes, prototype_scores


class PrototypePredictor:
    """Scores labels by cosine to their (renormalized) mean train embedding."""

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("document", "query")
    ALLOWED_PARAMS: frozenset[str] = frozenset()
    NAME: str = "prototype"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        train_vectors = context.require_train_vectors()
        self._labels = context.labels
        label_matrix = build_label_matrix(context.train_records, context.labels)
        self._prototypes = label_prototypes(train_vectors, label_matrix)
        self._n_train = len(context.train_records)
        self._n_labels_with_prototype = int((np.linalg.norm(self._prototypes, axis=1) > 0).sum())

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Cosine to prototypes: ``(n_texts, n_labels)`` float32 scores.

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
        return prototype_scores(self._prototypes, query)

    def describe(self) -> dict[str, Any]:
        """Name and the prototype table facts (train size, labels covered)."""
        return {
            "name": self.NAME,
            "n_train_examples": self._n_train,
            "n_labels": len(self._labels),
            "n_labels_with_prototype": self._n_labels_with_prototype,
        }
