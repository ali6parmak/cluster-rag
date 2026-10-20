"""Zero-shot label predictor (Phase 4): cosine between texts and label texts.

Scores every dataset label by the cosine between the text's **query-kind**
embedding and the label's label-kind embedding from a prior ``upr-labelembed-*``
run (``eval.label_embedding_run_id``): the label text is config-driven (bare
name or a template such as ``"This recommendation is about {label}."``, see
:class:`~cluster_rag.config.LabelTextsConfig`) so no train examples are ever
needed — the only route to labels absent from train (the holdout/unseen
buckets), and explicitly train-legal for those (AGENTS.md: zero-shot
name/description embeddings may cover held-out labels).

The artifact loader guarantees the label vectors match the harness label space
exactly (order included) and share the primary embedding run's model and
dimension, so column ``i`` is always ``labels[i]``. Pure module: no model
clients, no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.predictors.base import PredictorContext, check_no_extra_params, check_vectors


class ZeroShotPredictor:
    """Scores labels by cosine against their name/template label vectors."""

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("query",)  # no train-document matrix involved
    REQUIRES_LABEL_EMBEDDING: bool = True
    ALLOWED_PARAMS: frozenset[str] = frozenset()
    NAME: str = "zeroshot"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        self._labels = context.labels
        # Boundary check (reject, don't repair): float32 unit rows, one per label.
        self._label_vectors = context.require_label_vectors()

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Query↔label cosines: ``(n_texts, n_labels)`` float32 scores.

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
        return (query @ self._label_vectors.T).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        """Name and the label-vector facts this predictor scores against."""
        return {
            "name": self.NAME,
            "n_labels": len(self._labels),
            "dimension": int(self._label_vectors.shape[1]),
        }
