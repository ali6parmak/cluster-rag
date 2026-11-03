"""The **union of the lexical and the embedding kNN** (the track's CS-46).

Two retrievers reach nearly the same **breadth** — the lexical kNN (CS-41) a
shortlist R@50 of **0.9437**, the embedding kNN **0.9576** — and differ in their
**failures**: the words miss what only the embedding geometry sees, the
embedding misses what only the words see. This predictor fuses the two
**rankings** with the track's own union rule — the per-row **z-max** of the two
score matrices (CS-14's rule) — and is a **measurement, not a component**: it
says how much breadth the two retrievers have **together**, which is the number
PLAN.md's deferred BM25 decision is waiting for.

**Nothing is re-implemented.** The lexical half is the existing
:class:`~cluster_rag.predictors.lexical_knn.LexicalKNNPredictor` (the same
TF-IDF, the same votes, the same machinery) and the embedding half the existing
:class:`~cluster_rag.predictors.knn.KNNVotingPredictor`, both constructed
**unmodified** (the same class, the same k, the same power, the same TF-IDF
knobs); the union is exactly
:func:`cluster_rag.text_processors.decision.upr_union.z_max_union` of their two
matrices.

**The label-blind counterpart.** With ``label_blind`` set, the train records'
label sets are permuted **once** (the track's yardstick) and the same permuted
records are handed to **both** halves, so the fusion's floor is measured on the
same control as the arms'.

Pure module: scikit-learn (no model download, no network, no I/O) and numpy —
the embedding half reads the train document vectors the evaluate command wires
from a prior ``upr-embed-*`` run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from cluster_rag.predictors.base import PredictorContext, check_no_extra_params
from cluster_rag.predictors.knn import KNNVotingPredictor
from cluster_rag.predictors.lexical_knn import LexicalKNNPredictor
from cluster_rag.predictors.lexical_tfidf import TFIDF_PARAMS
from cluster_rag.text_processors.decision.upr_union import z_max_union
from cluster_rag.text_processors.lexical.label_blind import permuted_label_sets


class KNNUnionPredictor:
    """Scores labels by the per-row z-max of the lexical and embedding kNNs.

    Params (``eval.predictor.params``):
        k: required, positive int — the number of neighbours each half votes
            with (CS-41's primary k = 10, the champion's own kNN k). Distinct
            from the decision rule's ``k`` (labels predicted).
        power: optional, positive float — the votes' exponent for both halves
            (default 1.0, the machinery's default).
        label_blind: the **label-blind counterpart** (default false): the train
            records' label sets are permuted once and handed to both halves,
            so the retrieval is untouched and every label↔neighbour association
            is destroyed. It is the fusion's empirical floor.
        the TF-IDF knobs (``ngram_max``, ``min_df``, ``sublinear_tf``,
        ``lowercase``, ``stop_words``): CS-29's/CS-41's own, passed to the
        lexical half.

    Deterministic: both halves are (scikit-learn's TF-IDF has no stochastic
    part, the kNN machinery's tie-break is the stable sort), and the z-max is
    elementwise. The seed is used only by the label-blind permutation.
    """

    REQUIRES_EMBEDDINGS: bool = True
    REQUIRED_KINDS: tuple[str, ...] = ("document", "query")
    ALLOWED_PARAMS = frozenset({"k", "power", "label_blind", *TFIDF_PARAMS})
    NAME: str = "knn_union"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        if "k" not in params:
            raise ValueError(
                f"predictor {self.NAME!r} requires params['k'] (positive int): the "
                "number of nearest train neighbours each half votes with. Distinct "
                "from the decision rule's top_k k (labels predicted)."
            )
        k = params["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"params['k'] must be a positive integer, got {k!r}")
        power = params.get("power", 1.0)
        if not isinstance(power, (int, float)) or isinstance(power, bool):
            raise ValueError(f"params['power'] must be a number, got {power!r}")
        label_blind = params.get("label_blind", False)
        if not isinstance(label_blind, bool):
            raise ValueError(f"params['label_blind'] must be a boolean, got {label_blind!r}")
        self._k = k
        self._power = float(power)
        self._label_blind = label_blind
        self._labels = context.labels

        half_context = context
        if label_blind:
            # the permutation happens ONCE, so both halves read the same labels
            permuted = permuted_label_sets(
                [list(record.labels) for record in context.train_records], seed
            )
            half_context = replace(
                context,
                train_records=tuple(
                    record.model_copy(update={"labels": list(label_set)})
                    for record, label_set in zip(context.train_records, permuted, strict=True)
                ),
            )
        # The halves are the existing predictors, unmodified: the lexical half
        # takes the TF-IDF knobs (never its own label_blind -- the permutation,
        # when asked for, is already in the context), the embedding half the
        # same k and power.
        lex_params = {key: value for key, value in params.items() if key != "label_blind"}
        self._lexical = LexicalKNNPredictor(half_context, seed, lex_params)
        self._embedding = KNNVotingPredictor(half_context, seed, {"k": k, "power": float(power)})

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """The union's scores: ``(n_texts, n_labels)`` float32, the per-row z-max.

        ``query_vectors`` must carry one query-kind embedding per ``texts`` row
        (aligned by the evaluate command from ``eval.embedding_run_id``) for the
        embedding half; the lexical half ignores them.
        """
        lexical = self._lexical.score(texts)
        embedding = self._embedding.score(texts, query_vectors=query_vectors)
        return z_max_union(lexical, embedding).astype(np.float32, copy=False)

    def describe(self) -> dict[str, Any]:
        """Name, parameters, the union rule and both halves' lineage."""
        return {
            "name": self.NAME,
            "k": self._k,
            "power": self._power,
            "label_blind": self._label_blind,
            "n_labels": len(self._labels),
            "union_rule": (
                "cluster_rag.text_processors.decision.upr_union.z_max_union: the "
                "per-row z-max of the two score matrices (CS-14's rule); the z-scores "
                "are taken per row, so the two halves' scales do not matter"
            ),
            "halves": {
                "lexical": self._lexical.describe(),
                "embedding": self._embedding.describe(),
            },
        }
