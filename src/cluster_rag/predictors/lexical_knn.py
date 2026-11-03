"""The **lexical kNN** (the track's CS-41): the words alone as a retriever.

The lexical floor (CS-29) reads the **words** as a **classifier**: a TF-IDF
matrix and one logistic regression per label, and it is the round's reference
*floor* (micro-F1 0.3845). CS-41 reads the same words as a **retriever**: the
text's TF-IDF **nearest neighbours** over the **train** split, their labels
voted with the main pipeline's **own** kNN machinery
(:func:`cluster_rag.retrieve.knn.knn_vote_scores`) — the same votes, the same
clipping at zero, the same power, the same top-k and the same tie-break as the
embedding kNN reads.

**The geometry is the same.** The TF-IDF's rows are l2-normalised (CS-29's
vectorizer's own ``norm="l2"``), so the dot product the kNN machinery computes
**is** the cosine: the embedding kNN's own similarity, computed on the words
instead of on the embedding.

**What is measured.** The candidate generator's **ceiling**: the harness's own
shortlist recall — the share of gold cells whose label sits anywhere in the
row's top-50 — which is the candidate set a reranker would have to work with.
CS-29 gave the lexical **classifier**'s floor; this gives the lexical
**retrieval**'s ceiling, and the two together say how much of the pipeline's
retrieval the words alone can carry.

**The label-blind counterpart.** With ``label_blind`` set, the train records'
label sets are permuted among the records (the track's own control, CS-32's
and CS-36's): the retrieval is untouched and every label↔neighbour
association is destroyed. Whatever the blind arm buys, it bought on noise.

Pure module: scikit-learn (no model download, no network, no I/O) and numpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import sklearn

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params
from cluster_rag.predictors.lexical_tfidf import (
    TFIDF_PARAMS,
    build_tfidf_vectorizer,
    resolve_tfidf_params,
)
from cluster_rag.retrieve.knn import knn_vote_scores
from cluster_rag.text_processors.lexical.label_blind import permuted_label_sets


class SparseMatrix(Protocol):
    """The sparse matrix's own interface (scipy's, without importing scipy).

    scikit-learn returns a scipy sparse matrix; scipy itself is scikit-learn's
    dependency, not a declared one of this package, so the two methods this
    module needs are declared as a **port** instead of importing it.
    """

    def astype(self, dtype: Any) -> SparseMatrix:
        """A copy of the matrix with the given dtype."""
        ...

    def toarray(self) -> np.ndarray:
        """The matrix as a dense array."""
        ...


def _dense_rows(matrix: SparseMatrix) -> np.ndarray:
    """A sparse TF-IDF matrix as a dense float32 matrix (the kNN core's shape).

    The kNN machinery is dense (it multiplies two matrices), so the sparse
    matrix is materialised. The **numbers do not change**: the zeros stay
    zeros, and the l2-normalised rows keep their norm.

    Args:
        matrix: The sparse TF-IDF matrix.

    Returns:
        The dense float32 matrix.
    """
    return np.ascontiguousarray(np.asarray(matrix.astype(np.float32).toarray()))


class LexicalKNNPredictor:
    """Scores labels by the votes of the TF-IDF nearest train neighbours.

    Params (``eval.predictor.params``):
        k: required, positive int — the number of neighbours that vote.
            Distinct from the decision rule's ``k`` (labels predicted), exactly
            as in the embedding kNN.
        power: optional, positive float — the vote-weight exponent
            (default 1.0, the machinery's default and the embedding kNN's own).
        label_blind: the **label-blind counterpart** (default false): the train
            records' label sets are permuted among the records, so the
            retrieval is untouched and every label↔neighbour association is
            destroyed.
        the TF-IDF knobs (``ngram_max``, ``min_df``, ``sublinear_tf``,
        ``lowercase``, ``stop_words``): CS-29's floor's own, shared with it —
        the vectorizer is fitted on the **train** split only.

    Deterministic: scikit-learn's TF-IDF has no stochastic part and the kNN
    machinery's tie-break is the stable sort, so the same train split gives the
    same scores bit-for-bit. The seed is used only by the label-blind
    permutation.
    """

    REQUIRES_EMBEDDINGS = False
    REQUIRED_KINDS: tuple[str, ...] = ()
    ALLOWED_PARAMS = frozenset({"k", "power", "label_blind", *TFIDF_PARAMS})
    NAME: str = "lexical_knn"

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
                "number of nearest train neighbours that vote. Distinct from the "
                "decision rule's top_k k (labels predicted)."
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
        self._tfidf = resolve_tfidf_params(params)
        self._labels = context.labels

        train_texts = [record.text for record in context.train_records]
        self._vectorizer = build_tfidf_vectorizer(self._tfidf)
        matrix = self._vectorizer.fit_transform(train_texts)
        self._n_features = int(matrix.shape[1])
        self._train_vectors = _dense_rows(matrix)
        # a train text with no known feature (a row of zeros): it has no
        # neighbour with a positive similarity, so it never votes
        self._n_empty_rows = int((~self._train_vectors.any(axis=1)).sum())

        train_records = context.train_records
        if label_blind:
            permuted = permuted_label_sets([list(record.labels) for record in train_records], seed)
            train_records = tuple(
                record.model_copy(update={"labels": list(label_set)})
                for record, label_set in zip(train_records, permuted, strict=True)
            )
        self._label_matrix = build_label_matrix(train_records, context.labels)

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """The neighbours' votes: ``(n_texts, n_labels)`` float32 scores.

        ``query_vectors`` is ignored: this predictor is embedding-free — the
        neighbours are the text's TF-IDF nearest train texts.
        """
        matrix = self._vectorizer.transform(list(texts))
        query = _dense_rows(matrix)
        return knn_vote_scores(
            self._train_vectors,
            self._label_matrix,
            query,
            k=self._k,
            power=self._power,
        )

    def describe(self) -> dict[str, Any]:
        """Name, parameters and the index facts this predictor was built from."""
        return {
            "name": self.NAME,
            "k": self._k,
            "power": self._power,
            "label_blind": self._label_blind,
            "n_train_examples": int(self._train_vectors.shape[0]),
            "n_features": self._n_features,
            "n_labels": len(self._labels),
            "n_train_texts_without_a_known_feature": self._n_empty_rows,
            "tfidf_params": dict(self._tfidf),
            "sklearn_version": sklearn.__version__,
            "vectorizer": (
                f"TfidfVectorizer(word 1-{self._tfidf['ngram_max']} grams, "
                "sublinear tf, l2 rows), fitted on the train split only"
            ),
            "votes": (
                "cluster_rag.retrieve.knn.knn_vote_scores: the main pipeline's own "
                "kNN machinery (the same votes, the same clipping, the same power, "
                "the same top-k and the same tie-break as the embedding kNN)"
            ),
        }
