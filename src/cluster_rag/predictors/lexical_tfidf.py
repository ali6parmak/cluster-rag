"""The lexical floor (CS-29): TF-IDF + one-vs-rest logistic regression.

The measurement that says whether the pipeline's numbers are good. Every other
predictor in this package reads an **embedding**; this one reads the **words**,
and it is the reference the whole lexical methods round (CS-29 … CS-35) is read
against: a plain bag of words, a plain linear model, the same harness, the same
metric code, the same decision rules.

**The mechanism** (all of it fixed by the pre-registration in
``TEXT_PROCESSING_PLAN.md`` section 7.4):

- the **TF-IDF vectorizer** is fitted on the **train split only** (its
  vocabulary and its idf), word 1–2 grams, sublinear tf, scikit-learn's default
  token pattern and lowercasing, no stop-word removal, ``min_df=1``;
- **one logistic regression per label** (one-vs-rest), fitted on the train
  texts' matrix against that label's train indicator, with scikit-learn's
  defaults (``C=1.0``, ``solver="lbfgs"``, no class weighting) and
  ``max_iter=1000`` so the fit converges;
- the **score** is the positive-class **probability**, a bounded [0, 1] scale.

**The eight held-out labels.** A label with no train example cannot be fitted,
so it scores **0.0** for every text — and a fitted label's probability is
strictly above 0, so the eight always rank last and never enter a top-2. That
is what makes the unseen bucket **0.0000 by construction** (comparable with the
champion's own 0.0000; the decision-model run's 0.5697 is the one number that
is not comparable).

**The precedent.** This is the first module under ``predictors/`` to import
scikit-learn. Scikit-learn is not a model client — no model download, no
network, no I/O — and it is a core dependency of the main pipeline, so it is
allowed here (the track's pure package may not import it outside its
combiners, which is why this module is not under ``text_processors/``).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cluster_rag.data import build_label_matrix
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params

# The pre-registered defaults (TEXT_PROCESSING_PLAN.md section 7.4). The run
# uses exactly these; the knobs exist so a later change set can vary one thing
# at a time without editing code.
DEFAULT_PARAMS: dict[str, Any] = {
    "ngram_max": 2,
    "min_df": 1,
    "sublinear_tf": True,
    "lowercase": True,
    "stop_words": None,
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": None,
}

# The **TF-IDF knobs** the lexical predictors share: this floor (CS-29) and the
# lexical kNN (CS-41) read the same words with the same vectorizer, so the
# knobs live here once (the pre-registered values above).
TFIDF_PARAMS: dict[str, Any] = {
    "ngram_max": DEFAULT_PARAMS["ngram_max"],
    "min_df": DEFAULT_PARAMS["min_df"],
    "sublinear_tf": DEFAULT_PARAMS["sublinear_tf"],
    "lowercase": DEFAULT_PARAMS["lowercase"],
    "stop_words": DEFAULT_PARAMS["stop_words"],
}


def resolve_tfidf_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The effective TF-IDF knobs, validated (a bad value fails before the fit).

    Args:
        params: The configured knobs (the shared subset).

    Returns:
        The effective knobs: the pre-registered defaults, overridden by the
        configured ones.

    Raises:
        ValueError: On an unsupported type or an out-of-range value.
    """
    effective = dict(TFIDF_PARAMS)
    effective.update({key: value for key, value in params.items() if key in TFIDF_PARAMS})

    ngram_max = effective["ngram_max"]
    if not isinstance(ngram_max, int) or isinstance(ngram_max, bool) or ngram_max < 1:
        raise ValueError(
            f"the TF-IDF params['ngram_max'] must be a positive integer, got {ngram_max!r}"
        )
    min_df = effective["min_df"]
    if not isinstance(min_df, int) or isinstance(min_df, bool) or min_df < 1:
        raise ValueError(f"the TF-IDF params['min_df'] must be a positive integer, got {min_df!r}")
    for flag in ("sublinear_tf", "lowercase"):
        if not isinstance(effective[flag], bool):
            raise ValueError(
                f"the TF-IDF params[{flag!r}] must be a boolean, got {effective[flag]!r}"
            )
    if effective["stop_words"] not in (None, "english"):
        raise ValueError(
            f"the TF-IDF params['stop_words'] must be null or 'english', "
            f"got {effective['stop_words']!r}"
        )
    return effective


def build_tfidf_vectorizer(params: Mapping[str, Any]) -> TfidfVectorizer:
    """The TF-IDF vectorizer (the shared knobs), fitted by the caller.

    The rows are **l2-normalised** (``norm="l2"``), so the dot product of two
    rows is their cosine — the geometry the kNN machinery (CS-41) reads.

    Args:
        params: The effective TF-IDF knobs (from :func:`resolve_tfidf_params`).

    Returns:
        The vectorizer (unfitted).
    """
    return TfidfVectorizer(
        ngram_range=(1, params["ngram_max"]),
        min_df=params["min_df"],
        sublinear_tf=params["sublinear_tf"],
        lowercase=params["lowercase"],
        stop_words=params["stop_words"],
        norm="l2",
    )


def _resolve_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """The effective parameters, validated (a bad value fails before the fit).

    Args:
        params: The configured ``eval.predictor.params``.

    Returns:
        The effective parameters: the pre-registered defaults, overridden by
        the configured ones.

    Raises:
        ValueError: On an unsupported type or an out-of-range value.
    """
    effective = dict(DEFAULT_PARAMS)
    effective.update(params)
    effective.update(resolve_tfidf_params(params))

    c_value = effective["C"]
    if not isinstance(c_value, (int, float)) or isinstance(c_value, bool) or c_value <= 0:
        raise ValueError(
            f"predictor 'lexical_tfidf' params['C'] must be a positive number, got {c_value!r}"
        )
    max_iter = effective["max_iter"]
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError(
            f"predictor 'lexical_tfidf' params['max_iter'] must be a positive integer, "
            f"got {max_iter!r}"
        )
    if effective["class_weight"] not in (None, "balanced"):
        raise ValueError(
            "predictor 'lexical_tfidf' params['class_weight'] must be null or 'balanced', "
            f"got {effective['class_weight']!r}"
        )
    effective["C"] = float(c_value)
    return effective


class LexicalTfidfPredictor:
    """Scores labels by a TF-IDF + one-vs-rest logistic regression per label.

    Deterministic: scikit-learn's TF-IDF and its L-BFGS logistic regression
    have no stochastic part, so the same train split gives the same scores
    bit-for-bit. The seed is not used (the uniform constructor signature
    across predictors keeps it).
    """

    REQUIRES_EMBEDDINGS: bool = False
    REQUIRED_KINDS: tuple[str, ...] = ()
    ALLOWED_PARAMS: frozenset[str] = frozenset(DEFAULT_PARAMS)
    NAME: str = "lexical_tfidf"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,  # unused; uniform constructor signature across predictors
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        self._params = _resolve_params(params)
        self._labels = context.labels
        self._column_of = {label: index for index, label in enumerate(context.labels)}

        train_texts = [record.text for record in context.train_records]
        self._vectorizer = build_tfidf_vectorizer(self._params)
        matrix = self._vectorizer.fit_transform(train_texts)
        self._n_features = int(matrix.shape[1])

        label_matrix = build_label_matrix(context.train_records, context.labels)
        self._models: dict[str, LogisticRegression] = {}
        self._positive_columns: dict[str, int] = {}
        self._train_positives: dict[str, int] = {}
        self._n_convergence_warnings = 0
        self._n_degenerate = 0
        for column, label in enumerate(context.labels):
            positives = label_matrix[:, column]
            n_positive = int(positives.sum())
            self._train_positives[label] = n_positive
            if n_positive == 0:
                # No train example (the eight held-out labels): there is
                # nothing to fit, so the label keeps the 0.0 score and never
                # enters a top-k.
                continue
            if n_positive == len(positives):
                # Every train text carries the label: the class is not
                # identifiable (scikit-learn refuses to fit it), so the label
                # keeps the 0.0 score too and the run records the count.
                self._n_degenerate += 1
                continue
            model = LogisticRegression(
                C=self._params["C"],
                max_iter=self._params["max_iter"],
                solver="lbfgs",
                class_weight=self._params["class_weight"],
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(matrix, positives.astype(np.int64))
            self._n_convergence_warnings += sum(
                1 for item in caught if issubclass(item.category, ConvergenceWarning)
            )
            classes = np.asarray(model.classes_)
            self._positive_columns[label] = int(np.flatnonzero(classes == 1)[0])
            self._models[label] = model

        self._n_train = len(context.train_records)

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Per-label positive-class probability: ``(n_texts, n_labels)`` float64.

        ``query_vectors`` is ignored: this predictor is embedding-free.
        """
        rows = list(texts)
        matrix = self._vectorizer.transform(rows)
        scores = np.zeros((len(rows), len(self._labels)), dtype=np.float64)
        for label, model in self._models.items():
            probabilities = model.predict_proba(matrix)
            scores[:, self._column_of[label]] = probabilities[:, self._positive_columns[label]]
        return scores

    def describe(self) -> dict[str, Any]:
        """Name, the effective parameters and the index facts of the fit."""
        positives = [count for count in self._train_positives.values() if count]
        return {
            "name": self.NAME,
            "params": dict(self._params),
            "n_train_examples": self._n_train,
            "n_labels": len(self._labels),
            "n_labels_fitted": len(self._models),
            "n_labels_without_examples": sum(
                1 for count in self._train_positives.values() if count == 0
            ),
            "n_labels_degenerate": self._n_degenerate,
            "n_features": self._n_features,
            "train_positives": {
                "min": min(positives) if positives else 0,
                "max": max(positives) if positives else 0,
            },
            "n_convergence_warnings": self._n_convergence_warnings,
            "sklearn_version": sklearn.__version__,
            "vectorizer": (
                f"TfidfVectorizer(word 1-{self._params['ngram_max']} grams, "
                "sublinear tf, l2 rows), fitted on the train split only"
            ),
        }
