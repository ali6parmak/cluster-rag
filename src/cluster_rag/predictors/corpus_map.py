"""The corpus map as a **predictor** (CS-30): the lexical route's own answer.

The corpus map (CS-23) is the train split's own word↔label association — a
description of the data. This predictor turns it into a scorer: every
(text, label) cell gets a number from the map's own weights, with no
embedding and no model call in the loop. It is the round's second lexical
instrument after the TF-IDF floor (CS-29), and the two are read against each
other: CS-29 learns the weights from the text, this one reads them off the
corpus.

**The fit side.** The map is built from the **train** split only, so no val
text, no val label and no test text enters it. The choice between the four
scoring functions is made on the **fit side** and never on val — and the fit
side's number is the **out-of-fold** one (the cross-fit): the map has seen the
train cells it scores, so the in-fold number is optimistically biased, and most
of all for the arms that use the log-odds the map itself computed (a hapax
word's log-odds is huge: it contributes its full size to a sum and only 1 to a
count). A selection on the in-fold number would prejudge the very ablation the
change set exists to run. Both numbers are reported, so the bias is a
measurement and not an assertion.

The scorer is pure and lives in the track's package
(:mod:`cluster_rag.text_processors.lexical.corpus_map_score`); this module is
the adapter: the construction wiring, the contract checks, the fit side's
metric and the lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.data import build_label_matrix
from cluster_rag.eval.multilabel import apply_decision_rule, decision_f1_metrics
from cluster_rag.predictors.base import PredictorContext, check_no_extra_params
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import (
    FLOOR_GAP,
    SCORING_FUNCTIONS,
    cross_fit_scores,
    score_cells,
)
from cluster_rag.text_processors.lexical.pos_class import POS_CLASSES
from cluster_rag.text_processors.lexical.word_class import WORD_CLASSES


def _micro_f1_at(scores: np.ndarray, gold: np.ndarray, k: int) -> float:
    """The micro-F1 of a score matrix at the ``top_k`` operating point.

    The harness's own metric code (no second implementation of the metric).

    Args:
        scores: The score matrix.
        gold: The bool gold matrix of the same shape.
        k: The operating point's k (the labels predicted per row).

    Returns:
        The micro-F1.
    """
    predicted = apply_decision_rule(scores, rule="top_k", k=k)
    return float(decision_f1_metrics(predicted, gold)["micro_f1"])


class CorpusMapPredictor:
    """Scores labels by the corpus map's own word↔label weights.

    Params (``eval.predictor.params``):
        scoring: which of the four scoring functions to score with (default
            ``sum_logodds``; see
            :data:`~cluster_rag.text_processors.lexical.corpus_map_score.SCORING_FUNCTIONS`).
        fit_side_k: the k of the fit-side selection metric (default 2 — the
            operating point this change set is measured at).
        folds: the cross-fit's number of folds (default 5).
        restrict: the word classes to **keep** (CS-33; default every class) —
            the map with the generic class dropped, or the distinctive-only map.
        restrict_pos: the grammatical classes to **keep** (CS-34; default every
            class) — the map with the nouns only, or the verbs only.

    Deterministic: the map, the fold split and the scoring are all
    deterministic given the train split and the seed.
    """

    REQUIRES_EMBEDDINGS: bool = False
    REQUIRED_KINDS: tuple[str, ...] = ()
    ALLOWED_PARAMS: frozenset[str] = frozenset(
        {"scoring", "fit_side_k", "folds", "restrict", "restrict_pos"}
    )
    NAME: str = "corpus_map"

    def __init__(
        self,
        context: PredictorContext,
        seed: int,
        params: Mapping[str, Any],
    ) -> None:
        check_no_extra_params(params, self.ALLOWED_PARAMS, self.NAME)
        scoring = params.get("scoring", "sum_logodds")
        if scoring not in SCORING_FUNCTIONS:
            raise ValueError(
                f"predictor {self.NAME!r} params['scoring'] must be one of "
                f"{list(SCORING_FUNCTIONS)}, got {scoring!r}"
            )
        fit_side_k = params.get("fit_side_k", 2)
        if not isinstance(fit_side_k, int) or isinstance(fit_side_k, bool) or fit_side_k < 1:
            raise ValueError(
                f"predictor {self.NAME!r} params['fit_side_k'] must be a positive integer, "
                f"got {fit_side_k!r}"
            )
        folds = params.get("folds", 5)
        if not isinstance(folds, int) or isinstance(folds, bool) or folds < 2:
            raise ValueError(
                f"predictor {self.NAME!r} params['folds'] must be an integer >= 2, got {folds!r}"
            )
        restrict = params.get("restrict")
        if restrict is not None:
            if isinstance(restrict, str) or not isinstance(restrict, (list, tuple)):
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict'] must be a list of word "
                    f"classes, got {restrict!r}"
                )
            unknown = [name for name in restrict if name not in WORD_CLASSES]
            if unknown:
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict'] has unknown word class(es) "
                    f"{unknown}; expected some of {list(WORD_CLASSES)}"
                )
            if not restrict:
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict'] must keep at least one word class"
                )
        restrict_pos = params.get("restrict_pos")
        if restrict_pos is not None:
            if isinstance(restrict_pos, str) or not isinstance(restrict_pos, (list, tuple)):
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict_pos'] must be a list of "
                    f"grammatical classes, got {restrict_pos!r}"
                )
            unknown = [name for name in restrict_pos if name not in POS_CLASSES]
            if unknown:
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict_pos'] has unknown grammatical "
                    f"class(es) {unknown}; expected some of {list(POS_CLASSES)}"
                )
            if not restrict_pos:
                raise ValueError(
                    f"predictor {self.NAME!r} params['restrict_pos'] must keep at least one "
                    "grammatical class"
                )
        if restrict_pos is not None and not context.word_pos:
            raise ValueError(
                f"predictor {self.NAME!r} params['restrict_pos'] needs the corpus's "
                "grammatical classes, but the context carries none: enable the "
                "part-of-speech tagger (the pos: config section) so the train "
                "corpus is tagged"
            )
        self._scoring = scoring
        self._fit_side_k = fit_side_k
        self._folds = folds
        self._restrict = tuple(restrict) if restrict is not None else None
        self._restrict_pos = tuple(restrict_pos) if restrict_pos is not None else None
        self._labels = context.labels
        self._seed = seed

        train_texts = [record.text for record in context.train_records]
        train_label_sets = [record.labels for record in context.train_records]
        self._map = build_corpus_map(
            train_texts,
            train_label_sets,
            context.labels,
            word_pos=context.word_pos or None,
        )
        self._fit_side = self._fit_side_block(context, train_texts, train_label_sets)

    def _fit_side_block(
        self,
        context: PredictorContext,
        train_texts: Sequence[str],
        train_label_sets: Sequence[Sequence[str]],
    ) -> dict[str, Any]:
        """The fit side's numbers: the in-fold and the out-of-fold micro-F1, per arm.

        The **selection** is the out-of-fold maximum (the honest fit side);
        the in-fold number is reported beside it, so the bias the cross-fit
        removes is visible rather than asserted.
        """
        gold = build_label_matrix(context.train_records, context.labels)
        in_fold: dict[str, float] = {}
        for arm in SCORING_FUNCTIONS:
            scores = score_cells(
                train_texts,
                self._map,
                scoring=arm,
                restrict=self._restrict,
                restrict_pos=self._restrict_pos,
            )
            in_fold[arm] = _micro_f1_at(scores, gold, self._fit_side_k)
        out_of_fold: dict[str, float] = {}
        crossed = cross_fit_scores(
            train_texts,
            train_label_sets,
            list(context.labels),
            folds=self._folds,
            seed=self._seed,
            restrict=self._restrict,
            restrict_pos=self._restrict_pos,
            word_pos=context.word_pos or None,
        )
        for arm, scores in crossed.items():
            out_of_fold[arm] = _micro_f1_at(scores, gold, self._fit_side_k)
        selected = max(SCORING_FUNCTIONS, key=lambda arm: out_of_fold[arm])
        return {
            "side": "the train split (the fit side); the val split is never fitted",
            "metric": f"micro-F1 at top_k k={self._fit_side_k}",
            "cross_fit_folds": self._folds,
            "restrict": list(self._restrict) if self._restrict is not None else None,
            "restrict_pos": list(self._restrict_pos) if self._restrict_pos is not None else None,
            "in_fold_micro_f1": {arm: round(value, 6) for arm, value in in_fold.items()},
            "out_of_fold_micro_f1": {arm: round(value, 6) for arm, value in out_of_fold.items()},
            "selected": selected,
            "selected_by": "out_of_fold_micro_f1",
            "note": (
                "the map is built from the train split, so the in-fold number is "
                "optimistically biased (most of all for the arms that use the "
                "log-odds the map itself computed); the selection is made on the "
                "out-of-fold number, the fit side's honest reading. A restricted "
                "run (CS-33, CS-34) reads the same restricted map on the fit side, "
                "so the numbers here belong to the map the val split is scored with."
            ),
        }

    def score(self, texts: Sequence[str], *, query_vectors: np.ndarray | None = None) -> np.ndarray:
        """Per-label map scores: ``(n_texts, n_labels)`` float64.

        ``query_vectors`` is ignored: this predictor is embedding-free.
        """
        return score_cells(
            texts,
            self._map,
            scoring=self._scoring,
            restrict=self._restrict,
            restrict_pos=self._restrict_pos,
        )

    def describe(self) -> dict[str, Any]:
        """Name, the parameters, the map's shape and the fit side's numbers."""
        classes = self._map.word_class_index()
        return {
            "name": self.NAME,
            "scoring": self._scoring,
            "fit_side_k": self._fit_side_k,
            "restrict": list(self._restrict) if self._restrict is not None else None,
            "restrict_pos": list(self._restrict_pos) if self._restrict_pos is not None else None,
            "map": {
                "split": "train",
                "n_texts": self._map.n_texts,
                "n_labels": len(self._map.labels),
                "n_words": self._map.n_words,
                "n_pairs": self._map.n_pairs,
                "labels_with_examples": sum(
                    1 for label in self._map.labels if self._map.label_counts.get(label)
                ),
                "labels_without_examples": list(self._map.labels_without_examples()),
            },
            "word_classes": {
                "n_labelled_tables": classes.n_labelled_tables,
                "generic_threshold": classes.generic_threshold,
                "words_by_class": classes.counts_by_class(),
                "note": (
                    "the map's own word classes (CS-33): a word's class is how "
                    "many of the labelled tables it sits in. The restriction "
                    "keeps only the words of the named classes."
                ),
            },
            "fit_side": self._fit_side,
            "floor": {
                "rule": (
                    "a label with no train example has an empty table, so it has no "
                    "evidence at all and is ranked last in its row"
                ),
                "gap_below_the_row's_smallest_seen_score": FLOOR_GAP,
            },
        }
