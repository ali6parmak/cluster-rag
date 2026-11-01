"""The corpus map as a **scorer** (CS-30): the four pre-registered scoring functions.

The corpus map (CS-23) is a **description of the data**: for every label, the
words that go with it in the train corpus and how strongly they do. This
module turns that description into a **scorer**: a number for a (text, label)
cell. It is the lexical route's own answer, with no embedding and no model
call in it.

**The four scoring functions** (registered in advance in
``TEXT_PROCESSING_PLAN.md`` section 7.4, all four measured because all four are
cheap, and the choice between them made on the fit side):

- ``sum_logodds`` — the sum of the map's log-odds over the text's words (a
  word missing from the label's table contributes 0);
- ``sum_positive`` — the same sum, over the words whose log-odds is
  **positive** (the "evidence **for**" score);
- ``max_logodds`` — the **single best** word: the largest log-odds among the
  text's words that are in the label's table, and 0.0 when the text has no
  word in the table ("no evidence");
- ``support_count`` — how many of the label's table words the text contains,
  unweighted (the **measure-free control**: the same words, no measure).

**CS-31 adds two z-scored arms**: ``sum_zscore`` and ``sum_zscore_positive``
— the same two sums, over the map's **z-score** (the informative log-odds over
its standard error) instead of the log-odds. They are the re-measurement of
the classifier with the estimator's better measure as the weight.

**The unit is the map's own**: a **distinct** content word, tokenized exactly
as the map was built (``content_tokens`` — the lower-cased alphanumeric tokens
without the stopwords). A word repeated in the text counts once, because the
map's unit is the word's *presence*.

**The empty tables.** A label with no train example has an **empty** table
(CS-23 measures it: 8 of the 68 labels), so there is no evidence at all about
it: such a label is ranked **last in every row** and can never enter a top-k.
The floor is **finite and per row** — the smallest score among the labels that
*do* have a table, minus one — because the harness rejects non-finite scores
(``eval/multilabel.py``: "scores must be finite (NaN/inf would corrupt every
ranking)") and because the two signed arms are unbounded below, so only a
per-row floor is strictly below every seen label's score on every text. The
floor touches the empty tables alone: the seen labels' scores and their order
are untouched.

**The cross-fit.** :func:`cross_fit_scores` is the fit side's honest reading
of the map: the map is built from the train split, so scoring the train split
with it is optimistically biased. Each fold's texts are therefore scored by a
map built from the **other** folds — the track's own guard (CS-9b's fit-side
threshold, CS-32's cross-fit), applied here to choose between the arms.

Pure core: numpy + the standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cluster_rag.text_processors.features.corpus_map import CorpusMap, build_corpus_map
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens

# The four scoring functions of CS-30, plus the two **z-scored** arms of
# CS-31 (the map's weights read as the z-score instead of the log-odds). The
# order is also the tie-break order of the fit-side selection (the first
# maximal wins).
SCORING_FUNCTIONS: tuple[str, ...] = (
    "sum_logodds",
    "sum_positive",
    "max_logodds",
    "support_count",
    "sum_zscore",
    "sum_zscore_positive",
)

# What each arm does with the map's weights: the aggregation ("sum", "max" or
# "count"), which measure it reads ("log_odds" or "z_score") and whether it
# keeps only the words whose value is **positive** (the "evidence for" score).
_ARMS: dict[str, tuple[str, str, bool]] = {
    "sum_logodds": ("sum", "log_odds", False),
    "sum_positive": ("sum", "log_odds", True),
    "max_logodds": ("max", "log_odds", False),
    "support_count": ("count", "", False),
    "sum_zscore": ("sum", "z_score", False),
    "sum_zscore_positive": ("sum", "z_score", True),
}

# The gap between the empty tables' floor and the row's smallest seen score.
# Any strictly positive gap works; one unit is enough, because the floor only
# has to sit below the seen labels, never between them.
FLOOR_GAP = 1.0


def _word_index(corpus_map: CorpusMap) -> dict[str, tuple[tuple[int, float, float], ...]]:
    """The map's reverse tables as ``word -> ((column, log_odds, z_score), ...)``.

    Args:
        corpus_map: The corpus map (the train split's word↔label association).

    Returns:
        The word index: every word the corpus holds, with the label columns it
        belongs to, the log-odds of each (label, word) pair and the z-score
        (CS-31's measure).
    """
    columns = {label: index for index, label in enumerate(corpus_map.labels)}
    index: dict[str, tuple[tuple[int, float, float], ...]] = {}
    for word, rows in corpus_map.per_word.items():
        index[word] = tuple((columns[pair.label], pair.log_odds, pair.z_score) for pair in rows)
    return index


def score_cells(
    texts: Sequence[str],
    corpus_map: CorpusMap,
    *,
    scoring: str = "sum_logodds",
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> np.ndarray:
    """Score every text against every label from the map's own weights.

    Args:
        texts: The texts being scored.
        corpus_map: The corpus map (built from the train split only).
        scoring: Which of the four scoring functions to apply (see
            :data:`SCORING_FUNCTIONS`).
        stopwords: The tokenizer's stopwords — the map's own, so the words
            match the tables.

    Returns:
        A ``(n_texts, n_labels)`` float64 score matrix, higher = more likely.
        The labels with no train example carry the row's floor (the smallest
        seen score minus one), so they are ranked last in every row.

    Raises:
        ValueError: On an unknown scoring function.
    """
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring function {scoring!r}; expected one of {SCORING_FUNCTIONS}"
        )
    aggregation, measure, positive_only = _ARMS[scoring]
    labels = list(corpus_map.labels)
    index = _word_index(corpus_map)
    scores = np.zeros((len(texts), len(labels)), dtype=np.float64)
    empty_columns = [
        column for column, label in enumerate(labels) if not corpus_map.label_counts.get(label)
    ]
    take_max = aggregation == "max"
    for row, text in enumerate(texts):
        matched = np.zeros(len(labels), dtype=bool) if take_max else None
        for token in set(content_tokens(text, stopwords)):
            entries = index.get(token)
            if not entries:
                continue
            for column, log_odds, z_score in entries:
                value = log_odds if measure == "log_odds" else z_score
                if aggregation == "count":
                    scores[row, column] += 1.0
                elif take_max:
                    if not matched[column] or value > scores[row, column]:
                        scores[row, column] = value
                        matched[column] = True
                elif not positive_only or value > 0:
                    scores[row, column] += value
    if empty_columns:
        # the empty tables: no evidence at all, so they are ranked last in
        # their row (the harness's scores must be finite, so the floor is the
        # row's smallest seen score minus one).
        seen = np.delete(scores, empty_columns, axis=1)
        floor = (seen.min(axis=1) - FLOOR_GAP) if seen.shape[1] else np.full(len(texts), -FLOOR_GAP)
        scores[:, empty_columns] = floor[:, None]
    return scores


def cross_fit_scores(
    texts: Sequence[str],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
    *,
    scorings: Sequence[str] = SCORING_FUNCTIONS,
    folds: int = 5,
    seed: int = 42,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, np.ndarray]:
    """The **out-of-fold** score matrices: each fold's texts scored by the other folds.

    The map is built from the corpus it scores, so an in-fold score is
    optimistically biased — and most of all for the arms that use the log-odds
    the map itself computed. This is the fit side's honest reading: every text
    is scored by a map that never saw it.

    Args:
        texts: The corpus's texts (the train split, at the fit side).
        label_sets: Each text's labels, in the same order.
        labels: The label space (the map's columns).
        scorings: Which scoring functions to compute.
        folds: How many folds to split the corpus into (at least 2).
        seed: The fold split's seed (deterministic).
        stopwords: The tokenizer's stopwords.

    Returns:
        One ``(n_texts, n_labels)`` score matrix per scoring function, the rows
        in the corpus's own order.

    Raises:
        ValueError: On fewer than two folds, on an unknown scoring function,
            or when the texts and the label sets have different lengths.
    """
    if len(texts) != len(label_sets):
        raise ValueError(f"the corpus has {len(texts)} texts and {len(label_sets)} label sets")
    if folds < 2:
        raise ValueError(f"a cross-fit needs at least 2 folds, got {folds}")
    unknown = [name for name in scorings if name not in SCORING_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown scoring function(s) {unknown}; expected {SCORING_FUNCTIONS}")
    n_texts = len(texts)
    order = np.random.default_rng(seed).permutation(n_texts)
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(order, folds)]
    out = {name: np.zeros((n_texts, len(labels)), dtype=np.float64) for name in scorings}
    for chunk in chunks:
        held = set(chunk.tolist())
        keep = [index for index in range(n_texts) if index not in held]
        corpus_map = build_corpus_map(
            [texts[index] for index in keep],
            [label_sets[index] for index in keep],
            labels,
            stopwords=stopwords,
        )
        fold_texts = [texts[index] for index in chunk]
        for name in scorings:
            out[name][chunk] = score_cells(
                fold_texts, corpus_map, scoring=name, stopwords=stopwords
            )
    return out
