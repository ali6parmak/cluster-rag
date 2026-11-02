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

**CS-33 adds the class restriction**: a scoring call may keep only **some of
the word classes** (``restrict``) — the map with the **generic** class dropped
(the register) or the **distinctive-only** map. The restriction rides into the
cross-fit too, so the fit side reads the same map the evaluated split is
scored with.

**CS-36 adds the stability filter**: a scoring call may keep only the
(label, word) pairs that **survive the resampling** (``keep`` — the label ->
the surviving words mapping the stability report gives). A label whose every
pair is dropped has **no evidence left** and is then treated exactly like a
label with no train example: it carries the row's floor and can never be
predicted. The filter rides into the cross-fit too (``stability_min``): each
fold's filter is estimated on **that fold's own map**, so the fit side reads
the same kind of map the evaluated split is scored with.

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

from collections.abc import Mapping, Sequence

import numpy as np

from cluster_rag.text_processors.features.corpus_map import CorpusMap, build_corpus_map
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.pos_class import POS_CLASSES
from cluster_rag.text_processors.lexical.stability import HEAD_K, estimate_stability
from cluster_rag.text_processors.lexical.word_class import WORD_CLASSES

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


def _word_index(
    corpus_map: CorpusMap,
    *,
    restrict: Sequence[str] | None = None,
    restrict_pos: Sequence[str] | None = None,
    keep: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, tuple[tuple[int, float, float], ...]]:
    """The map's reverse tables as ``word -> ((column, log_odds, z_score), ...)``.

    Args:
        corpus_map: The corpus map (the train split's word↔label association).
        restrict: The word classes to **keep** (CS-33): the words of the other
            classes contribute nothing, which is the same as dropping them from
            the tables. ``None`` keeps every class (the full map).
        restrict_pos: The grammatical classes to **keep** (CS-34): the same
            rule on the POS axis. ``None`` keeps every class.
        keep: The **stability filter** (CS-36): the label -> the surviving
            words mapping. A pair that is not in the mapping's set for its
            label contributes nothing — it is as if it had never been in the
            table. ``None`` keeps every pair.

    Returns:
        The word index: every word the corpus holds, with the label columns it
        belongs to, the log-odds of each (label, word) pair and the z-score
        (CS-31's measure).

    Raises:
        ValueError: On an unknown class, or on an empty restriction.
    """
    columns = {label: index for index, label in enumerate(corpus_map.labels)}
    allowed = _allowed_classes(restrict)
    allowed_pos = _allowed_pos_classes(restrict_pos)
    index: dict[str, tuple[tuple[int, float, float], ...]] = {}
    classes = corpus_map.word_class_index() if allowed is not None else None
    for word, rows in corpus_map.per_word.items():
        if classes is not None and classes.class_of(word) not in allowed:
            continue
        if allowed_pos is not None and corpus_map.pos_classes.get(word, "") not in allowed_pos:
            continue
        entries = []
        for pair in rows:
            if keep is not None and word not in keep.get(pair.label, frozenset()):
                continue
            entries.append((columns[pair.label], pair.log_odds, pair.z_score))
        if entries:
            index[word] = tuple(entries)
    return index


def _allowed_classes(restrict: Sequence[str] | None) -> frozenset[str] | None:
    """The word classes a restriction keeps (``None`` = every class).

    Raises:
        ValueError: On an unknown class or an empty restriction.
    """
    if restrict is None:
        return None
    names = tuple(restrict)
    unknown = [name for name in names if name not in WORD_CLASSES]
    if unknown:
        raise ValueError(f"Unknown word class(es) {unknown}; expected some of {list(WORD_CLASSES)}")
    if not names:
        raise ValueError("a restriction must keep at least one word class")
    return frozenset(names)


def _allowed_pos_classes(restrict: Sequence[str] | None) -> frozenset[str] | None:
    """The grammatical classes a restriction keeps (``None`` = every class).

    Raises:
        ValueError: On an unknown class or an empty restriction.
    """
    if restrict is None:
        return None
    names = tuple(restrict)
    unknown = [name for name in names if name not in POS_CLASSES]
    if unknown:
        raise ValueError(
            f"Unknown grammatical class(es) {unknown}; expected some of {list(POS_CLASSES)}"
        )
    if not names:
        raise ValueError("a restriction must keep at least one grammatical class")
    return frozenset(names)


def score_cells(
    texts: Sequence[str],
    corpus_map: CorpusMap,
    *,
    scoring: str = "sum_logodds",
    restrict: Sequence[str] | None = None,
    restrict_pos: Sequence[str] | None = None,
    keep: Mapping[str, frozenset[str]] | None = None,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> np.ndarray:
    """Score every text against every label from the map's own weights.

    Args:
        texts: The texts being scored.
        corpus_map: The corpus map (built from the train split only).
        scoring: Which of the four scoring functions to apply (see
            :data:`SCORING_FUNCTIONS`).
        restrict: The word classes to **keep** (CS-33): the map with the other
            classes dropped — the "generic class dropped" and the
            "distinctive-only" maps. ``None`` is the full map.
        restrict_pos: The grammatical classes to **keep** (CS-34): the same
            rule on the POS axis (the "nouns only" and the "verbs only" maps).
        keep: The **stability filter** (CS-36): the label -> the surviving
            words mapping. A label with no surviving pair carries the row's
            floor, exactly like a label with no train example. ``None`` keeps
            every pair.
        stopwords: The tokenizer's stopwords — the map's own, so the words
            match the tables.

    Returns:
        A ``(n_texts, n_labels)`` float64 score matrix, higher = more likely.
        The labels with no train example carry the row's floor (the smallest
        seen score minus one), so they are ranked last in every row.

    Raises:
        ValueError: On an unknown scoring function, an unknown word class or an
            empty restriction.
    """
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring function {scoring!r}; expected one of {SCORING_FUNCTIONS}"
        )
    aggregation, measure, positive_only = _ARMS[scoring]
    labels = list(corpus_map.labels)
    index = _word_index(
        corpus_map,
        restrict=restrict,
        restrict_pos=restrict_pos,
        keep=keep,
    )
    scores = np.zeros((len(texts), len(labels)), dtype=np.float64)
    empty_columns = [
        column for column, label in enumerate(labels) if not corpus_map.label_counts.get(label)
    ]
    if keep is not None:
        # the stability filter (CS-36): a label whose every pair was dropped has
        # no evidence left, so it joins the empty tables (the floor).
        empty_columns = [column for column, label in enumerate(labels) if not keep.get(label)]
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
    restrict: Sequence[str] | None = None,
    restrict_pos: Sequence[str] | None = None,
    word_pos: Mapping[str, str] | None = None,
    keep: Mapping[str, frozenset[str]] | None = None,
    stability_min: float | None = None,
    stability_resamples: int = 500,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, np.ndarray]:
    """The **out-of-fold** score matrices: each fold's texts scored by the other folds.

    The map is built from the corpus it scores, so an in-fold score is
    optimistically biased — and most of all for the arms that use the log-odds
    the map itself computed. This is the fit side's honest reading: every text
    is scored by a map that never saw it.

    **The stability filter (CS-36)** rides in here too: when ``stability_min``
    is given, each fold's filter is estimated on **that fold's own map** (the
    fold's training texts resampled), so the fit side reads the same kind of
    map the evaluated split is scored with. The ``keep`` argument is the
    fallback for a caller that wants the same filter on every fold (it is what
    the map's own in-fold reading uses).

    Args:
        texts: The corpus's texts (the train split, at the fit side).
        label_sets: Each text's labels, in the same order.
        labels: The label space (the map's columns).
        scorings: Which scoring functions to compute.
        folds: How many folds to split the corpus into (at least 2).
        seed: The fold split's seed (deterministic).
        restrict: The word classes to **keep** (CS-33): the restriction rides
            into every fold's map, so the fit side reads the same map the
            evaluated split is scored with.
        restrict_pos: The grammatical classes to **keep** (CS-34).
        word_pos: The words' grammatical classes (CS-34) — a property of the
            word, not of a label, so every fold's map carries them.
        keep: The stability filter to apply on **every** fold (used when
            ``stability_min`` is not given).
        stability_min: The survival threshold: when given, each fold's filter
            is estimated on that fold's own map.
        stability_resamples: How many bootstrap resamples the fold's stability
            estimate rests on.
        stopwords: The tokenizer's stopwords.

    Returns:
        One ``(n_texts, n_labels)`` score matrix per scoring function, the rows
        in the corpus's own order.

    Raises:
        ValueError: On fewer than two folds, on an unknown scoring function,
            on an unknown class, or when the texts and the label sets have
            different lengths.
    """
    if len(texts) != len(label_sets):
        raise ValueError(f"the corpus has {len(texts)} texts and {len(label_sets)} label sets")
    if folds < 2:
        raise ValueError(f"a cross-fit needs at least 2 folds, got {folds}")
    unknown = [name for name in scorings if name not in SCORING_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown scoring function(s) {unknown}; expected {SCORING_FUNCTIONS}")
    _allowed_classes(restrict)
    _allowed_pos_classes(restrict_pos)
    n_texts = len(texts)
    order = np.random.default_rng(seed).permutation(n_texts)
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(order, folds)]
    out = {name: np.zeros((n_texts, len(labels)), dtype=np.float64) for name in scorings}
    for chunk in chunks:
        held = set(chunk.tolist())
        keep_indexes = [index for index in range(n_texts) if index not in held]
        corpus_map = build_corpus_map(
            [texts[index] for index in keep_indexes],
            [label_sets[index] for index in keep_indexes],
            labels,
            stopwords=stopwords,
            word_pos=word_pos,
        )
        fold_keep = keep
        if stability_min is not None:
            # the filter is estimated on THIS fold's own map (the fold's
            # training texts resampled), never on the whole corpus.
            report = estimate_stability(
                [texts[index] for index in keep_indexes],
                [label_sets[index] for index in keep_indexes],
                corpus_map,
                resamples=stability_resamples,
                seed=seed,
                head_k=HEAD_K,
                stopwords=stopwords,
            )
            fold_keep = report.label_survivors(stability_min)
        fold_texts = [texts[index] for index in chunk]
        for name in scorings:
            out[name][chunk] = score_cells(
                fold_texts,
                corpus_map,
                scoring=name,
                restrict=restrict,
                restrict_pos=restrict_pos,
                keep=fold_keep,
                stopwords=stopwords,
            )
    return out
