"""The lexical feature's own per-word decomposition (the track's CS-18, Version A).

``lex_label_coverage`` is the pack's **soft lexical overlap** between a text and
a label's *name*: the average over the name's content words of how much of each
word appears in the text (a word counts as covered in proportion to the share of
its character n-grams the text has). The average is therefore **exactly** the
mean of the words' own coverages — so the per-word contribution of one of a
label's words to one cell's feature is **exact, not an approximation**.

That is what this module exposes: the decomposition of the feature's own
definition, word by word. It is a map of the **word-overlap feature**, never of
the embedding model and never of the boosted-tree ensemble (a tree ensemble has
no per-word attribution; the pack's importances are per *feature*).

The pieces:

- :func:`label_word_contributions` — one cell's per-word decomposition (the
  exact share of every word of the label's name);
- :func:`label_word_classes` — every label's words marked **distinctive** (no
  other label in the space contains them) or **generic** (another label has
  them too), which is what the CS-18 falsification for question 2 is read
  against;
- :func:`build_lexical_map` and :func:`aggregate_lexical_map` — the map over a
  grid of cells and its aggregates (the per-word map, the concentration, the
  held-out labels' generic/distinctive split and the catch-alls' rows).

Pure core: numpy and the standard library only — no model client, no I/O.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    char_ngrams,
    content_tokens,
    soft_coverage,
)

# The word classes of the map: a word is **distinctive** when it is the label's
# own and **generic** when another label in the space has it too.
WORD_DISTINCTIVE = "distinctive"
WORD_GENERIC = "generic"


@dataclass(frozen=True)
class WordContribution:
    """One label word's contribution to one cell's lexical coverage.

    Attributes:
        word: The label name's content word (lower-cased, stopwords dropped —
            the coverage's own unit).
        coverage: The word's own soft coverage in [0, 1]: the share of the
            word's character n-grams the text has.
        contribution: The word's share of the cell's feature value, i.e.
            ``coverage / n_words`` — the exact per-word contribution.
    """

    word: str
    coverage: float
    contribution: float


def label_word_contributions(
    text: str,
    label_name: str,
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[WordContribution, ...]:
    """The exact per-word decomposition of ``lex_label_coverage`` for one cell.

    The feature is the **average** over the label name's content words of each
    word's soft coverage, so each word's contribution is its own coverage
    divided by the number of the name's content words. The contributions of one
    cell therefore sum to the feature's value exactly.

    Args:
        text: The cell's text.
        label_name: The cell's label name (the pack compares the **bare name**
            with the text).
        stopwords: The stopwords of the coverage (the feature's own unit: a
            word that is a stopword carries no weight).

    Returns:
        The name's content words and their contributions, in the name's order.
        An empty tuple when the name has no content word (the feature is then
        0.0).
    """
    text_ngrams = char_ngrams(text)
    tokens = content_tokens(label_name, stopwords)
    contributions: list[WordContribution] = []
    for token in tokens:
        grams = char_ngrams(token)
        if not grams:
            contributions.append(WordContribution(word=token, coverage=0.0, contribution=0.0))
            continue
        hits = sum(1 for gram in grams if gram in text_ngrams)
        coverage = hits / len(grams)
        contributions.append(
            WordContribution(
                word=token,
                coverage=coverage,
                contribution=coverage / len(tokens),
            )
        )
    return tuple(contributions)


def cell_coverage(
    text: str,
    label_name: str,
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> float:
    """The cell's ``lex_label_coverage`` value (the feature's own definition).

    The pack computes it with :func:`soft_coverage`; this is the same
    computation, exposed next to the decomposition so the two can be compared
    (the map's exactness check).
    """
    name_ngrams = [char_ngrams(token) for token in content_tokens(label_name, stopwords)]
    return soft_coverage(name_ngrams, char_ngrams(text))


def label_word_classes(
    labels: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, dict[str, str]]:
    """Every label's content words, marked **distinctive** or **generic**.

    A word is **distinctive** when no *other* label in the space contains it,
    and **generic** when another label has it too ("rights", "human",
    "recommendation" — the words every label has). This is the split the CS-18
    falsification for question 2 is read against: a held-out label's name
    transfers because of its own topic word (distinctive) or because of the
    word every label has (generic).

    Args:
        labels: The label space (the names only).
        stopwords: The stopwords of the coverage.

    Returns:
        ``{label: {word: "distinctive" | "generic"}}``.
    """
    tokens_per_label = {label: content_tokens(label, stopwords) for label in labels}
    holders: Counter[str] = Counter()
    for tokens in tokens_per_label.values():
        holders.update(set(tokens))
    return {
        label: {word: (WORD_DISTINCTIVE if holders[word] == 1 else WORD_GENERIC) for word in tokens}
        for label, tokens in tokens_per_label.items()
    }


@dataclass(frozen=True)
class LexicalMapCell:
    """One cell of the map: a (text, label) pair and its per-word decomposition.

    Attributes:
        text_row: The text's row in the grid.
        label_index: The label's column in the grid.
        label: The label's name.
        value: The cell's ``lex_label_coverage`` value.
        contributions: The label name's words and their contributions.
    """

    text_row: int
    label_index: int
    label: str
    value: float
    contributions: tuple[WordContribution, ...]


def build_lexical_map(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> list[LexicalMapCell]:
    """The map over a grid of cells: every (text, label) cell's decomposition.

    The text's n-grams are computed once per text and re-used for its cells
    (the same memoisation the pack does), so the map costs one pass over the
    texts and one soft-coverage pass per cell.

    Args:
        texts: The texts (the grid's rows).
        labels: The label space (the grid's columns).
        stopwords: The stopwords of the coverage.

    Returns:
        The cells, in the grid's own order (the texts' order, the labels'
        order).
    """
    text_ngrams = [char_ngrams(text) for text in texts]
    label_tokens = {label: content_tokens(label, stopwords) for label in labels}
    cells: list[LexicalMapCell] = []
    for row, ngrams in enumerate(text_ngrams):
        for column, label in enumerate(labels):
            tokens = label_tokens[label]
            contributions: list[WordContribution] = []
            total = 0.0
            for token in tokens:
                grams = char_ngrams(token)
                if not grams:
                    contributions.append(WordContribution(token, 0.0, 0.0))
                    continue
                coverage = sum(1 for gram in grams if gram in ngrams) / len(grams)
                total += coverage
                contributions.append(
                    WordContribution(
                        word=token,
                        coverage=coverage,
                        contribution=coverage / len(tokens),
                    )
                )
            value = (total / len(tokens)) if tokens else 0.0
            cells.append(
                LexicalMapCell(
                    text_row=row,
                    label_index=column,
                    label=label,
                    value=value,
                    contributions=tuple(contributions),
                )
            )
    return cells


def exactness_check(
    cells: Sequence[LexicalMapCell],
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, float | int]:
    """The decomposition's **exactness**: the largest deviation from the feature.

    The map claims the per-word contributions are *exact*, so the claim is
    checked: every cell's contributions must sum to the ``lex_label_coverage``
    value the pack computes for the same cell.

    Args:
        cells: The map's cells.
        texts: The texts (the grid's rows).
        labels: The label space (the grid's columns).
        stopwords: The stopwords of the coverage.

    Returns:
        The check's block: the number of cells and the largest absolute
        deviation.
    """
    text_ngrams = [char_ngrams(text) for text in texts]
    max_error = 0.0
    for cell in cells:
        tokens = content_tokens(labels[cell.label_index], stopwords)
        expected = soft_coverage(
            [char_ngrams(token) for token in tokens], text_ngrams[cell.text_row]
        )
        max_error = max(max_error, abs(expected - cell.value))
    return {"n_cells": len(cells), "max_abs_error": max_error}


def aggregate_lexical_map(
    cells: Sequence[LexicalMapCell],
    *,
    labels: Sequence[str],
    gold: np.ndarray,
    buckets: Mapping[str, str],
    word_classes: Mapping[str, Mapping[str, str]],
    held_out_labels: Sequence[str],
    catch_all_labels: Sequence[str],
    top_words: int = 25,
) -> dict[str, object]:
    """The map's aggregates: the per-word view, the concentration and the splits.

    The per-word view is read over the **gold cells** — the (text, label) pairs
    where the label is true for that text — because that is where the map's
    question lives ("which words of the label's name does the text actually
    contain when the label is true"). The grid-wide totals are reported beside
    them as the map's background.

    Args:
        cells: The map's cells (from :func:`build_lexical_map`).
        labels: The label space (the grid's columns).
        gold: The bool gold matrix of the grid.
        buckets: The ``label -> bucket`` mapping (head / torso / tail / unseen).
        word_classes: The per-label word classes (from
            :func:`label_word_classes`).
        held_out_labels: The labels the fit side never saw (the "unseen"
            bucket): the CS-18 question-2 split is read on them.
        catch_all_labels: The generic catch-alls ("General", "Other").
        top_words: How many words the headline word list carries.

    Returns:
        The aggregate block (JSON-friendly).

    Raises:
        ValueError: If the gold matrix's shape does not match the grid.
    """
    truth = np.asarray(gold, dtype=bool)
    n_labels = len(labels)
    if truth.ndim != 2 or truth.shape[1] != n_labels:
        raise ValueError(f"the gold matrix must have {n_labels} columns, got {truth.shape}")
    n_texts = truth.shape[0]
    if len(cells) != n_texts * n_labels:
        raise ValueError(f"the map carries {len(cells)} cells, the grid has {n_texts * n_labels}")

    # The grid-wide totals (every cell) and the gold-cell totals.
    grid_totals: Counter[str] = Counter()
    grid_class_totals: Counter[str] = Counter()
    gold_word_totals: dict[tuple[str, str], float] = {}
    label_gold_cells: Counter[str] = Counter()
    label_gold_total: Counter[str] = Counter()
    label_grid_total: Counter[str] = Counter()
    label_grid_cells: Counter[str] = Counter()
    word_labels: dict[str, set[str]] = {}
    for cell in cells:
        label_grid_total[cell.label] += cell.value
        label_grid_cells[cell.label] += 1
        is_gold = bool(truth[cell.text_row, cell.label_index])
        if is_gold:
            label_gold_cells[cell.label] += 1
            label_gold_total[cell.label] += cell.value
        for contribution in cell.contributions:
            grid_totals[contribution.word] += contribution.contribution
            word_labels.setdefault(contribution.word, set()).add(cell.label)
            word_class = word_classes.get(cell.label, {}).get(contribution.word, WORD_GENERIC)
            grid_class_totals[word_class] += contribution.contribution
            if is_gold:
                gold_word_totals[(cell.label, contribution.word)] = (
                    gold_word_totals.get((cell.label, contribution.word), 0.0)
                    + contribution.contribution
                )

    total_contribution = float(sum(grid_totals.values()))
    ranked = sorted(grid_totals.items(), key=lambda item: (-item[1], item[0]))
    shares: list[float] = []
    cumulative = 0.0
    for _, value in ranked:
        cumulative += value
        shares.append(cumulative / total_contribution if total_contribution else 0.0)

    # The held-out labels' split (the question-2 falsification).
    held_out = list(held_out_labels)
    catch_alls = list(catch_all_labels)
    held_out_generic = 0.0
    held_out_distinctive = 0.0
    for label in held_out:
        classes = word_classes.get(label, {})
        for word, word_class in classes.items():
            share = gold_word_totals.get((label, word), 0.0)
            if word_class == WORD_GENERIC:
                held_out_generic += share
            else:
                held_out_distinctive += share
    held_out_total = held_out_generic + held_out_distinctive

    per_word_rows = [
        {
            "word": word,
            "total_contribution": round(float(value), 6),
            "share": round(float(value) / total_contribution, 6) if total_contribution else 0.0,
            "n_labels": len(word_labels.get(word, ())),
            "labels": sorted(word_labels.get(word, ())),
        }
        for word, value in ranked
    ]

    return {
        "feature": "lex_label_coverage",
        "note": (
            "a map of the word-overlap feature's own definition (the pack's "
            "lexical block), NOT of the embedding model and NOT of the "
            "boosted-tree ensemble: a tree ensemble has no per-word "
            "attribution"
        ),
        "grid": {"n_texts": n_texts, "n_labels": n_labels, "n_cells": len(cells)},
        "concentration": {
            "total_contribution": round(total_contribution, 6),
            "n_words": len(grid_totals),
            "top1_word_share": round(shares[0], 6) if shares else None,
            "top5_word_share": round(shares[4], 6) if len(shares) > 4 else None,
            "top20_word_share": round(shares[19], 6) if len(shares) > 19 else None,
            "note": (
                "the share of the grid's total per-word contribution carried by "
                "the top-ranked words: a value near 1 means one word carries "
                "the feature, a value near 0 means the signal is distributed"
            ),
        },
        "word_class_split": {
            "distinctive": round(float(grid_class_totals.get(WORD_DISTINCTIVE, 0.0)), 6),
            "generic": round(float(grid_class_totals.get(WORD_GENERIC, 0.0)), 6),
            "generic_share": (
                round(
                    float(grid_class_totals.get(WORD_GENERIC, 0.0)) / total_contribution,
                    6,
                )
                if total_contribution
                else None
            ),
            "note": (
                "the grid's total per-word contribution split between the words "
                "a label owns (distinctive) and the words other labels have too "
                "(generic)"
            ),
        },
        "held_out_labels": {
            "labels": held_out,
            "generic_contribution": round(held_out_generic, 6),
            "distinctive_contribution": round(held_out_distinctive, 6),
            "generic_share": (
                round(held_out_generic / held_out_total, 6) if held_out_total else None
            ),
            "per_label": [
                {
                    "label": label,
                    "bucket": buckets.get(label),
                    "n_gold_cells": int(label_gold_cells.get(label, 0)),
                    "gold_coverage_mean": _mean(
                        label_gold_total.get(label, 0.0), label_gold_cells.get(label, 0)
                    ),
                    "words": _label_words(label, gold_word_totals, word_classes),
                }
                for label in held_out
            ],
            "note": (
                "the pre-registered falsification for question 2: the held-out "
                "labels' coverage must be carried by their OWN distinctive "
                "words, not by the generic ones every label has"
            ),
        },
        "catch_all_labels": {
            "labels": catch_alls,
            "per_label": [
                {
                    "label": label,
                    "bucket": buckets.get(label),
                    "n_gold_cells": int(label_gold_cells.get(label, 0)),
                    "gold_coverage_mean": _mean(
                        label_gold_total.get(label, 0.0), label_gold_cells.get(label, 0)
                    ),
                    "words": _label_words(label, gold_word_totals, word_classes),
                }
                for label in catch_alls
            ],
        },
        "per_label": [
            {
                "label": label,
                "bucket": buckets.get(label),
                "mean_coverage": _mean(
                    label_grid_total.get(label, 0.0), label_grid_cells.get(label, 0)
                ),
                "gold_coverage_mean": _mean(
                    label_gold_total.get(label, 0.0), label_gold_cells.get(label, 0)
                ),
                "n_gold_cells": int(label_gold_cells.get(label, 0)),
            }
            for label in labels
        ],
        "top_words": per_word_rows[:top_words],
        "per_word": per_word_rows,
    }


def _mean(total: float, count: int) -> float | None:
    """A mean, or ``None`` when the denominator is zero."""
    return round(total / count, 6) if count else None


def _label_words(
    label: str,
    gold_word_totals: Mapping[tuple[str, str], float],
    word_classes: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    """One label's words and their mean contribution over its **gold** cells."""
    classes = word_classes.get(label, {})
    rows = []
    for word, word_class in classes.items():
        total = gold_word_totals.get((label, word), 0.0)
        rows.append(
            {
                "word": word,
                "word_class": word_class,
                "gold_contribution": round(total, 6),
            }
        )
    rows.sort(key=lambda row: (-float(row["gold_contribution"]), str(row["word"])))
    return rows
