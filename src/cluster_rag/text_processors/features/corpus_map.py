"""The **corpus map** (the track's CS-23): the corpus's own word↔label association.

The hit map's four modes (CS-18's topic-side decomposition + the causal
ablation, CS-19's text-side decomposition + the surrogate attribution) all
answer a question about the **model** or about a **feature**. None of them
answers the plain statistical question: *which words go with which labels in
the corpus itself?* That is this module.

**The measures** (every one stored with the counts behind it, so any number can
be recomputed and audited):

- ``n_with`` — the corpus texts that contain the word **and** carry the label;
- ``df`` — the corpus texts that contain the word at all;
- ``lift`` = ``P(label | word present) / P(label)`` — how much likelier a text
  with this word is to carry the label; 1.0 = the word says nothing, above 1.0
  it pulls the label, below 1.0 it pushes it away;
- ``log_odds`` = ``log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))`` over the 2×2
  table (``a`` word+label, ``b`` word+no label, ``c`` no word+label, ``d``
  neither) — the 0.5-smoothed signed version: above 0 with the label, below 0
  against it, 0 nothing. It is preferred to lift because lift explodes on rare
  words (a word in 2 of 2 texts gives lift 57 and means nothing).

**No thresholds.** Every (label, word) pair that occurs is stored, with the
four counts; the noise is handled at **display** time (the counts sit next to
the number and the viewer offers a ``--min-df`` filter, default off), never by
dropping data.

**The corpus is the train split.** The map is built from the **training**
texts and their labels, so no val text and no val label enters the counts and
``test`` is not touched at all. A label with no train example (the eight
held-out labels) has an **empty** table by construction — reported as empty,
never fabricated.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens

# The log-odds' additive constant: the "add one half" of the conventional 2×2
# table. It keeps a zero cell from sending the log to infinity, so a word that
# never co-occurs with a label still gets a finite (negative) score.
SMOOTHING = 0.5


@dataclass(frozen=True)
class CorpusPair:
    """One (label, word) pair of the corpus map, with the counts behind it.

    Attributes:
        label: The label's name.
        word: The word, as the track's own content-word tokenizer writes it
            (lower-cased alphanumeric tokens, the stopwords dropped).
        n_with: The corpus texts that have **both** the word and the label.
        df: The corpus texts that have the word at all.
        lift: ``P(label | word present) / P(label)``.
        log_odds: The 0.5-smoothed signed log-odds over the 2×2 table.
    """

    label: str
    word: str
    n_with: int
    df: int
    lift: float
    log_odds: float

    def to_row(self) -> dict[str, object]:
        """The pair as a JSON-friendly row (the artifact's own shape)."""
        return {
            "word": self.word,
            "n_with": self.n_with,
            "df": self.df,
            "lift": round(self.lift, 6),
            "log_odds": round(self.log_odds, 6),
        }


@dataclass(frozen=True)
class CorpusMap:
    """The corpus map: the label space's word tables and the reverse tables.

    Attributes:
        n_texts: The corpus's size (the train texts' count).
        labels: The label space (every label, including the ones with no
            train example).
        label_counts: How many corpus texts carry each label.
        per_label: Each label's word table, strongest log-odds first (the
            label's own view of the corpus).
        per_word: Each word's label table, strongest log-odds first (the same
            data read the other way).
    """

    n_texts: int
    labels: tuple[str, ...]
    label_counts: Mapping[str, int]
    per_label: Mapping[str, tuple[CorpusPair, ...]]
    per_word: Mapping[str, tuple[CorpusPair, ...]]

    @property
    def n_words(self) -> int:
        """How many distinct words the corpus holds."""
        return len(self.per_word)

    @property
    def n_pairs(self) -> int:
        """How many (label, word) pairs the corpus holds."""
        return sum(len(rows) for rows in self.per_label.values())

    def topic_rows(self, label: str) -> tuple[CorpusPair, ...]:
        """One label's word table (empty when the label has no train example)."""
        return tuple(self.per_label.get(label, ()))

    def word_rows(self, word: str) -> tuple[CorpusPair, ...]:
        """One word's label table (empty when the corpus does not have the word)."""
        return tuple(self.per_word.get(word, ()))

    def labels_without_examples(self) -> tuple[str, ...]:
        """The labels the corpus has no example of (the held-out ones)."""
        return tuple(label for label in self.labels if not self.label_counts.get(label))


def build_corpus_map(
    texts: Sequence[str],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> CorpusMap:
    """Build the corpus map from a corpus's texts and their labels.

    The corpus is whatever the caller passes — the track's **train** split, so
    no val text and no val label enters the counts. A label with no example in
    the corpus gets an **empty** table (never a fabricated one).

    Args:
        texts: The corpus's texts.
        label_sets: Each text's labels, in the same order as ``texts``.
        labels: The label space (every label, even the ones the corpus never
            carries — they are reported as empty).
        stopwords: The stopwords of the tokenizer.

    Returns:
        The corpus map.

    Raises:
        ValueError: If the texts and the label sets have different lengths.
    """
    if len(texts) != len(label_sets):
        raise ValueError(f"the corpus has {len(texts)} texts and {len(label_sets)} label sets")
    label_space = tuple(labels)
    document_frequency: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    pairs: dict[str, Counter[str]] = {label: Counter() for label in label_space}
    for text, record_labels in zip(texts, label_sets, strict=True):
        words = set(content_tokens(text, stopwords))
        document_frequency.update(words)
        for label in set(record_labels):
            if label not in pairs:
                # a label outside the space is ignored: the map's columns are
                # the label space's, so the space is the contract.
                continue
            label_counts[label] += 1
            pairs[label].update(words)

    n_texts = len(texts)
    per_label: dict[str, tuple[CorpusPair, ...]] = {}
    per_word: dict[str, list[CorpusPair]] = {}
    for label in label_space:
        n_label = label_counts.get(label, 0)
        if not n_label or not n_texts:
            # no train example: the table is empty by construction
            per_label[label] = ()
            continue
        p_label = n_label / n_texts
        rows: list[CorpusPair] = []
        for word, n_with in pairs[label].items():
            df = document_frequency.get(word, 0)
            if not df:
                continue
            pair = CorpusPair(
                label=label,
                word=word,
                n_with=n_with,
                df=df,
                lift=(n_with / df) / p_label,
                log_odds=log_odds(n_with=n_with, df=df, n_label=n_label, n_texts=n_texts),
            )
            rows.append(pair)
        rows.sort(key=lambda row: (-row.log_odds, row.word))
        per_label[label] = tuple(rows)
        for pair in rows:
            per_word.setdefault(pair.word, []).append(pair)
    frozen_per_word = {
        word: tuple(sorted(rows, key=lambda row: (-row.log_odds, row.label)))
        for word, rows in per_word.items()
    }
    return CorpusMap(
        n_texts=n_texts,
        labels=label_space,
        label_counts=dict(label_counts),
        per_label=per_label,
        per_word=frozen_per_word,
    )


def log_odds(*, n_with: int, df: int, n_label: int, n_texts: int) -> float:
    """The 0.5-smoothed signed log-odds of one (label, word) pair.

    The 2×2 table over the corpus: ``a`` = the texts with the word **and** the
    label, ``b`` = the texts with the word but not the label, ``c`` = the texts
    with the label but not the word, ``d`` = the texts with neither. The score
    is ``log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))``: above 0 with the label,
    below 0 against it, 0 nothing.

    Args:
        n_with: The texts with both the word and the label.
        df: The texts with the word at all.
        n_label: The texts with the label at all.
        n_texts: The corpus's size.

    Returns:
        The log-odds.

    Raises:
        ValueError: If the counts do not form a 2×2 table (a negative cell).
    """
    a = n_with
    b = df - n_with
    c = n_label - n_with
    d = n_texts - n_label - b
    if min(a, b, c, d) < 0:
        raise ValueError(
            "the counts do not form a 2x2 table: "
            f"n_with={n_with}, df={df}, n_label={n_label}, n_texts={n_texts}"
        )
    return math.log((a + SMOOTHING) / (b + SMOOTHING)) + math.log((d + SMOOTHING) / (c + SMOOTHING))


def corpus_map_block(corpus_map: CorpusMap, *, split: str = "train") -> dict[str, object]:
    """The corpus map as the artifact's own block (JSON-friendly).

    The block carries the corpus's shape, the four measures' own definitions
    and **every** (label, word) pair that occurs — in both directions (the
    label's word table and the word's label table), so the viewer can read the
    association either way without recomputing anything.

    Args:
        corpus_map: The map (from :func:`build_corpus_map`).
        split: The split the corpus is (the artifact records it).

    Returns:
        The block (JSON-friendly).
    """
    labels_without = corpus_map.labels_without_examples()
    return {
        "corpus": {
            "split": split,
            "n_texts": corpus_map.n_texts,
            "n_labels": len(corpus_map.labels),
            "n_words": corpus_map.n_words,
            "n_pairs": corpus_map.n_pairs,
            "labels_with_examples": sum(
                1 for label in corpus_map.labels if corpus_map.label_counts.get(label)
            ),
            "labels_without_examples": list(labels_without),
            "note": (
                "the train split only: no val text and no val label enters the "
                "counts and test is not touched. A label with no train example "
                "has an empty table by construction — reported as empty, never "
                "fabricated."
            ),
        },
        "measures": {
            "n_with": "the corpus texts that have the word AND the label",
            "df": "the corpus texts that have the word at all",
            "lift": (
                "P(label | word present) / P(label): 1.0 says nothing, above 1.0 "
                "pulls the label, below 1.0 pushes it away"
            ),
            "log_odds": (
                "log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5)) over the 2x2 table "
                "(a word+label, b word+no label, c no word+label, d neither): "
                "above 0 with the label, below 0 against it, 0 nothing"
            ),
            "note": (
                "no thresholds: every pair that occurs is stored, with the counts "
                "behind it, so any number can be recomputed and audited. The "
                "noise is handled at display time (the counts sit next to the "
                "number), never by dropping data."
            ),
        },
        "per_label": [
            {
                "label": label,
                "n_corpus_texts": int(corpus_map.label_counts.get(label, 0)),
                "n_words": len(corpus_map.topic_rows(label)),
                "words": [pair.to_row() for pair in corpus_map.topic_rows(label)],
                **(
                    {}
                    if corpus_map.label_counts.get(label)
                    else {"note": "no train example: the table is empty by construction"}
                ),
            }
            for label in corpus_map.labels
        ],
        "per_word": [
            {
                "word": word,
                "df": int(rows[0].df),
                "labels": [
                    {
                        "label": pair.label,
                        "n_with": pair.n_with,
                        "lift": round(pair.lift, 6),
                        "log_odds": round(pair.log_odds, 6),
                    }
                    for pair in rows
                ],
            }
            for word, rows in sorted(corpus_map.per_word.items())
        ],
    }
