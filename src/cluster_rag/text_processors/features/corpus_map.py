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
from cluster_rag.text_processors.lexical.lexicon import pair_measures, prior_total

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
        log_odds_informative: The **informative-prior** log-odds (CS-31): the
            same table with the word's global document frequency as the
            pseudo-count (Monroe et al. 2008's Dirichlet prior, adapted to the
            document level).
        standard_error: The standard error of that log-odds.
        z_score: δ over the standard error — the plan's z-score.
        g2: The likelihood-ratio G² statistic on the same 2×2 table (1 df).
        p_value: The G² statistic's p-value.
        log_odds_conditional: The log-odds with the 0.5 added only where a
            cell is actually zero (the conditional smoothing).
    """

    label: str
    word: str
    n_with: int
    df: int
    lift: float
    log_odds: float
    log_odds_informative: float = 0.0
    standard_error: float = 0.0
    z_score: float = 0.0
    g2: float = 0.0
    p_value: float = 1.0
    log_odds_conditional: float = 0.0

    def to_row(self) -> dict[str, object]:
        """The pair as a JSON-friendly row (the artifact's own shape)."""
        return {
            "word": self.word,
            "n_with": self.n_with,
            "df": self.df,
            "lift": round(self.lift, 6),
            "log_odds": round(self.log_odds, 6),
            "log_odds_informative": round(self.log_odds_informative, 6),
            "standard_error": round(self.standard_error, 6),
            "z_score": round(self.z_score, 6),
            "g2": round(self.g2, 6),
            "p_value": round(self.p_value, 8),
            "log_odds_conditional": round(self.log_odds_conditional, 6),
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
    # The estimator's prior total (CS-31): the corpus's own (text, word)
    # incidences plus the floor's contribution. A map-level constant.
    alpha0 = prior_total(document_frequency.values())
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
            measures = pair_measures(n_with, df, n_label, n_texts, alpha0=alpha0)
            pair = CorpusPair(
                label=label,
                word=word,
                n_with=n_with,
                df=df,
                lift=(n_with / df) / p_label,
                log_odds=log_odds(n_with=n_with, df=df, n_label=n_label, n_texts=n_texts),
                log_odds_informative=measures.log_odds_informative,
                standard_error=measures.standard_error,
                z_score=measures.z_score,
                g2=measures.g2,
                p_value=measures.p_value,
                log_odds_conditional=measures.log_odds_conditional,
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


def estimator_block(corpus_map: CorpusMap) -> dict[str, object]:
    """The estimator's summary block (CS-31): what the new measures say.

    The block carries the measures' own definitions, the **head word** of every
    labelled table under each of them (with its counts), how far the measures
    agree on the head word, and the conditional smoothing's sign-flip share —
    the four readings the change set's falsification is stated against.

    Args:
        corpus_map: The corpus map.

    Returns:
        The block (JSON-friendly).
    """
    heads: dict[str, dict[str, dict[str, object]]] = {
        name: {} for name in ("log_odds", "log_odds_informative", "z_score", "p_value")
    }
    for label in corpus_map.labels:
        rows = corpus_map.topic_rows(label)
        if not rows:
            continue
        for name in heads:
            best = max(rows, key=lambda row: (_head_value(row, name), row.word))
            heads[name][label] = {
                "word": best.word,
                "n_with": best.n_with,
                "df": best.df,
                "log_odds": round(best.log_odds, 6),
                "log_odds_informative": round(best.log_odds_informative, 6),
                "z_score": round(best.z_score, 6),
                "p_value": round(best.p_value, 8),
            }
    labelled = [label for label in corpus_map.labels if corpus_map.topic_rows(label)]
    n_labelled = len(labelled)
    changed = sum(
        1
        for label in labelled
        if heads["z_score"][label]["word"] != heads["log_odds"][label]["word"]
    )
    n_pairs = corpus_map.n_pairs
    sign_flips = sum(
        1
        for label in corpus_map.labels
        for pair in corpus_map.topic_rows(label)
        if (pair.log_odds > 0) != (pair.log_odds_conditional > 0)
    )
    return {
        "measures": {
            "log_odds": ("the 0.5-smoothed signed log-odds over the 2x2 table (CS-23, unchanged)"),
            "log_odds_informative": (
                "the informative-prior log-odds: the same table with the word's "
                "global document frequency as the pseudo-count (alpha_w = df + "
                "0.01; alpha_0 = the corpus's own (text, word) incidences), in the "
                "form of Monroe et al. 2008's Dirichlet prior adapted to the "
                "document level"
            ),
            "standard_error": "sqrt(1/(a+alpha_w) + 1/(b+alpha_w))",
            "z_score": "the informative log-odds over its standard error",
            "g2": "the likelihood-ratio G^2 statistic on the same 2x2 table (1 df)",
            "p_value": "the G^2 statistic's p-value (the chi-square 1-df survival)",
            "log_odds_conditional": (
                "the log-odds with the 0.5 added only where a cell is actually zero "
                "(the conditional smoothing)"
            ),
        },
        "heads": heads,
        "head_agreement": {
            "n_labels_with_a_table": n_labelled,
            "z_score_head_differs_from_log_odds": changed,
            "z_score_head_is_a_hapax": sum(
                1 for label in labelled if heads["z_score"][label]["df"] == 1
            ),
            "log_odds_head_is_a_hapax": sum(
                1 for label in labelled if heads["log_odds"][label]["df"] == 1
            ),
            "note": (
                "the z-score's head is the label's most significant word under the "
                "informative prior and the standard error; the informative prior "
                "shrinks the COMMON words (the register) and the standard error is "
                "what shrinks the rare ones"
            ),
        },
        "conditional_smoothing": {
            "n_pairs": n_pairs,
            "n_sign_flips": sign_flips,
            "sign_flip_share": round(sign_flips / n_pairs, 6) if n_pairs else 0.0,
        },
    }


def _head_value(pair: CorpusPair, measure: str) -> float:
    """The value a measure is read by when it picks a table's head word.

    The p-value is read the other way round: the head is the **smallest**
    p-value (the most significant word), so the key is negated.
    """
    if measure == "log_odds":
        return pair.log_odds
    if measure == "log_odds_informative":
        return pair.log_odds_informative
    if measure == "z_score":
        return pair.z_score
    return -pair.p_value


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
            "log_odds_informative": (
                "the informative-prior log-odds (CS-31): the same table with the "
                "word's global document frequency as the pseudo-count"
            ),
            "standard_error": "the standard error of the informative log-odds",
            "z_score": "the informative log-odds over its standard error",
            "g2": "the likelihood-ratio G^2 statistic on the same 2x2 table (1 df)",
            "p_value": "the G^2 statistic's p-value",
            "log_odds_conditional": (
                "the log-odds with the 0.5 added only where a cell is actually zero"
            ),
            "note": (
                "no thresholds: every pair that occurs is stored, with the counts "
                "behind it, so any number can be recomputed and audited. The "
                "noise is handled at display time (the counts sit next to the "
                "number), never by dropping data."
            ),
        },
        "estimator": estimator_block(corpus_map),
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
