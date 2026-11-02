"""The **phrase table** (the track's CS-39): the map's unit, one step finer.

The corpus map (CS-23) is a relation between labels and **single words**. This
module builds the same relation over the corpus's **bigrams**: the (label,
phrase) pairs of the train corpus, with the same measures (the log-odds, the
z-score, the counts) — the estimator's 2×2 table over the phrase's presence in
a text ✓.

**The unit.** The consecutive bigrams of the track's own **token stream**
(`tokenize` — the lower-cased alphanumeric tokens, the stopwords **kept**).
The stopwords stay because the genre's formulas live in them ("continue to",
"ensure that") and the corpus's formulaic structure is what the phrase table
is for; the topical phrases ("human rights", "death penalty") are unaffected
by the choice.

**The readout** answers three questions: what the phrase table's **shape** is
(how many phrases, how many pairs, how fragile they are — the hapax share),
**how much of it is the register** (the class split, CS-33's own machinery
applied to phrases), and **how much more a phrase says than its two words**
(the pair interaction: the phrase's log-odds minus the sum of its words' log-
odds in the same label's table).

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cluster_rag.text_processors.features.corpus_map import (
    CorpusMap,
    build_corpus_map,
    log_odds,
)
from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    tokenize,
)
from cluster_rag.text_processors.lexical.word_class import (
    class_readout,
)


def phrase_units(text: str) -> list[str]:
    """The text's bigrams over the track's own token stream.

    Args:
        text: The text.

    Returns:
        The bigrams, in the text's own order ("death penalty" is one unit).
    """
    tokens = tokenize(text)
    return [f"{first} {second}" for first, second in zip(tokens, tokens[1:], strict=False)]


def build_phrase_map(
    texts: Sequence[str],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> CorpusMap:
    """Build the phrase table: the (label, bigram) pairs of the corpus.

    The corpus is whatever the caller passes — the track's **train** split, so
    no val text and no val label enters the counts and ``test`` is not touched.
    A label with no example in the corpus gets an **empty** table (never a
    fabricated one).

    The shape is the corpus map's own (:class:`CorpusMap`), with the phrase in
    the place of the word — the two tables are then read by the same code, and
    the **unit** is the only thing that differs ✓.

    Args:
        texts: The corpus's texts.
        label_sets: Each text's labels, in the same order as ``texts``.
        labels: The label space (every label, even the ones the corpus never
            carries — they are reported as empty).
        stopwords: The tokenizer's stopwords (the map's own, for the
            tokenization).

    Returns:
        The phrase table.

    Raises:
        ValueError: If the texts and the label sets have different lengths.
    """
    return build_corpus_map(
        texts,
        label_sets,
        labels,
        stopwords=stopwords,
        units=phrase_units,
    )


def _log_odds(*, n_with: int, df: int, n_label: int, n_texts: int) -> float:
    """The 0.5-smoothed signed log-odds of one (label, phrase) pair.

    The corpus map's own formula — the same 2×2 table, over the phrase's
    presence.
    """
    return log_odds(n_with=n_with, df=df, n_label=n_label, n_texts=n_texts)


@dataclass(frozen=True)
class PhraseInteraction:
    """One (label, phrase) pair's **interaction** with its two words.

    Attributes:
        label: The label.
        phrase: The phrase.
        log_odds: The phrase's own log-odds in the label's table.
        words_log_odds: The sum of the two words' log-odds in the same table
            (0 for a word the label's table does not hold).
        interaction: ``log_odds - words_log_odds`` — how much more the phrase
            says than its parts.
    """

    label: str
    phrase: str
    log_odds: float
    words_log_odds: float
    interaction: float

    def to_row(self) -> dict[str, object]:
        """The interaction as a JSON-friendly row (the artifact's own shape)."""
        return {
            "label": self.label,
            "phrase": self.phrase,
            "log_odds": round(self.log_odds, 6),
            "words_log_odds": round(self.words_log_odds, 6),
            "interaction": round(self.interaction, 6),
        }


def phrase_interactions(
    phrase_map: CorpusMap,
    word_map: CorpusMap,
) -> tuple[PhraseInteraction, ...]:
    """Every (label, phrase) pair's interaction with its two words.

    The phrase's log-odds minus the sum of its two words' log-odds **in the
    same label's table** ✓ — a word the label's table does not hold counts as
    0 (it contributes nothing on its own) ✓.

    Args:
        phrase_map: The phrase table.
        word_map: The word map (the words' own log-odds).

    Returns:
        The interactions (one per pair of the phrase table).
    """
    out: list[PhraseInteraction] = []
    for label in phrase_map.labels:
        word_rows = {pair.word: pair.log_odds for pair in word_map.topic_rows(label)}
        for pair in phrase_map.topic_rows(label):
            words = pair.word.split(" ", 1)
            words_log_odds = sum(word_rows.get(word, 0.0) for word in words)
            out.append(
                PhraseInteraction(
                    label=label,
                    phrase=pair.word,
                    log_odds=pair.log_odds,
                    words_log_odds=words_log_odds,
                    interaction=pair.log_odds - words_log_odds,
                )
            )
    return tuple(out)


def phrase_map_block(
    phrase_map: CorpusMap,
    word_map: CorpusMap,
    *,
    n_top: int = 20,
) -> dict[str, object]:
    """The **phrase table**'s readout (CS-39): the artifact's own block.

    The block carries the table's **shape** (the distinct phrases, the pairs
    and the **hapax share** — the round's fragility one unit finer), the
    **class split** over the phrase table (CS-33's own machinery ✓) beside the
    word table's own numbers, and the **pair interaction** (how much more a
    phrase says than its two words) with its mean, its distribution and the
    pairs that carry the most.

    Args:
        phrase_map: The phrase table.
        word_map: The word map (the word table's own numbers and the words'
            log-odds).
        n_top: How many interactions the readout lists.

    Returns:
        The block (JSON-friendly).
    """
    classes = phrase_map.word_class_index()
    word_classes = word_map.word_class_index()
    readout = class_readout(
        (pair.word_class, pair.log_odds)
        for label in phrase_map.labels
        for pair in phrase_map.topic_rows(label)
    )
    word_readout = class_readout(
        (pair.word_class, pair.log_odds)
        for label in word_map.labels
        for pair in word_map.topic_rows(label)
    )
    interactions = phrase_interactions(phrase_map, word_map)
    values = sorted(pair.interaction for pair in interactions)
    n_pairs = len(interactions)
    hapax = sum(
        1
        for label in phrase_map.labels
        for pair in phrase_map.topic_rows(label)
        if pair.n_with == 1
    )
    top = sorted(interactions, key=lambda pair: (-pair.interaction, pair.label, pair.phrase))
    generic_share = readout["generic"].n_pairs / n_pairs if n_pairs else 0.0
    word_generic_share = (
        word_readout["generic"].n_pairs / word_map.n_pairs if word_map.n_pairs else 0.0
    )
    return {
        "available": True,
        "definitions": {
            "unit": (
                "the consecutive bigrams of the track's own token stream (the "
                "stopwords kept: the genre's formulas live in them)"
            ),
            "hapax_share": (
                "the share of the phrase table's pairs that rest on a single "
                "train text -- the round's fragility, one unit finer"
            ),
            "interaction": (
                "the phrase's log-odds minus the sum of its two words' log-odds "
                "in the same label's table: how much more the phrase says than "
                "its parts"
            ),
        },
        "shape": {
            "n_texts": phrase_map.n_texts,
            "n_labels": len(phrase_map.labels),
            "n_labels_with_a_table": sum(
                1 for label in phrase_map.labels if phrase_map.label_counts.get(label)
            ),
            "n_phrases": phrase_map.n_words,
            "n_pairs": n_pairs,
            "hapax_pairs": hapax,
            "hapax_share": round(hapax / n_pairs, 6) if n_pairs else 0.0,
        },
        "classes": {
            "n_labelled_tables": classes.n_labelled_tables,
            "generic_threshold": classes.generic_threshold,
            "words_by_class": classes.counts_by_class(),
            "readout": {name: row.to_row("mean_log_odds") for name, row in readout.items()},
            "generic_share": round(generic_share, 6),
            "word_table": {
                "n_labelled_tables": word_classes.n_labelled_tables,
                "generic_threshold": word_classes.generic_threshold,
                "readout": {
                    name: row.to_row("mean_log_odds") for name, row in word_readout.items()
                },
                "generic_share": round(word_generic_share, 6),
            },
        },
        "interaction": {
            "n_pairs": n_pairs,
            "mean": round(sum(values) / len(values), 6) if values else 0.0,
            "median": round(values[len(values) // 2], 6) if values else 0.0,
            "share_positive": round(sum(1 for value in values if value > 0) / len(values), 6)
            if values
            else 0.0,
            "quantiles": {
                "q10": round(values[int(0.10 * (len(values) - 1))], 6) if values else 0.0,
                "q90": round(values[int(0.90 * (len(values) - 1))], 6) if values else 0.0,
            },
            "top": [pair.to_row() for pair in top[:n_top]],
        },
    }
