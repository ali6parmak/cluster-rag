"""The **label lexicon** (the track's CS-35): the label-side representation.

The **corpus map** (CS-23/CS-31) *already is* the label→word lexicon: every
(label, word) pair of the train corpus, with the counts behind it. What this
module adds is the **label-side representation** — per label, the words that
characterise it — so the lexicon can be used as a *channel* (the label side of
the pack) instead of only as a description (CS-33's readout) or a score (CS-30's
classifier).

**Two routes.**

- The **mined words** (the PIFA idea, X-Transformer, KDD 2020: the label's
  embedding is the TF-IDF centroid of its positive instances). For a label
  ``L`` with ``n_L`` train instances and a word ``w`` in its table, the
  centroid's coordinate for ``w`` is exactly the mean of the instances' TF-IDF
  values, i.e. ``tf(L,w) × idf(w)`` with ``tf`` the share of the label's
  instances that contain the word and ``idf`` the word's inverse document
  frequency in the corpus. The label's mined words are the **top K** of those
  coordinates.
- The **expansion** (the labels with no train example — the eight held-out
  ones). The label's **name** is matched against the **corpus's vocabulary**:
  a name token that is in the vocabulary contributes exactly that word, a token
  that is not contributes the corpus words sharing a prefix of at least
  ``expansion_min_prefix`` characters with it (a crude stemmer). **No labels
  are involved**: the route reads the corpus's *words* and their document
  frequencies, never a label association.

**The filter (CS-45).** The mined words carry the corpus's **register**
("ensure", "right", "system") beside its topic. A build may keep only the words
of the requested **classes** (CS-33's: ``distinctive`` = one or two labelled
tables, ``shared`` = 3-14, ``generic`` = at least a quarter): the class is the
map's own, the filter applies to the **candidate pool before the top-K cut**, and
it applies to both routes — an expansion label left without a word falls back to
its name. ``None`` (the default) keeps every class, so every earlier artifact
keeps its exact content.

**The routes are the label-holdout's answer.** The mined route needs train
examples, so it exists for the sixty labels that have some and is empty for the
eight that have none; the expansion route is the only lexical path to those
eight, and it uses no gold label and no val text.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from cluster_rag.text_processors.features.corpus_map import CorpusMap
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.word_class import WORD_CLASSES

# The number of mined words per label (CS-35's K). It is fixed at 10 so the
# mined-words label side has the same length as the catalog description's (6-11
# content tokens): the comparison against the name+description channel then
# changes the label side's *content*, not its length.
DEFAULT_TOP_K = 10

# The minimum number of characters two words must share to count as a
# prefix match in the expansion route (a crude stemmer: "seekers" -> "seek",
# "belief" -> "beliefs"). Four characters is the shortest prefix that keeps
# "refugees" -> "re" out.
DEFAULT_EXPANSION_MIN_PREFIX = 4

# The route a label's words came from.
ROUTE_MINED = "mined"
ROUTE_EXPANDED = "expanded"
ROUTE_FALLBACK = "fallback"

# The routes, in the order the artifact's summary counts them.
ROUTES: tuple[str, ...] = (ROUTE_MINED, ROUTE_EXPANDED, ROUTE_FALLBACK)


@dataclass(frozen=True)
class LexiconWord:
    """One word of a label's representation, with the numbers behind it.

    Attributes:
        word: The word itself (the track's own content-word tokenizer wrote it).
        weight: The word's weight in the label's representation: the TF-IDF
            centroid's coordinate for the mined route, the word's inverse
            document frequency for the expansion route (no labels involved).
        n_with: The corpus texts that have **both** the word and the label (the
            mined route; 0 for the expansion route).
        df: The corpus texts that have the word at all.
        log_odds: The map's log-odds for the pair (the mined route; 0 for the
            expansion route).
        z_score: The map's z-score for the pair (CS-31's measure; 0 for the
            expansion route).
    """

    word: str
    weight: float
    n_with: int = 0
    df: int = 0
    log_odds: float = 0.0
    z_score: float = 0.0

    def to_row(self) -> dict[str, object]:
        """The word as a JSON-friendly row (the artifact's own shape)."""
        return {
            "word": self.word,
            "weight": round(self.weight, 6),
            "n_with": int(self.n_with),
            "df": int(self.df),
            "log_odds": round(self.log_odds, 6),
            "z_score": round(self.z_score, 6),
        }


@dataclass(frozen=True)
class LabelLexicon:
    """The label-side representation of the whole label space.

    Attributes:
        labels: The label space (every label, including the ones with no train
            example).
        words: Each label's words, strongest first (the mined route: by the
            centroid's coordinate; the expansion route: by the word's rarity).
        routes: Each label's route (``mined`` / ``expanded`` / ``fallback``).
        texts: Each label's **text**: its words joined by a space, or its name
            for the fallback route (a label whose name's tokens are all outside
            the corpus).
        top_k: The K of the mined words.
        expansion_min_prefix: The expansion route's minimum shared prefix.
        n_texts: The corpus's size (``N``, the idf's denominator).
        word_classes: The word classes the build kept (CS-45), or ``None`` when
            it kept every class.
    """

    labels: tuple[str, ...]
    words: Mapping[str, tuple[LexiconWord, ...]] = field(default_factory=dict)
    routes: Mapping[str, str] = field(default_factory=dict)
    texts: Mapping[str, str] = field(default_factory=dict)
    top_k: int = DEFAULT_TOP_K
    expansion_min_prefix: int = DEFAULT_EXPANSION_MIN_PREFIX
    n_texts: int = 0
    word_classes: tuple[str, ...] | None = None

    def label_words(self, label: str) -> tuple[str, ...]:
        """One label's words (empty when the label has none)."""
        return tuple(entry.word for entry in self.words.get(label, ()))

    def label_text(self, label: str) -> str:
        """One label's text (its name when it has no words)."""
        return self.texts.get(label, label)

    def route(self, label: str) -> str:
        """One label's route (``fallback`` when the label is unknown)."""
        return self.routes.get(label, ROUTE_FALLBACK)

    def route_counts(self) -> dict[str, int]:
        """How many labels each route served."""
        counts = dict.fromkeys(ROUTES, 0)
        for label in self.labels:
            counts[self.route(label)] = counts.get(self.route(label), 0) + 1
        return counts

    def labels_without_words(self) -> tuple[str, ...]:
        """The labels with no words at all (the fallback route)."""
        return tuple(label for label in self.labels if not self.words.get(label))


def inverse_document_frequency(df: int, n_texts: int) -> float:
    """The smoothed inverse document frequency of a word.

    ``log((1 + N) / (1 + df))`` — the standard smoothed form, so a word in
    every text still has a finite (zero) weight.

    Args:
        df: The corpus texts that have the word.
        n_texts: The corpus's size.

    Returns:
        The word's idf.
    """
    return math.log((1 + n_texts) / (1 + df))


def pifa_weight(
    *,
    n_with: int,
    df: int,
    n_label: int,
    n_texts: int,
) -> float:
    """The **TF-IDF centroid's** coordinate for one (label, word) pair.

    The centroid of a label's positive instances has, for the word ``w``, the
    coordinate ``tf(L,w) × idf(w)`` — the mean of the instances' TF-IDF values.
    That is the PIFA construction (X-Transformer, KDD 2020) computed from the
    corpus map's own counts.

    Args:
        n_with: The corpus texts that have **both** the word and the label.
        df: The corpus texts that have the word at all.
        n_label: The corpus texts that have the label at all.
        n_texts: The corpus's size.

    Returns:
        The coordinate (the word's weight in the label's representation).

    Raises:
        ValueError: If the counts are not usable (a non-positive label count or
            a word in more texts than the label has).
    """
    if n_label <= 0:
        raise ValueError(f"the label must have at least one train example, got {n_label}")
    if n_with > n_label:
        raise ValueError(
            f"the word cannot be in more texts than the label: n_with={n_with}, n_label={n_label}"
        )
    tf = n_with / n_label
    return tf * inverse_document_frequency(df, n_texts)


def expansion_words(
    label: str,
    vocabulary: Mapping[str, int],
    *,
    min_prefix: int = DEFAULT_EXPANSION_MIN_PREFIX,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[str, ...]:
    """The **expansion route**: a label's name matched against the corpus.

    The label's name's content tokens (the track's own tokenizer) are matched
    against the corpus's vocabulary: a token that is in the vocabulary
    contributes exactly that word; a token that is not contributes the corpus
    words that share a prefix of at least ``min_prefix`` characters with it.
    The result is ordered by **rising document frequency** (the rarest word
    first — the most specific word leads), ties alphabetically.

    **No labels are involved**: the route reads the corpus's words and their
    document frequencies, never a label association and never a gold label.

    Args:
        label: The label's name.
        vocabulary: The corpus's words and their document frequencies.
        min_prefix: The minimum number of shared leading characters.
        stopwords: The tokenizer's stopwords.

    Returns:
        The matched words (possibly empty).
    """
    matched: list[str] = []
    for token in content_tokens(label, stopwords=stopwords):
        if token in vocabulary:
            candidates = [token]
        elif len(token) >= min_prefix:
            prefix = token[:min_prefix]
            candidates = [
                word for word in vocabulary if len(word) >= min_prefix and word.startswith(prefix)
            ]
        else:
            candidates = []
        for word in candidates:
            if word not in matched:
                matched.append(word)
    return tuple(sorted(matched, key=lambda word: (vocabulary[word], word)))


def build_label_lexicon(
    corpus_map: CorpusMap,
    *,
    top_k: int = DEFAULT_TOP_K,
    expansion_min_prefix: int = DEFAULT_EXPANSION_MIN_PREFIX,
    word_classes: Sequence[str] | None = None,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> LabelLexicon:
    """Build the label-side representation from the corpus map.

    The map is built from the **train** split (the command's job), so the words
    are the train corpus's own: no val text, no val label and no test text
    enters them.

    Args:
        corpus_map: The corpus map (the train split's word↔label association).
        top_k: How many mined words a label keeps.
        expansion_min_prefix: The expansion route's minimum shared prefix.
        word_classes: The word classes to **keep** (the track's CS-45): the
            candidate pool loses the words of the other classes before the
            top-K cut, and the expansion route is filtered the same way (a label
            left with no word falls back to its name). ``None`` (the default)
            keeps every class, so every earlier artifact keeps its exact
            content.
        stopwords: The tokenizer's stopwords (the map's own, so the words
            match the tables).

    Returns:
        The lexicon: every label's words, its route and its text.

    Raises:
        ValueError: If ``top_k`` is not positive, on an unknown class, or on an
            empty restriction.
    """
    if top_k <= 0:
        raise ValueError(f"the lexicon needs at least one mined word per label, got {top_k}")
    allowed = _allowed_classes(word_classes)
    classes = corpus_map.word_class_index() if allowed is not None else None

    def _kept(word: str) -> bool:
        """Whether the word's class survives the filter (CS-45)."""
        if classes is None or allowed is None:
            return True
        return classes.class_of(word) in allowed

    labels = tuple(corpus_map.labels)
    vocabulary = {word: rows[0].df for word, rows in corpus_map.per_word.items()}
    words: dict[str, tuple[LexiconWord, ...]] = {}
    routes: dict[str, str] = {}
    texts: dict[str, str] = {}
    for label in labels:
        rows = corpus_map.topic_rows(label)
        n_label = corpus_map.label_counts.get(label, 0)
        if rows and n_label:
            scored = [
                LexiconWord(
                    word=pair.word,
                    weight=pifa_weight(
                        n_with=pair.n_with,
                        df=pair.df,
                        n_label=n_label,
                        n_texts=corpus_map.n_texts,
                    ),
                    n_with=pair.n_with,
                    df=pair.df,
                    log_odds=pair.log_odds,
                    z_score=pair.z_score,
                )
                for pair in rows
                if _kept(pair.word)
            ]
            scored.sort(key=lambda entry: (-entry.weight, entry.word))
            chosen = tuple(scored[:top_k])
            routes[label] = ROUTE_MINED
        else:
            expanded = [
                word
                for word in expansion_words(
                    label,
                    vocabulary,
                    min_prefix=expansion_min_prefix,
                    stopwords=stopwords,
                )
                if _kept(word)
            ]
            chosen = tuple(
                LexiconWord(
                    word=word,
                    weight=inverse_document_frequency(vocabulary[word], corpus_map.n_texts),
                    df=vocabulary[word],
                )
                for word in expanded
            )
            routes[label] = ROUTE_EXPANDED if chosen else ROUTE_FALLBACK
        words[label] = chosen
        texts[label] = " ".join(entry.word for entry in chosen) if chosen else label
    return LabelLexicon(
        labels=labels,
        words=words,
        routes=routes,
        texts=texts,
        top_k=top_k,
        expansion_min_prefix=expansion_min_prefix,
        n_texts=corpus_map.n_texts,
        word_classes=allowed,
    )


def _allowed_classes(word_classes: Sequence[str] | None) -> tuple[str, ...] | None:
    """The classes the build keeps (``None`` = every class).

    Raises:
        ValueError: On an unknown class or an empty restriction.
    """
    if word_classes is None:
        return None
    names = tuple(word_classes)
    unknown = [name for name in names if name not in WORD_CLASSES]
    if unknown:
        raise ValueError(f"Unknown word class(es) {unknown}; expected some of {list(WORD_CLASSES)}")
    if not names:
        raise ValueError("a word-class restriction must keep at least one class")
    return names


def label_lexicon_block(lexicon: LabelLexicon) -> dict[str, object]:
    """The lexicon as the artifact's own block (JSON-friendly).

    The block carries the routes' own definitions (so a reader knows what
    "mined" and "expanded" mean without the plan), the corpus's shape, the
    routes' summary and **every** label's words with the numbers behind them.

    Args:
        lexicon: The lexicon.

    Returns:
        The block (JSON-friendly).
    """
    block: dict[str, object] = {
        "definitions": {
            "weight": (
                "the mined route: tf(L,w) x idf(w), the TF-IDF centroid's coordinate "
                "for the word (the PIFA construction); the expansion route: idf(w) "
                "alone (no labels are involved)"
            ),
            "top_k": "how many mined words a label keeps (the strongest coordinates)",
            "expansion_min_prefix": (
                "the minimum number of shared leading characters in the expansion "
                "route's prefix match"
            ),
            "mined": "the label has train examples: its words are the mined ones",
            "expanded": (
                "the label has no train example: its words come from its NAME "
                "matched against the corpus's vocabulary (no labels involved)"
            ),
            "fallback": (
                "the label has no train example and no matched word: its text is its NAME"
            ),
        },
        "corpus": {
            "split": "train",
            "n_texts": lexicon.n_texts,
            "n_labels": len(lexicon.labels),
        },
        "top_k": lexicon.top_k,
        "expansion_min_prefix": lexicon.expansion_min_prefix,
        "routes": lexicon.route_counts(),
        "labels_without_words": list(lexicon.labels_without_words()),
        "per_label": [
            {
                "label": label,
                "route": lexicon.route(label),
                "n_words": len(lexicon.words.get(label, ())),
                "text": lexicon.label_text(label),
                "words": [entry.to_row() for entry in lexicon.words.get(label, ())],
            }
            for label in lexicon.labels
        ],
    }
    if lexicon.word_classes is not None:
        # the class filter's provenance (the track's CS-45): the key is absent
        # when the build kept every class, so every earlier artifact keeps its
        # exact content.
        block["word_classes"] = list(lexicon.word_classes)
        block["definitions"]["word_classes"] = (
            "the word classes the build kept (CS-33's classes: distinctive = the "
            "word sits in one or two labelled tables, shared = 3-14, generic = at "
            "least a quarter of them); the filter applies to the candidate pool "
            "before the top-K cut and to both routes, and a label left with no "
            "word falls back to its name"
        )
    return block
