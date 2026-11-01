"""The **grammatical classes** (the track's CS-34): the readout on the POS axis.

CS-33 answered the reviewer's part-of-speech question from the corpus map's
own **frequency** classes (how many labels' tables a word sits in). This
module answers the same question on the **grammatical** axis: the tagger's
tags, grouped into coarse classes, crossed with the sign of the map's
log-odds.

**The classes.** The **Penn tagset** grouped into five coarse classes:

- **noun** — NN, NNS, NNP, NNPS (the topic words);
- **verb** — VB, VBD, VBG, VBN, VBP, VBZ (the predicate words — and, in this
  genre, the *register*: "recommend", "implement", "ensure");
- **adjective** — JJ, JJR, JJS;
- **adverb** — RB, RBR, RBS, WRB;
- **other** — everything else (the numbers, the symbols, the foreign words).

**A word's class is the class of its modal tag** — the tag it takes most often
across the corpus. But a word often takes **several** tags ("children" is a
noun and, in another sentence, a verb), so the **per-word tag distribution** is
kept and reported: the readout is reported twice — by the **modal tag** (the
partition: every pair counted once) and by **every tag the word takes** (the
non-partition, so an ambiguous word is visible in both) — and the share of the
words with more than one tag is reported next to them. A single tag per word
would lie; the plan's own check says 16.3 % of the words take more than one.

**The alignment.** The tagger's tokens are the **track's own tokenization**
(the port's contract), so the tags line up with the map's words **by
position**: the map's words are the content tokens of the very same stream.
No join is ever guessed and no word is silently dropped.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, tokenize
from cluster_rag.text_processors.lexical.word_class import (
    SEPARATION_MARGIN,
    ClassReadout,
)

# The coarse classes, in the order the readout reports them.
POS_CLASSES: tuple[str, ...] = ("noun", "verb", "adjective", "adverb", "other")

# The Penn tags' coarse classes. A tag that is not listed here is "other".
TAG_CLASSES: Mapping[str, str] = {
    # nouns
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    # verbs
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    # adjectives
    "JJ": "adjective",
    "JJR": "adjective",
    "JJS": "adjective",
    # adverbs
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
    "WRB": "adverb",
}


def tag_class(tag: str) -> str:
    """A tag's coarse class.

    Args:
        tag: The Penn tag.

    Returns:
        The class (``"other"`` for a tag the grouping does not list).
    """
    return TAG_CLASSES.get(str(tag), "other")


def is_content_token(token: str, *, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> bool:
    """Whether a token is one of the corpus map's words (its content tokens).

    Args:
        token: The token.
        stopwords: The tokenizer's stopwords.

    Returns:
        ``True`` when the map's own tokenizer would keep the token.
    """
    return token not in stopwords and len(token) > 1


def word_tag_counts(
    text: str,
    tagged: Sequence[tuple[str, str]],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, Counter[str]]:
    """The tag counts of the text's **content words**.

    The tags line up with the token stream **by position** (the port's
    contract), so the map's words — the content tokens of that same stream —
    get their tags from the places where they occur. A word that occurs twice
    with two different tags gets both, counted.

    Args:
        text: The text.
        tagged: The text's ``(token, tag)`` pairs, in the token order.
        stopwords: The tokenizer's stopwords.

    Returns:
        The per-word tag counts (only the content words).

    Raises:
        ValueError: If the tags do not line up with the text's token stream.
    """
    tokens = tokenize(text)
    if len(tagged) != len(tokens):
        raise ValueError(
            f"the tagger returned {len(tagged)} tags for {len(tokens)} tokens: "
            "the tagger's tokens must be the track's own tokenization"
        )
    counts: dict[str, Counter[str]] = {}
    for token, (_tagged_token, tag) in zip(tokens, tagged, strict=True):
        if not is_content_token(token, stopwords=stopwords):
            continue
        counts.setdefault(token, Counter())[str(tag)] += 1
    return counts


def merge_distributions(
    distributions: Iterable[Mapping[str, Counter[str]]],
) -> dict[str, Counter[str]]:
    """The corpus's per-word tag counts (the texts' counts added up).

    Args:
        distributions: The per-text distributions.

    Returns:
        The corpus's per-word tag counts.
    """
    merged: dict[str, Counter[str]] = {}
    for distribution in distributions:
        for word, counts in distribution.items():
            merged.setdefault(word, Counter()).update(counts)
    return merged


def modal_tag(counts: Mapping[str, int]) -> str:
    """A word's **modal tag**: the tag it takes most often.

    The tie-break is the tagset's own order (the sorted tag names), so the
    choice is deterministic and never depends on a dictionary's order.

    Args:
        counts: The word's tag counts.

    Returns:
        The modal tag, or an empty string when the word has no tag.
    """
    if not counts:
        return ""
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def modal_classes(
    distribution: Mapping[str, Counter[str]],
) -> dict[str, str]:
    """Each word's class (the class of its modal tag).

    Args:
        distribution: The corpus's per-word tag counts.

    Returns:
        The per-word class.
    """
    return {word: tag_class(modal_tag(counts)) for word, counts in distribution.items()}


def tag_counts_by_class(
    distribution: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    """Each **tag**'s word counts (the corpus's words per tag).

    A word with several tags counts under each of them — the non-partition the
    plan's warning asks for.

    Args:
        distribution: The corpus's per-word tag counts.

    Returns:
        The per-tag word counts.
    """
    counts: dict[str, Counter[str]] = {}
    for _word, tags in distribution.items():
        for tag, count in tags.items():
            counts.setdefault(tag, Counter())[tag] += count
    return counts


def ambiguity_summary(distribution: Mapping[str, Counter[str]]) -> dict[str, object]:
    """The corpus's **ambiguity**: how many words take more than one tag.

    Args:
        distribution: The corpus's per-word tag counts.

    Returns:
        The summary (JSON-friendly): the word count, the ambiguous words' count
        and share, and the ambiguous words themselves with their tag counts.
    """
    words = len(distribution)
    ambiguous = {
        word: dict(sorted(counts.items()))
        for word, counts in sorted(distribution.items())
        if len(counts) > 1
    }
    return {
        "n_words": words,
        "n_words_with_one_tag": words - len(ambiguous),
        "n_words_with_several_tags": len(ambiguous),
        "share_with_several_tags": round(len(ambiguous) / words, 6) if words else 0.0,
        "words_with_several_tags": ambiguous,
        "note": (
            "a word that takes several tags is counted under each of them in the "
            "per-tag readout; the per-class readout counts it once, under the class "
            "of its modal tag"
        ),
    }


def pos_separation_test(
    readout: Mapping[str, ClassReadout],
    *,
    margin: float = SEPARATION_MARGIN,
) -> dict[str, object]:
    """CS-34's registered falsification: do the **nouns** separate from the verbs?

    The test is CS-33's, on the grammatical axis: the **noun** class's mean
    log-odds must exceed the **verb** class's by at least ``margin`` (one
    natural-log unit) **and** the noun class's share of positive pairs must be
    higher. (The nouns are the label-carrying words; the verbs are this
    genre's register.)

    Its own block, with its own keys: CS-33's block is published and is never
    reshaped.

    Args:
        readout: The readout (from :func:`class_readout` with
            :data:`POS_CLASSES`).
        margin: The margin the mean values must differ by.

    Returns:
        The test's block (JSON-friendly), with the ``holds`` flag.

    Raises:
        ValueError: If the readout does not carry the noun and the verb class.
    """
    nouns = readout.get("noun")
    verbs = readout.get("verb")
    if nouns is None or verbs is None:
        raise ValueError(
            f"the readout must carry the 'noun' and the 'verb' class, got {sorted(readout)}"
        )
    if not nouns.n_pairs or not verbs.n_pairs:
        return {
            "holds": False,
            "margin": margin,
            "margin_met": False,
            "verb_share_positive_lower": False,
            "noun_mean": nouns.mean_value,
            "verb_mean": verbs.mean_value,
            "difference": 0.0,
            "note": (
                "one of the two classes holds no pair, so the classes cannot be "
                "separated: the readout is a null"
            ),
        }
    difference = nouns.mean_value - verbs.mean_value
    margin_met = difference >= margin
    share_higher = nouns.share_positive > verbs.share_positive
    return {
        "holds": bool(margin_met and share_higher),
        "margin": margin,
        "margin_met": bool(margin_met),
        "verb_share_positive_lower": bool(share_higher),
        "noun_mean": round(nouns.mean_value, 6),
        "verb_mean": round(verbs.mean_value, 6),
        "difference": round(difference, 6),
        "noun_share_positive": round(nouns.share_positive, 6),
        "verb_share_positive": round(verbs.share_positive, 6),
        "noun_n_pairs": nouns.n_pairs,
        "verb_n_pairs": verbs.n_pairs,
        "value_name": "mean_log_odds",
        "note": (
            "the readout is a null unless the noun class's mean exceeds the verb "
            "class's by at least the margin AND the noun class's share of positive "
            "pairs is higher"
        ),
    }
