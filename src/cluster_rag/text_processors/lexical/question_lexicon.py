"""The label side's own words (the track's CS-55): the question's vocabulary as a channel.

CS-35/45's label-side channel, ported to the pair unit. OMCT has **no label
catalog**: the only text a label has is the **question's own text**, so the
label's "name" and its "description" are the same object and the channel is the
question's own vocabulary read against the paragraph — no pair label, no map,
no embedding.

**The arms** (all label-side only):

- ``all`` — the count of the question's content words present in the paragraph
  (the primary arm: the "name" analog is the whole question text);
- ``idf`` — the same count, weighted by the word's label-side idf
  ``log(n_questions / the number of questions whose text holds it)``;
- ``topic`` — the count over the question's **distinctive** words only (1-2
  questions' texts: CS-45's filter, ported to the label side);
- ``register`` — the count over the question's **generic** words only (in at
  least a quarter of the questions' texts: the questionnaire's register).

**The register, stated before measuring.** On UPR the register was a
*corpus-side* property (the words in many labels' map tables); here it is a
**label-side** one (the words in many questions' texts). The difference is
stated because it is where the port could fail: the questionnaire's register
("question", "choices", "answer", "torture") is the same text for every
question, so it may carry no question-specific signal at all.

**The blind counterpart.** The same channel with the **questions permuted**
across the pairs (the seeded permutation CS-49 uses), scored pair by pair: the
blind channel's gap is the label-driven part.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    content_tokens,
)
from cluster_rag.text_processors.lexical.word_class import (
    WORD_CLASSES,
    classify_word,
    generic_threshold,
)

# The channel's arms, in the pre-registration's order (the primary is named
# explicitly there; the order is the tie-break).
LEXICON_ARMS: tuple[str, ...] = ("all", "idf", "topic", "register")

# The arm fixed a priori by the pre-registration (the "name" analog is the
# whole question text, so the primary reads every word).
PRIMARY_LEXICON_ARM = "all"

# The registered bar: the arm's within-document AUC minus the reference's must
# be at least this for "it is a channel" / "it is a lever".
LEXICON_BAR = 0.05


def question_vocabularies(
    question_texts: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, frozenset[str]]:
    """Each distinct question's content words (the label side's own vocabulary).

    Args:
        question_texts: The questions (duplicates collapse; the order is the
            question space's first-occurrence order).
        stopwords: The tokenizer's stopwords (the same as the map's).

    Returns:
        The question -> its content words (a frozenset, so no order enters a
        score path).
    """
    vocabularies: dict[str, frozenset[str]] = {}
    for question in question_texts:
        if question not in vocabularies:
            vocabularies[question] = frozenset(content_tokens(question, stopwords))
    return vocabularies


def word_question_counts(vocabularies: Mapping[str, frozenset[str]]) -> dict[str, int]:
    """How many questions' texts each label-side word sits in."""
    counts: Counter[str] = Counter()
    for words in vocabularies.values():
        counts.update(words)
    return dict(counts)


def classify_question_words(counts: Mapping[str, int], n_questions: int) -> dict[str, str]:
    """Each label-side word's class (CS-33's rule, read on the label side).

    ``distinctive`` 1-2 questions, ``shared`` the middle, ``generic`` at least a
    quarter of the questions' texts (the questionnaire's register).
    """
    return {word: classify_word(count, n_questions) for word, count in counts.items()}


def label_side_idf(counts: Mapping[str, int], n_questions: int) -> dict[str, float]:
    """The label side's idf: ``log(n_questions / the word's question count)``."""
    idf: dict[str, float] = {}
    for word, count in counts.items():
        idf[word] = float(np.log(n_questions / count)) if count > 0 else 0.0
    return idf


def _kept_classes(arm: str) -> tuple[str, ...] | None:
    """The classes an arm keeps (``None`` keeps every class)."""
    if arm == "topic":
        return ("distinctive",)
    if arm == "register":
        return ("generic",)
    return None


def _word_weight(arm: str, word: str, idf: Mapping[str, float]) -> float:
    """The weight an arm gives one label-side word."""
    return idf.get(word, 0.0) if arm == "idf" else 1.0


def lexicon_grid(
    paragraph_texts: Sequence[str],
    questions: Sequence[str],
    vocabularies: Mapping[str, frozenset[str]],
    word_classes: Mapping[str, str],
    *,
    arm: str = PRIMARY_LEXICON_ARM,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    idf: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Score every (paragraph, question) cell by the question's own words.

    Args:
        paragraph_texts: The grid's paragraph rows.
        questions: The grid's question columns.
        vocabularies: The question -> its content words.
        word_classes: The label-side words' classes (the arms `topic` and
            `register` filter on them).
        arm: Which of :data:`LEXICON_ARMS` to apply.
        stopwords: The tokenizer's stopwords.
        idf: The label-side idf (needed by the ``idf`` arm).

    Returns:
        A ``(n_paragraphs, n_questions)`` float64 matrix, higher = more
        likely. Every cell is scored: the channel has no empty table.

    Raises:
        ValueError: On an unknown arm.
    """
    if arm not in LEXICON_ARMS:
        raise ValueError(f"Unknown lexicon arm {arm!r}; expected one of {LEXICON_ARMS}")
    kept = _kept_classes(arm)
    weights = idf or {}
    token_index: dict[str, list[int]] = {}
    for row, text in enumerate(paragraph_texts):
        # sorted tokens: the cell's sum order must not depend on a hash
        for token in sorted(set(content_tokens(text, stopwords))):
            token_index.setdefault(token, []).append(row)
    grid = np.zeros((len(paragraph_texts), len(questions)), dtype=np.float64)
    for column, question in enumerate(questions):
        for word in sorted(vocabularies.get(question, frozenset())):
            if kept is not None and word_classes.get(word) not in kept:
                continue
            rows = token_index.get(word)
            if not rows:
                continue
            grid[np.asarray(rows), column] += _word_weight(arm, word, weights)
    return grid


def lexicon_pair_scores(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    vocabularies: Mapping[str, frozenset[str]],
    word_classes: Mapping[str, str],
    *,
    arm: str = PRIMARY_LEXICON_ARM,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    idf: Mapping[str, float] | None = None,
) -> np.ndarray:
    """The channel's score for each pair (the label-blind counterpart's form).

    The grid version reads a fixed (paragraph, question) pair; this one reads
    the question **each pair carries**, which is what the blind counterpart
    needs (the question permuted across the pairs, the paragraph in place).

    Args:
        paragraph_texts: Each pair's paragraph.
        question_texts: Each pair's question (the permuted one for the blind).
        vocabularies: The question -> its content words.
        word_classes: The label-side words' classes.
        arm: Which of :data:`LEXICON_ARMS` to apply.
        stopwords: The tokenizer's stopwords.
        idf: The label-side idf.

    Returns:
        One score per pair, in the pairs' order.
    """
    if arm not in LEXICON_ARMS:
        raise ValueError(f"Unknown lexicon arm {arm!r}; expected one of {LEXICON_ARMS}")
    kept = _kept_classes(arm)
    weights = idf or {}
    scores = np.zeros(len(question_texts), dtype=np.float64)
    pairs = zip(paragraph_texts, question_texts, strict=True)
    for index, (paragraph, question) in enumerate(pairs):
        present = set(content_tokens(paragraph, stopwords))
        total = 0.0
        for word in sorted(vocabularies.get(question, frozenset())):
            if kept is not None and word_classes.get(word) not in kept:
                continue
            if word in present:
                total += _word_weight(arm, word, weights)
        scores[index] = total
    return scores


@dataclass(frozen=True)
class QuestionLexicon:
    """The label side's vocabulary and its classes (the readout's raw material).

    Attributes:
        questions: The question space.
        vocabularies: The question -> its content words.
        counts: The label-side word -> the number of questions' texts holding it.
        classes: The label-side word -> its class.
        idf: The label-side word -> its idf.
        n_questions: The label space's size.
        n_words: The distinct label-side words.
    """

    questions: tuple[str, ...]
    vocabularies: Mapping[str, frozenset[str]]
    counts: Mapping[str, int]
    classes: Mapping[str, str]
    idf: Mapping[str, float]
    n_questions: int
    n_words: int


def build_question_lexicon(
    question_texts: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> QuestionLexicon:
    """Build the label side's vocabulary, its classes and its idf."""
    questions = tuple(dict.fromkeys(question_texts))
    vocabularies = question_vocabularies(questions, stopwords=stopwords)
    counts = word_question_counts(vocabularies)
    classes = classify_question_words(counts, len(questions))
    idf = label_side_idf(counts, len(questions))
    return QuestionLexicon(
        questions=questions,
        vocabularies=vocabularies,
        counts=counts,
        classes=classes,
        idf=idf,
        n_questions=len(questions),
        n_words=len(counts),
    )


def lexicon_readout(
    lexicon: QuestionLexicon,
    *,
    pair_paragraphs: Sequence[str],
    pair_questions: Sequence[str],
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, Any]:
    """The label side's shape (CS-55): the vocabulary, its classes, its coverage."""
    sizes = sorted(len(words) for words in lexicon.vocabularies.values())
    per_class: dict[str, dict[str, int]] = {name: {"words": 0, "pairs": 0} for name in WORD_CLASSES}
    for name in lexicon.classes.values():
        if name in per_class:
            per_class[name]["words"] += 1
    for words in lexicon.vocabularies.values():
        for word in words:
            name = lexicon.classes.get(word, "")
            if name in per_class:
                per_class[name]["pairs"] += 1
    if len(pair_paragraphs) != len(pair_questions):
        raise ValueError(
            "the pairs' paragraphs and questions must have the same length, got "
            f"{len(pair_paragraphs)} / {len(pair_questions)}"
        )
    register = generic_threshold(lexicon.n_questions)
    top = sorted(lexicon.counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    overlap = []
    for paragraph, question in zip(pair_paragraphs, pair_questions, strict=True):
        present = set(content_tokens(paragraph, stopwords))
        words = lexicon.vocabularies.get(question, frozenset())
        overlap.append(sum(1 for word in words if word in present))
    return {
        "unit": (
            "the label side is the QUESTION TEXT only (OMCT has no catalog): the "
            "label's 'name' and its 'description' are the same object"
        ),
        "register": (
            "a label-side word in at least a quarter of the questions' texts "
            "(the questionnaire's register); the UPR register was corpus-side, "
            "this one is label-side"
        ),
        "n_questions": lexicon.n_questions,
        "n_words": lexicon.n_words,
        "vocabulary_size": {
            "min": sizes[0] if sizes else 0,
            "median": float(np.median(sizes)) if sizes else None,
            "max": sizes[-1] if sizes else 0,
        },
        "classes": per_class,
        "thresholds": {
            "quarter": 0.25,
            "generic_min_questions": register,
        },
        "register_top": [{"word": word, "questions": count} for word, count in top],
        "paragraph_overlap": {
            "mean": float(np.mean(overlap)) if overlap else None,
            "median": float(np.median(overlap)) if overlap else None,
            "zero_share": (sum(1 for value in overlap if value == 0) / len(overlap))
            if overlap
            else None,
        },
        "note": (
            "the channel reads the question texts and the paragraph texts only — "
            "no pair label, no map, no embedding; the blind counterpart permutes "
            "the questions across the pairs"
        ),
    }
