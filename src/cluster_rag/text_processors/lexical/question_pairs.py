"""The question-pair relation (the track's CS-57): the questionnaire's own structure.

CS-38 measured the UPR corpus's **label-pair** relation and found it was the
**taxonomy's** (the sibling pairs were lexically closer). OMCT has **no
catalog**, so the taxonomy's claim cannot be the test. This module reads the
relation the reviewed set itself carries: do questions that **co-occur** (they
were judged against the same paragraph — the same document's questionnaire)
share **vocabulary**?

**The statistic.** The questions' vocabulary is their own content words (the
track's tokenizer, the stopwords dropped); the Jaccard of two questions is
``|A ∩ B| / |A ∪ B|``; two questions **co-occur** when at least one reviewed
pair's paragraph carries both. The relation is the co-occurring pairs' mean
Jaccard minus the non-co-occurring pairs', with a **question-level** bootstrap
(the questions are resampled; the statistic is recomputed on the distinct
resampled questions), and the **closest-20** readout (the 20 highest-Jaccard
pairs' co-occurrence share against the base rate).

**The vocabulary-less class** (CS-38's analog): a question whose every content
word also appears in another question's text has no vocabulary of its own.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    content_tokens,
)

# The bootstrap's resamples for the relation's interval (the registration's).
RESAMPLES = 2000

# How many of the closest question pairs the readout lists.
CLOSEST_PAIRS = 20


def _vocabularies(question_texts: Sequence[str]) -> dict[str, frozenset[str]]:
    """Each distinct question's content words."""
    vocabularies: dict[str, frozenset[str]] = {}
    for question in question_texts:
        if question not in vocabularies:
            vocabularies[question] = frozenset(content_tokens(question, DEFAULT_STOPWORDS))
    return vocabularies


def jaccard_matrix(
    questions: Sequence[str], vocabularies: Mapping[str, frozenset[str]]
) -> np.ndarray:
    """The pairwise vocabulary Jaccard of the questions (the diagonal is 0)."""
    words = sorted({word for question in questions for word in vocabularies[question]})
    word_of = {word: index for index, word in enumerate(words)}
    matrix = np.zeros((len(questions), len(words)), dtype=np.float64)
    for row, question in enumerate(questions):
        for word in vocabularies[question]:
            matrix[row, word_of[word]] = 1.0
    sizes = matrix.sum(axis=1)
    intersection = matrix @ matrix.T
    union = sizes[:, None] + sizes[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(jaccard, 0.0)
    return jaccard


def co_occurrence_matrix(
    questions: Sequence[str], pair_questions: Sequence[str], pair_paragraphs: Sequence[str]
) -> np.ndarray:
    """Which question pairs share at least one reviewed paragraph."""
    index = {question: position for position, question in enumerate(questions)}
    by_paragraph: dict[str, list[int]] = {}
    for question, paragraph in zip(pair_questions, pair_paragraphs, strict=True):
        position = index.get(question)
        if position is None:
            continue
        by_paragraph.setdefault(paragraph, []).append(position)
    matrix = np.zeros((len(questions), len(questions)), dtype=bool)
    for positions in by_paragraph.values():
        for left in positions:
            for right in positions:
                if left != right:
                    matrix[left, right] = True
    return matrix


def _upper(matrix: np.ndarray) -> np.ndarray:
    """The strict upper triangle as a flat array (the pair order is fixed)."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _difference(jaccard: np.ndarray, co_occurrence: np.ndarray) -> tuple[float, float]:
    """The co-occurring pairs' mean Jaccard minus the others'."""
    values = _upper(jaccard)
    mask = _upper(co_occurrence)
    if not mask.any() or mask.all():
        return 0.0, float(values.mean()) if values.size else 0.0
    return float(values[mask].mean() - values[~mask].mean()), float(values.mean())


def question_pairs_readout(
    questions: Sequence[str],
    pair_questions: Sequence[str],
    pair_paragraphs: Sequence[str],
    *,
    documents: Sequence[str] | None = None,
    resamples: int = RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """The question-pair relation: the test, the closest pairs and the class.

    Args:
        questions: The question space (the 294 questions).
        pair_questions: Each reviewed pair's question.
        pair_paragraphs: Each reviewed pair's paragraph (the co-occurrence's
            source: two questions co-occur when they share a paragraph).
        documents: Each reviewed pair's document (for the per-document
            co-occurrence readout).
        resamples: The question-level bootstrap's resamples.
        seed: The bootstrap's seed.

    Returns:
        A JSON-serializable block: the relation statistic and its interval,
        the closest-20 readout, the vocabulary-less class and the shape.

    Raises:
        ValueError: If the inputs are ragged, or the request is degenerate.
    """
    n = len(questions)
    if n < 2:
        raise ValueError(f"a question-pair relation needs at least 2 questions, got {n}")
    if len(pair_questions) != len(pair_paragraphs):
        raise ValueError(
            "the pairs' questions and paragraphs must have the same length, got "
            f"{len(pair_questions)} / {len(pair_paragraphs)}"
        )
    if documents is not None and len(documents) != len(pair_questions):
        raise ValueError(f"documents must have {len(pair_questions)} entries")
    if resamples < 1:
        raise ValueError(f"the bootstrap needs at least 1 resample, got {resamples}")

    vocabularies = _vocabularies(questions)
    jaccard = jaccard_matrix(questions, vocabularies)
    co_occurrence = co_occurrence_matrix(questions, pair_questions, pair_paragraphs)
    delta, mean_jaccard = _difference(jaccard, co_occurrence)
    values = _upper(jaccard)
    mask = _upper(co_occurrence)

    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(resamples):
        sampled = np.unique(rng.integers(0, n, size=n))
        if sampled.size < 2:
            continue
        sub_jaccard = jaccard[np.ix_(sampled, sampled)]
        sub_mask = co_occurrence[np.ix_(sampled, sampled)]
        boot.append(_difference(sub_jaccard, sub_mask)[0])
    interval = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot
        else (None, None)
    )

    # the closest pairs: the top of the Jaccard matrix, with their co-occurrence
    order = np.argsort(values)[::-1][:CLOSEST_PAIRS]
    base_rate = float(mask.mean()) if mask.size else None
    closest = []
    for flat in order:
        positions = np.triu_indices(n, k=1)
        left, right = int(positions[0][flat]), int(positions[1][flat])
        closest.append(
            {
                "question_a": questions[left],
                "question_b": questions[right],
                "jaccard": round(float(jaccard[left, right]), 6),
                "co_occur": bool(co_occurrence[left, right]),
            }
        )
    closest_share = sum(1 for row in closest if row["co_occur"]) / len(closest) if closest else None

    # the vocabulary-less class: every content word also in another question's text
    counts: dict[str, int] = {}
    for words in vocabularies.values():
        for word in words:
            counts[word] = counts.get(word, 0) + 1
    vocabulary_less = [
        question
        for question in questions
        if vocabularies[question] and all(counts[word] > 1 for word in vocabularies[question])
    ]

    by_document: dict[str, Any] = {}
    if documents is not None:
        for document in sorted(set(documents)):
            positions = [i for i, value in enumerate(documents) if value == document]
            doc_questions = sorted({pair_questions[i] for i in positions})
            doc_index = {question: index for index, question in enumerate(doc_questions)}
            doc_mask = np.zeros((len(doc_questions), len(doc_questions)), dtype=bool)
            by_paragraph: dict[str, list[int]] = {}
            for i in positions:
                position = doc_index.get(pair_questions[i])
                if position is not None:
                    by_paragraph.setdefault(pair_paragraphs[i], []).append(position)
            for members in by_paragraph.values():
                for left in members:
                    for right in members:
                        if left != right:
                            doc_mask[left, right] = True
            pair_count = len(doc_questions) * (len(doc_questions) - 1) // 2
            by_document[document] = {
                "n_questions": len(doc_questions),
                "n_question_pairs": pair_count,
                "co_occurrence_density": (float(_upper(doc_mask).mean()) if pair_count else None),
            }

    return {
        "unit": (
            "the question pair: the two questions' own vocabularies and their "
            "co-occurrence in the reviewed paragraphs (no catalog on OMCT, so the "
            "taxonomy's claim is replaced by the reviewed set's own structure)"
        ),
        "n_questions": n,
        "n_question_pairs": int(values.size),
        "mean_jaccard": mean_jaccard,
        "relation": {
            "co_occurring_mean": float(values[mask].mean()) if mask.any() else None,
            "other_mean": float(values[~mask].mean()) if (~mask).any() else None,
            "delta": delta,
            "ci_low": interval[0],
            "ci_high": interval[1],
            "resamples": resamples,
            "seed": seed,
            "method": (
                "question-level bootstrap: the questions are resampled with "
                "replacement, the statistic recomputed on the distinct resampled "
                "questions' pairs"
            ),
        },
        "closest": {
            "pairs": closest,
            "co_occurrence_share": closest_share,
            "base_rate": base_rate,
            "n": len(closest),
        },
        "co_occurrence": {
            "density": base_rate,
            "n_co_occurring_pairs": int(mask.sum()),
            "by_document": by_document,
        },
        "vocabulary_less": {
            "n_questions": len(vocabulary_less),
            "share": len(vocabulary_less) / n,
            "questions": vocabulary_less,
        },
        "note": (
            "the readout reads the question texts and the reviewed pairs' "
            "paragraph/question identities only — no pair label, no embedding, no model"
        ),
    }
