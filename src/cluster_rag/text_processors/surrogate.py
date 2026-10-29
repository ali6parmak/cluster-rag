"""The **surrogate attribution** (the track's hit map, mode C).

The hit map's two cheap modes — the topic-side decomposition (mode A) and the
text-side decomposition (mode B) — decompose a *feature*. This mode explains
the thing that actually ranks the labels on UPR: the **decision model's own
probability**. It does so with a **surrogate**: a ridge regression that reads
the text's words and predicts the model's probability for a topic. The
surrogate's coefficients are then a per-topic word attribution: "these words
push the model's confidence on this topic up, those push it down".

**The surrogate is an attribution, not a causal claim.** It says which words
*predict* the model's confidence; it does not say the model *used* them. The
causal half of the map is the ablation (mode D) — and the two are cross-checked
against each other: the surrogate's top words for a topic are looked for inside
the clause the ablation identified as the carrier.

**The fit side is the train split.** The surrogate is fitted on the train
split's texts and on the decision model's **train-side** probabilities (the
CS-9b run's fit side), so no val text and no val label ever enters the fit; the
val split is only used to *measure* the surrogate (its out-of-sample quality)
and for the cross-check against the clause map. The vocabulary is the train
corpus's words (the val corpus's vocabulary never enters either).

Pure core: numpy and the standard library only — no model client, no I/O, no
scikit-learn (the ridge is solved in closed form through one SVD, which the
shared design matrix pays for once).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens


@dataclass(frozen=True)
class SurrogateFit:
    """One fitted surrogate: the vocabulary, the coefficients and the intercepts.

    Attributes:
        vocabulary: The features' names, in the design matrix's column order.
        coefficients: The per-topic coefficients, shape ``(n_topics, n_features)``.
        intercepts: The per-topic intercepts, shape ``(n_topics,)``.
        explained: The per-topic in-sample R², shape ``(n_topics,)``.
        ridge_lambda: The ridge penalty the fit used.
    """

    vocabulary: tuple[str, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray
    explained: np.ndarray
    ridge_lambda: float


def build_vocabulary(
    texts: Sequence[str],
    *,
    min_document_frequency: int = 3,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[str, ...]:
    """The corpus's content words, kept when they are common enough.

    The vocabulary is the **fit side's** corpus (the train split), so a word
    that only appears in val never becomes a feature.

    Args:
        texts: The fit side's texts.
        min_document_frequency: The smallest number of texts a word must appear
            in to become a feature (a word in one text only cannot be estimated).
        stopwords: The stopwords of the tokenizer.

    Returns:
        The vocabulary, sorted (so the design matrix's columns — and therefore
        the fit — are the same from one run to the next).
    """
    counts: dict[str, int] = {}
    for text in texts:
        for word in set(content_tokens(text, stopwords)):
            counts[word] = counts.get(word, 0) + 1
    kept = [word for word, count in counts.items() if count >= min_document_frequency]
    return tuple(sorted(kept))


def build_design_matrix(
    texts: Sequence[str],
    vocabulary: Sequence[str],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> np.ndarray:
    """The design matrix: one row per text, one column per vocabulary word.

    The feature is the word's **presence** (1 or 0), not its count: the map
    asks "does this word carry the score", and a count would make a repeated
    boilerplate word look like a topic word.

    Args:
        texts: The texts (the rows).
        vocabulary: The features (the columns).
        stopwords: The stopwords of the tokenizer.

    Returns:
        The float64 design matrix, shape ``(n_texts, n_features)``.
    """
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    index = {word: position for position, word in enumerate(vocabulary)}
    for row, text in enumerate(texts):
        for word in set(content_tokens(text, stopwords)):
            position = index.get(word)
            if position is not None:
                matrix[row, position] = 1.0
    return matrix


def fit_ridge_surrogate(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float = 1.0,
) -> SurrogateFit:
    """Fit a per-topic ridge surrogate of the model's probability.

    One shared design matrix, one coefficient vector per topic. The ridge is
    solved in closed form through **one** SVD of the (centred) design matrix,
    which the per-topic solves then share — the whole fit is a few seconds.

    Args:
        design: The design matrix ``(n_texts, n_features)``.
        targets: The model's probabilities ``(n_texts, n_topics)``.
        ridge_lambda: The ridge penalty (the larger, the more the coefficients
            are shrunk towards zero).

    Returns:
        The fit (without the vocabulary's names — the caller owns them).

    Raises:
        ValueError: If the shapes do not match or the penalty is negative.
    """
    matrix = np.asarray(design, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if matrix.ndim != 2 or values.ndim != 2:
        raise ValueError("the design matrix and the targets must be two-dimensional")
    if matrix.shape[0] != values.shape[0]:
        raise ValueError(
            f"the design matrix has {matrix.shape[0]} rows, the targets {values.shape[0]}"
        )
    if ridge_lambda < 0:
        raise ValueError(f"the ridge penalty must be non-negative, got {ridge_lambda}")
    n_rows, n_features = matrix.shape
    n_topics = values.shape[1]
    if n_rows == 0 or n_features == 0:
        return SurrogateFit(
            vocabulary=(),
            coefficients=np.zeros((n_topics, n_features)),
            intercepts=np.zeros(n_topics),
            explained=np.zeros(n_topics),
            ridge_lambda=ridge_lambda,
        )
    feature_mean = matrix.mean(axis=0)
    target_mean = values.mean(axis=0)
    centred = matrix - feature_mean
    centred_targets = values - target_mean
    # the economy SVD of the centred design: the topics share it
    left, singular, right = np.linalg.svd(centred, full_matrices=False)
    factors = singular / (singular**2 + ridge_lambda)
    projected = factors[:, None] * (left.T @ centred_targets)
    # (n_features, n_topics) -> (n_topics, n_features): one row per topic
    coefficients = (right.T @ projected).T
    intercepts = target_mean - coefficients @ feature_mean
    predictions = matrix @ coefficients.T + intercepts
    residual = values - predictions
    total = values - target_mean
    denominator = (total**2).sum(axis=0)
    explained = np.where(
        denominator > 0,
        1.0 - (residual**2).sum(axis=0) / np.where(denominator > 0, denominator, 1.0),
        0.0,
    )
    return SurrogateFit(
        vocabulary=(),
        coefficients=coefficients,
        intercepts=intercepts,
        explained=explained,
        ridge_lambda=ridge_lambda,
    )


def surrogate_predictions(fit: SurrogateFit, design: np.ndarray) -> np.ndarray:
    """The surrogate's predicted probabilities, shape ``(n_texts, n_topics)``."""
    matrix = np.asarray(design, dtype=np.float64)
    return matrix @ fit.coefficients.T + fit.intercepts


def surrogate_quality(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, object]:
    """How well the surrogate tracks the model (the correlation and the R²).

    The per-topic Pearson correlation and R² against the model's own
    probabilities, plus their means — the numbers that say whether the
    surrogate is worth reading its coefficients off.

    Args:
        predictions: The surrogate's predictions.
        targets: The model's probabilities.

    Returns:
        The quality block (JSON-friendly).
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError(f"the predictions {predicted.shape} must match the targets {actual.shape}")
    correlations: list[float] = []
    r_squared: list[float] = []
    for column in range(actual.shape[1]):
        left = predicted[:, column]
        right = actual[:, column]
        if left.std() == 0 or right.std() == 0:
            correlations.append(0.0)
            r_squared.append(0.0)
            continue
        correlations.append(float(np.corrcoef(left, right)[0, 1]))
        denominator = float(((right - right.mean()) ** 2).sum())
        residual = float(((right - left) ** 2).sum())
        r_squared.append(1.0 - residual / denominator if denominator > 0 else 0.0)
    return {
        "per_topic_correlation": [round(value, 6) for value in correlations],
        "per_topic_r2": [round(value, 6) for value in r_squared],
        "mean_correlation": round(float(np.mean(correlations)), 6) if correlations else None,
        "mean_r2": round(float(np.mean(r_squared)), 6) if r_squared else None,
        "note": (
            "the surrogate's out-of-sample quality: the per-topic Pearson "
            "correlation between its prediction and the model's own "
            "probability, and the R²"
        ),
    }


def topic_attribution(
    fit: SurrogateFit,
    vocabulary: Sequence[str],
    topics: Sequence[str],
    *,
    top_words: int = 15,
) -> list[dict[str, object]]:
    """Each topic's word attribution: the words that push the model up and down.

    Args:
        fit: The fitted surrogate.
        vocabulary: The features' names (the design matrix's columns).
        topics: The topics (the label space, in the coefficients' row order).
        top_words: How many words each side carries.

    Returns:
        One row per topic: the strongest positive words, the strongest negative
        words, the intercept and the in-sample R².
    """
    rows: list[dict[str, object]] = []
    for index, topic in enumerate(topics):
        coefficients = fit.coefficients[index]
        order = np.argsort(-coefficients, kind="stable")
        positive = [
            {
                "word": vocabulary[position],
                "coefficient": round(float(coefficients[position]), 6),
            }
            for position in order[:top_words]
            if coefficients[position] > 0
        ]
        reverse = np.argsort(coefficients, kind="stable")
        negative = [
            {
                "word": vocabulary[position],
                "coefficient": round(float(coefficients[position]), 6),
            }
            for position in reverse[:top_words]
            if coefficients[position] < 0
        ]
        rows.append(
            {
                "topic": topic,
                "intercept": round(float(fit.intercepts[index]), 6),
                "in_sample_r2": round(float(fit.explained[index]), 6),
                "top_positive_words": positive,
                "top_negative_words": negative,
            }
        )
    return rows


def name_word_share(
    attribution: Mapping[str, object],
    topic: str,
    topics: Sequence[str],
    *,
    top_words: int = 15,
) -> float | None:
    """The share of a topic's attribution words that are the topic's own words.

    The yardstick of the topic-faithfulness check: a surrogate that reads the
    topic puts the topic's **own name words** at the top of that topic's
    attribution.

    Args:
        attribution: One topic's attribution row (from :func:`topic_attribution`).
        topic: The topic the row belongs to.
        topics: The topic space.
        top_words: How many words the row carries on each side.

    Returns:
        The share, or ``None`` when the topic is not in the space.
    """
    if topic not in topics:
        return None
    names = set(content_tokens(topic))
    if not names:
        return None
    words = list(attribution.get("top_positive_words") or [])[:top_words]
    if not words:
        return None
    hits = sum(1 for word in words if str(word["word"]) in names)
    return round(hits / len(words), 6)


def agreement_with_carrier(
    attribution: Mapping[str, Sequence[Mapping[str, object]]],
    carriers: Mapping[tuple[int, int], str],
    *,
    top_words: int = 15,
    text_words: Mapping[int, Sequence[str]] | None = None,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, object]:
    """The surrogate's words against the **causal** map's carrier clause.

    The cross-check between the cheap attribution and the expensive ablation:
    for every (text, topic) pair whose carrier clause the ablation identified,
    where do the surrogate's top words sit? Three shares are reported:

    - **inside the carrier** — the share of the topic's top words that the
      carrier clause contains;
    - **elsewhere in the text** — the share of the top words that the text
      contains but the carrier clause does not;
    - **absent** — the rest (the top words the text does not contain at all).

    The **chance level** is the carrier clause's own share of the text's
    content words: if the top words were scattered at random over the text,
    that is where they would land. The check passes when the observed inside
    share beats the chance level.
    """
    inside: list[float] = []
    elsewhere: list[float] = []
    absent: list[float] = []
    conditional: list[float] = []
    chance: list[float] = []
    checked = 0
    for (text_row, topic_index), clause in carriers.items():
        if topic_index >= len(attribution):
            continue
        words = [
            str(word["word"])
            for word in (attribution[topic_index].get("top_positive_words") or [])[:top_words]
        ]
        if not words:
            continue
        clause_words = set(content_tokens(clause, stopwords))
        row_words = (
            set(content_tokens(text_words[text_row], stopwords))
            if text_words is not None and text_row in text_words
            else set()
        )
        in_clause = [word for word in words if word in clause_words]
        in_text = [word for word in words if word in row_words]
        in_text_elsewhere = [word for word in in_text if word not in clause_words]
        checked += 1
        inside.append(len(in_clause) / len(words))
        elsewhere.append(len(in_text_elsewhere) / len(words))
        absent.append((len(words) - len(in_text)) / len(words))
        if in_text:
            conditional.append(len(in_clause) / len(in_text))
        if row_words:
            chance.append(len(clause_words & row_words) / len(row_words))
    return {
        "n_pairs": checked,
        "share_inside_the_carrier": round(float(np.mean(inside)), 6) if inside else None,
        "share_elsewhere_in_the_text": round(float(np.mean(elsewhere)), 6) if elsewhere else None,
        "share_absent_from_the_text": round(float(np.mean(absent)), 6) if absent else None,
        "share_inside_among_the_words_in_the_text": (
            round(float(np.mean(conditional)), 6) if conditional else None
        ),
        "chance_level": round(float(np.mean(chance)), 6) if chance else None,
        "beats_the_chance_level": (
            bool(float(np.mean(conditional)) > float(np.mean(chance)))
            if conditional and chance
            else None
        ),
        "note": (
            "the surrogate's top words for a topic, looked for inside the "
            "clause the ablation identified as the carrier (the causal map) "
            "and elsewhere in the text: the chance level is the carrier "
            "clause's own share of the text's words, so a higher inside-share "
            "than the chance level means the two methods agree about where "
            "the signal is"
        ),
    }
