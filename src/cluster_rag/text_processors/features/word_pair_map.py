"""The global word-pair table (the track's CS-59).

The per-question map (CS-48) is a **fixed questionnaire**: a question gets a
table only after a positive judged pair, so a new question is silent. This
module moves the unit off the question: **one global table** over the words a
paragraph and a question **share**, learned from every judged pair, with no
question in the key.

For a pair, the two sides are tokenized with the track's content-word
tokenizer and the pair's **shared words** are the set intersection. The
table's row is the shared word ``w`` — the diagonal ``(w, w)`` of the
paragraph-word x question-word matrix — with the same 2x2 statistics and the
same informative-prior z-score as the per-question map (CS-50)::

    df_w           the fit pairs whose paragraph and question both contain w
    n_with_w       those labelled 1
    paragraph_df_w the fit pairs whose paragraph contains w (the prior's df)

Only words with ``n_with_w >= 1`` become rows (the "evidence-for" rule of
CS-48's builder). A pair's score is the sum of its shared rows' weights under
the requested arm; a pair with **no** shared row carries the floor.

Two fold assemblies are registered (CS-59's plan, section 7.11):

- :func:`cross_fit_word_pair_seen` — the round's **pair-level** cross-fit:
  each fold's table from the other folds' pairs, every judged pair scored by
  its own fold's table (the "all questions seen" arm);
- :func:`cross_fit_word_pair_questions` — the **question-level** cross-fit:
  each fold's table from the other folds' *questions'* pairs, so a held-out
  question's pairs are scored by a table that never read any of them (the
  "held-out questions" arm).

The label-blind counterpart rebuilds the table from the **questions permuted
across the pairs** and scores the **real** grid, so an arm that beats it is
reading the real question-word association, not the table's general word
statistics.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.corpus_map import log_odds
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.lexicon import pair_measures, prior_total
from cluster_rag.text_processors.lexical.question_map_score import (
    ARM_SPECS,
    FILLS,
    FLOOR_GAP,
    SCORING_FUNCTIONS,
)

# The registered primary arm (CS-59): the same z-scored "evidence for" sum as
# CS-50/CS-56, fixed a priori, never selected on a run's numbers.
WORD_PAIR_PRIMARY = "sum_zscore_positive"

# The arms the block scores as the measured spread (the primary is not
# selected on them).
WORD_PAIR_SCORINGS: tuple[str, ...] = (WORD_PAIR_PRIMARY, "support_count")


@dataclass(frozen=True)
class WordPairRow:
    """One shared word's global statistics (the table's row).

    Attributes mirror the per-question map's :class:`~.question_map.QuestionPair`
    so the two tables are read with the same words.
    """

    word: str
    n_with: int
    df: int
    paragraph_df: int
    lift: float
    log_odds: float
    log_odds_informative: float
    standard_error: float
    z_score: float
    g2: float
    p_value: float
    log_odds_conditional: float

    def to_row(self) -> dict[str, object]:
        """The row's JSON shape (the artifact's table entry)."""
        return {
            "word": self.word,
            "n_with": self.n_with,
            "df": self.df,
            "paragraph_df": self.paragraph_df,
            "lift": self.lift,
            "log_odds": self.log_odds,
            "log_odds_informative": self.log_odds_informative,
            "standard_error": self.standard_error,
            "z_score": self.z_score,
            "g2": self.g2,
            "p_value": self.p_value,
            "log_odds_conditional": self.log_odds_conditional,
        }


@dataclass(frozen=True)
class WordPairMap:
    """The global table: one row per shared word with a positive fit pair."""

    n_pairs: int
    n_positive: int
    n_negative: int
    n_shared_words: int
    rows: Mapping[str, WordPairRow]

    @property
    def n_rows(self) -> int:
        """The number of stored rows (words with at least one positive pair)."""
        return len(self.rows)

    @property
    def n_single_rows(self) -> int:
        """The rows resting on a single judged pair (the CS-54 lesson)."""
        return sum(1 for row in self.rows.values() if row.df == 1)

    def stats(self) -> dict[str, Any]:
        """The table's shape and its single-observation share."""
        dfs = [row.df for row in self.rows.values()]
        return {
            "n_pairs": int(self.n_pairs),
            "n_positive": int(self.n_positive),
            "n_negative": int(self.n_negative),
            "n_shared_words": int(self.n_shared_words),
            "n_rows": int(self.n_rows),
            "n_single_rows": int(self.n_single_rows),
            "single_observation_share": (
                float(self.n_single_rows / self.n_rows) if self.n_rows else None
            ),
            "median_df": float(np.median(dfs)) if dfs else None,
            "max_df": int(max(dfs)) if dfs else None,
            "note": (
                "one row per shared word with at least one positive fit pair; df is "
                "the fit pairs whose paragraph and question both contain the word; "
                "paragraph_df is the fit pairs whose paragraph contains it (the prior)"
            ),
        }


def _shared_words(
    paragraph: str,
    question: str,
    stopwords: frozenset[str],
) -> frozenset[str]:
    """The pair's shared content words (the set intersection)."""
    return frozenset(content_tokens(paragraph, stopwords)) & frozenset(
        content_tokens(question, stopwords)
    )


def build_word_pair_map(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    min_df: int = 1,
) -> WordPairMap:
    """Build the global table from the **fit** pairs.

    Args:
        paragraph_texts: Each fit pair's paragraph.
        question_texts: Each fit pair's question text.
        labels: Each fit pair's 0/1 label, in the same order.
        stopwords: The track's content-word stopwords (the map's own).
        min_df: The smallest ``df`` a row may have (the default keeps every
            row; ``2`` drops the single-observation rows, the CS-54 diagnostic).

    Returns:
        The global table.

    Raises:
        ValueError: If the inputs have different lengths, or a label is not 0/1.
    """
    if not (len(paragraph_texts) == len(question_texts) == len(labels)):
        raise ValueError(
            "the fit pairs' paragraphs, questions and labels must have the same "
            f"length, got {len(paragraph_texts)} / {len(question_texts)} / {len(labels)}"
        )
    bad = sorted({int(label) for label in labels if int(label) not in (0, 1)})
    if bad:
        raise ValueError(f"the pair labels must be 0/1, got {bad}")

    n_pairs = len(labels)
    n_positive = sum(1 for label in labels if int(label) == 1)
    paragraph_sets = [frozenset(content_tokens(text, stopwords)) for text in paragraph_texts]
    question_sets = [frozenset(content_tokens(text, stopwords)) for text in question_texts]
    df: Counter[str] = Counter()
    n_with: Counter[str] = Counter()
    shared_words: set[str] = set()
    for paragraph_tokens, question_tokens, label in zip(
        paragraph_sets, question_sets, labels, strict=True
    ):
        shared = paragraph_tokens & question_tokens
        shared_words.update(shared)
        df.update(shared)
        if int(label) == 1:
            n_with.update(shared)
    paragraph_df: Counter[str] = Counter()
    for paragraph_tokens in paragraph_sets:
        paragraph_df.update(paragraph_tokens)

    # The estimator's map-level constant: the reviewed set's own (pair, word)
    # incidences over the **paragraph** vocabulary, the same prior_total call
    # as the per-question map (CS-50).
    alpha0 = prior_total(paragraph_df.values())
    rows: dict[str, WordPairRow] = {}
    for word, positives in n_with.items():
        if positives <= 0:
            continue
        word_df = df[word]
        prior_df = paragraph_df.get(word, word_df)
        measures = pair_measures(
            positives,
            word_df,
            n_positive,
            n_pairs,
            alpha0=alpha0,
            prior_df=prior_df,
        )
        if word_df < min_df:
            continue
        rows[word] = WordPairRow(
            word=word,
            n_with=int(positives),
            df=int(word_df),
            paragraph_df=int(prior_df),
            lift=((positives / word_df) / (n_positive / n_pairs) if n_pairs and word_df else 0.0),
            log_odds=log_odds(n_with=positives, df=word_df, n_label=n_positive, n_texts=n_pairs),
            log_odds_informative=measures.log_odds_informative,
            standard_error=measures.standard_error,
            z_score=measures.z_score,
            g2=measures.g2,
            p_value=measures.p_value,
            log_odds_conditional=measures.log_odds_conditional,
        )
    return WordPairMap(
        n_pairs=n_pairs,
        n_positive=n_positive,
        n_negative=n_pairs - n_positive,
        n_shared_words=len(shared_words),
        rows=rows,
    )


def _cell_value(rows: Sequence[WordPairRow], scoring: str) -> float:
    """One cell's raw score from its known shared rows (the CS-30 arms)."""
    aggregation, measure, positive_only = ARM_SPECS[scoring]
    if aggregation == "count":
        return float(len(rows))
    if not rows:
        return 0.0
    values = [row.z_score if measure == "z_score" else row.log_odds for row in rows]
    if aggregation == "max":
        return float(max(values))
    if positive_only:
        values = [value for value in values if value > 0.0]
    return float(sum(values))


def score_word_pair_grid(
    paragraphs: Sequence[str],
    questions: Sequence[str],
    table: WordPairMap,
    *,
    scoring: str = WORD_PAIR_PRIMARY,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    fills: Sequence[str] = ("floor",),
) -> dict[str, Any]:
    """Score every (paragraph, question) cell from the global table.

    Args:
        paragraphs: The grid's paragraph rows.
        questions: The grid's question columns.
        table: The global table (the full one, or one fold's).
        scoring: Which CS-30 arm to apply (the registered primary is
            :data:`WORD_PAIR_PRIMARY`).
        stopwords: The tokenizer's stopwords.
        fills: The empty-cell fills to return beside the raw matrix: ``floor``
            (the row's smallest seen score minus one — the registered rule),
            ``neutral`` (the row's median) or ``max`` (the row's maximum).

    Returns:
        A dict with ``scores`` (one matrix per requested fill), ``raw`` (the
        unfilled matrix), the per-cell ``floored`` flag, ``n_shared`` (any
        shared content word), ``n_known`` (a shared word with a table row) and
        ``n_single`` (the known rows whose ``df`` is 1).

    Raises:
        ValueError: On an unknown scoring function or an unknown fill.
    """
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring function {scoring!r}; expected one of {SCORING_FUNCTIONS}"
        )
    unknown = [fill for fill in fills if fill not in FILLS]
    if unknown:
        raise ValueError(f"Unknown empty-cell fill(s) {unknown}; expected one of {FILLS}")

    paragraph_sets = [frozenset(content_tokens(text, stopwords)) for text in paragraphs]
    question_sets = [frozenset(content_tokens(text, stopwords)) for text in questions]
    n_paragraphs = len(paragraph_sets)
    n_questions = len(question_sets)
    raw = np.zeros((n_paragraphs, n_questions), dtype=np.float64)
    n_shared = np.zeros((n_paragraphs, n_questions), dtype=np.int64)
    n_known = np.zeros((n_paragraphs, n_questions), dtype=np.int64)
    n_single = np.zeros((n_paragraphs, n_questions), dtype=np.int64)
    rows = table.rows
    for row_index, paragraph_tokens in enumerate(paragraph_sets):
        for column_index, question_tokens in enumerate(question_sets):
            shared = paragraph_tokens & question_tokens
            n_shared[row_index, column_index] = len(shared)
            # Sorted, not set order: the floating-point summation order must
            # reproduce field for field (the CS-49 fix, kept here).
            known = [rows[word] for word in sorted(shared) if word in rows]
            n_known[row_index, column_index] = len(known)
            n_single[row_index, column_index] = sum(1 for row in known if row.df == 1)
            raw[row_index, column_index] = _cell_value(known, scoring)

    floored = n_known == 0
    scores: dict[str, np.ndarray] = {}
    for fill in fills:
        matrix = raw.copy()
        if floored.any():
            for row_index in range(n_paragraphs):
                seen = ~floored[row_index]
                if not seen.any():
                    value = 0.0 if fill == "neutral" else -FLOOR_GAP
                elif fill == "neutral":
                    value = float(np.median(matrix[row_index, seen]))
                elif fill == "max":
                    value = float(matrix[row_index, seen].max())
                else:
                    value = float(matrix[row_index, seen].min()) - FLOOR_GAP
                matrix[row_index, floored[row_index]] = value
        scores[fill] = matrix
    return {
        "scores": scores,
        "raw": raw,
        "floored": floored,
        "n_shared": n_shared,
        "n_known": n_known,
        "n_single": n_single,
        "scoring": scoring,
    }


def _text_fold_ids(
    texts: Sequence[str],
    folds: int,
    seed: int,
    *,
    stream: int,
    unit: str,
) -> dict[str, int]:
    """A text-unit cross-fit's fold assignment (the CS-59/CS-60 scheme)."""
    values = tuple(dict.fromkeys(texts))
    if folds < 2:
        raise ValueError(f"a cross-fit needs at least two folds, got {folds}")
    if len(values) < folds:
        raise ValueError(f"the {unit} cross-fit needs at least {folds} {unit}s, got {len(values)}")
    order = np.random.default_rng([int(seed), int(stream)]).permutation(len(values))
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(order, folds)]
    mapping: dict[str, int] = {}
    for fold, chunk in enumerate(chunks):
        for index in chunk:
            mapping[values[int(index)]] = fold
    return mapping


def question_fold_ids(
    question_texts: Sequence[str],
    folds: int,
    seed: int,
) -> dict[str, int]:
    """The question-level cross-fit's fold assignment (CS-59's scheme).

    The distinct question texts are permuted by ``default_rng([seed, 2])`` — a
    stream independent of the pair folds (``default_rng(seed)``), the blind
    permutation (``default_rng([seed, 1])``) and the paragraph folds
    (``default_rng([seed, 3])``) — and split into contiguous chunks, the
    track's cross-fit scheme one level up.

    Args:
        question_texts: The distinct question texts (the fold unit).
        folds: The number of folds (at least 2).
        seed: The run's seed.

    Returns:
        A question text -> fold mapping.

    Raises:
        ValueError: If there are fewer questions than folds, or fewer than two
            folds.
    """
    return _text_fold_ids(question_texts, folds, seed, stream=2, unit="question")


def paragraph_fold_ids(
    paragraph_texts: Sequence[str],
    folds: int,
    seed: int,
) -> dict[str, int]:
    """The paragraph-level cross-fit's fold assignment (CS-60's scheme).

    The distinct paragraph texts are permuted by ``default_rng([seed, 3])`` —
    a stream independent of the pair folds (``[seed]``), the blind
    (``[seed, 1]``) and the question folds (``[seed, 2]``) — and split into
    contiguous chunks. The held-out unit is the paragraph text; the questions
    stay shared, which is the quantity a routing rule reads.

    Args:
        paragraph_texts: The distinct paragraph texts (the fold unit).
        folds: The number of folds (at least 2).
        seed: The run's seed.

    Returns:
        A paragraph text -> fold mapping.

    Raises:
        ValueError: If there are fewer paragraphs than folds, or fewer than two
            folds.
    """
    return _text_fold_ids(paragraph_texts, folds, seed, stream=3, unit="paragraph")


def _cross_fit(
    paragraph_texts: Sequence[str],
    table_question_texts: Sequence[str],
    labels: Sequence[int],
    fold_ids: np.ndarray,
    *,
    eval_question_texts: Sequence[str] | None,
    paragraphs: Sequence[str] | None,
    questions: Sequence[str] | None,
    scorings: Sequence[str],
    stopwords: frozenset[str],
    fills: Sequence[str],
    min_df: int = 1,
    control_cells: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The shared assembly: each judged cell from the table of its own fold.

    ``table_question_texts`` is the side the table is **built** from (the real
    questions for the arm, the permuted ones for the blind); ``eval_question_texts``
    is the side the judged cells are **read** at (always the real questions).
    """
    label_values = [int(label) for label in labels]
    eval_questions = (
        tuple(eval_question_texts)
        if eval_question_texts is not None
        else tuple(table_question_texts)
    )
    fold_values = sorted({int(value) for value in fold_ids})
    if len(fold_values) < 2:
        raise ValueError(f"a cross-fit needs at least two folds, got {fold_values}")
    grid_paragraphs = (
        tuple(paragraphs) if paragraphs is not None else tuple(dict.fromkeys(paragraph_texts))
    )
    grid_questions = (
        tuple(questions) if questions is not None else tuple(dict.fromkeys(eval_questions))
    )
    paragraph_of = {text: index for index, text in enumerate(grid_paragraphs)}
    question_of = {text: index for index, text in enumerate(grid_questions)}
    rows = np.asarray([paragraph_of[text] for text in paragraph_texts], dtype=np.int64)
    columns = np.asarray([question_of[text] for text in eval_questions], dtype=np.int64)

    full_table = build_word_pair_map(
        list(paragraph_texts),
        list(table_question_texts),
        label_values,
        stopwords=stopwords,
        min_df=min_df,
    )
    fold_tables: dict[int, WordPairMap] = {}
    for fold in fold_values:
        training = [index for index in range(len(label_values)) if int(fold_ids[index]) != fold]
        fold_tables[fold] = build_word_pair_map(
            [paragraph_texts[index] for index in training],
            [table_question_texts[index] for index in training],
            [label_values[index] for index in training],
            stopwords=stopwords,
            min_df=min_df,
        )

    floor = np.zeros(len(label_values), dtype=bool)
    n_shared = np.zeros(len(label_values), dtype=np.int64)
    n_known = np.zeros(len(label_values), dtype=np.int64)
    n_single = np.zeros(len(label_values), dtype=np.int64)
    assembled: dict[str, dict[str, np.ndarray]] = {}
    full_scores: dict[str, dict[str, np.ndarray]] = {}
    fold_scores: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for scoring in scorings:
        full = score_word_pair_grid(
            grid_paragraphs,
            grid_questions,
            full_table,
            scoring=scoring,
            stopwords=stopwords,
            fills=fills,
        )
        folds_grid = {
            fold: score_word_pair_grid(
                grid_paragraphs,
                grid_questions,
                fold_tables[fold],
                scoring=scoring,
                stopwords=stopwords,
                fills=fills,
            )
            for fold in fold_values
        }
        full_scores[scoring] = full["scores"]
        fold_scores[scoring] = {fold: grid["scores"] for fold, grid in folds_grid.items()}
        assembled[scoring] = {}
        for fill in fills:
            matrix = full["scores"][fill].copy()
            for index in range(len(label_values)):
                fold = int(fold_ids[index])
                matrix[rows[index], columns[index]] = folds_grid[fold]["scores"][fill][
                    rows[index], columns[index]
                ]
            assembled[scoring][fill] = matrix
        if scoring == scorings[0]:
            for index in range(len(label_values)):
                fold = int(fold_ids[index])
                grid = folds_grid[fold]
                floor[index] = bool(grid["floored"][rows[index], columns[index]])
                n_shared[index] = int(grid["n_shared"][rows[index], columns[index]])
                n_known[index] = int(grid["n_known"][rows[index], columns[index]])
                n_single[index] = int(grid["n_single"][rows[index], columns[index]])

    control_vintage: dict[str, Any] | None = None
    if control_cells is not None:
        cells = [(int(row), int(column)) for row, column in control_cells]
        primary = scorings[0]
        control_vintage = {
            "full_mean": (
                float(
                    np.mean([full_scores[primary]["floor"][row, column] for row, column in cells])
                )
                if cells
                else None
            ),
            "per_fold_mean": {
                str(fold): (
                    float(
                        np.mean(
                            [
                                fold_scores[primary][fold]["floor"][row, column]
                                for row, column in cells
                            ]
                        )
                    )
                    if cells
                    else None
                )
                for fold in fold_values
            },
            "note": (
                "the unjudged cells are scored by the full reviewed-set table (they never "
                "entered a fit); the judged cells by a table of 4/5 of the unit, so the "
                "full-vs-fold mean is the vintage difference, recorded not hidden"
            ),
        }
    return {
        "paragraphs": grid_paragraphs,
        "questions": grid_questions,
        "scorings": {scoring: assembled[scoring] for scoring in scorings},
        "floor": floor,
        "n_shared": n_shared,
        "n_known": n_known,
        "n_single": n_single,
        "fold_ids": np.asarray([int(value) for value in fold_ids], dtype=np.int64),
        "fold_values": fold_values,
        "table": full_table.stats(),
        "fold_tables": {str(fold): fold_tables[fold].stats() for fold in fold_values},
        "control_vintage": control_vintage,
        "note": (
            "each judged cell is scored by the table of the fold that did not contain "
            "it; the unjudged cells carry the full reviewed-set table's score"
        ),
    }


def cross_fit_word_pair_seen(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    fold_ids: np.ndarray,
    *,
    paragraphs: Sequence[str] | None = None,
    questions: Sequence[str] | None = None,
    eval_question_texts: Sequence[str] | None = None,
    scorings: Sequence[str] = WORD_PAIR_SCORINGS,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    fills: Sequence[str] = ("floor", "neutral"),
    min_df: int = 1,
    control_cells: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The **all-questions-seen** arm: the round's pair-level cross-fit.

    Args:
        paragraph_texts: Each judged pair's paragraph.
        question_texts: Each judged pair's question text (the real one; the
            blind caller passes the permuted texts).
        labels: Each judged pair's 0/1 label.
        fold_ids: Each pair's fold (the command's ``_dictionary_fold_ids``).
        paragraphs: The grid's paragraph rows.
        questions: The grid's question columns (the **real** question space).
        eval_question_texts: The questions the judged cells are read at — the
            real ones (the blind caller passes the permuted ``question_texts``
            for the table and the real texts here).
        scorings: The arms to measure (the primary first).
        stopwords: The tokenizer's stopwords.
        fills: The empty-cell fills to assemble.
        min_df: The smallest ``df`` a row may have (``2`` is the CS-54
            single-observation diagnostic).
        control_cells: The control cells' ``(row, column)`` positions, for the
            vintage block.

    Returns:
        The cross-fit's grids and readout arrays (see :func:`_cross_fit`).
    """
    return _cross_fit(
        paragraph_texts,
        question_texts,
        labels,
        np.asarray(fold_ids, dtype=np.int64),
        eval_question_texts=eval_question_texts,
        paragraphs=paragraphs,
        questions=questions,
        scorings=tuple(scorings),
        stopwords=stopwords,
        fills=fills,
        min_df=min_df,
        control_cells=control_cells,
    )


def cross_fit_word_pair_questions(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    question_folds: Mapping[str, int],
    *,
    paragraphs: Sequence[str] | None = None,
    questions: Sequence[str] | None = None,
    eval_question_texts: Sequence[str] | None = None,
    scorings: Sequence[str] = WORD_PAIR_SCORINGS,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    fills: Sequence[str] = ("floor", "neutral"),
    min_df: int = 1,
    control_cells: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The **held-out-questions** arm: the question-level cross-fit.

    Each pair's fold is the fold of its question text (the question split's
    mapping), so the fold's table is built from the other folds' questions'
    pairs and the held-out questions' pairs are the only ones evaluated.

    Args:
        paragraph_texts: Each judged pair's paragraph.
        question_texts: Each judged pair's question text (the real one; the
            blind caller passes the permuted texts).
        labels: Each judged pair's 0/1 label.
        question_folds: The question text -> fold mapping from
            :func:`question_fold_ids` (under the blind, the mapping of the
            **permuted** questions and the permuted texts are passed).
        paragraphs: The grid's paragraph rows.
        questions: The grid's question columns (the **real** question space).
        eval_question_texts: The questions the judged cells are read at — the
            real ones (the blind caller passes the permuted ``question_texts``
            for the table and the real texts here).
        scorings: The arms to measure (the primary first).
        stopwords: The tokenizer's stopwords.
        fills: The empty-cell fills to assemble.
        min_df: The smallest ``df`` a row may have (``2`` is the CS-54
            single-observation diagnostic).
        control_cells: The control cells' ``(row, column)`` positions.

    Returns:
        The cross-fit's grids and readout arrays (see :func:`_cross_fit`).

    Raises:
        ValueError: If a question has no fold.
    """
    missing = sorted({text for text in question_texts if text not in question_folds})
    if missing:
        raise ValueError(f"{len(missing)} question(s) have no fold, e.g. {missing[0]!r}")
    fold_ids = np.asarray([int(question_folds[text]) for text in question_texts], dtype=np.int64)
    return _cross_fit(
        paragraph_texts,
        question_texts,
        labels,
        fold_ids,
        eval_question_texts=eval_question_texts,
        paragraphs=paragraphs,
        questions=questions,
        scorings=tuple(scorings),
        stopwords=stopwords,
        fills=fills,
        min_df=min_df,
        control_cells=control_cells,
    )
