"""The **question map** (the track's CS-48): the reviewed pairs' own word↔question
association.

The OMCT twin of the UPR corpus map (CS-23). The UPR map answers "which words go
with which **label** in the training corpus"; on OMCT the only label a pair has
is its **question**, so the map answers "which words go with which **question**
in the reviewed pairs". The unit is pinned here because it is the whole
question of the transfer:

- **the row is the reviewed pair**;
- **the "text"** is the pair's **paragraph** — the only text a pair has;
- **the "label"** is the pair's **question**;
- the word is *present in the pair's paragraph*.

For question ``q`` and word ``w`` the 2×2 table is over ``q``'s reviewed pairs:
``a`` = positive pairs whose paragraph contains the word, ``b`` = negative
pairs whose paragraph contains it, ``c`` = positive pairs whose paragraph does
not, ``d`` = negative pairs whose paragraph does not. ``n_q = a + b + c + d``
is the question's reviewed-pair count and ``df_q(w) = a + b`` the word's
frequency **among that question's pairs**.

**The measures** (every one stored with the counts behind it, so any number can
be recomputed and audited):

- ``n_with`` — the question's reviewed pairs that are positive and whose
  paragraph contains the word;
- ``df`` — the question's reviewed pairs whose paragraph contains the word at
  all (positives and negatives);
- ``lift`` = ``P(positive | word present) / P(positive)`` within the question —
  1.0 says nothing, above 1.0 pulls the question, below 1.0 pushes it away;
- ``log_odds`` = ``log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5))`` — the
  0.5-smoothed signed version, the same formula as CS-23's map (preferred to
  the lift because the lift explodes on rare words).

**No thresholds.** Every (question, word) pair that occurs is stored, with its
counts; the noise is displayed, never removed.

**The reviewed pairs are the corpus.** The map is built from the **reviewed**
pairs only (subset A). The unjudged control (subset B) never enters it and is
never a label. A question with **no positive pair** has an **empty table** by
construction — reported as empty, never fabricated — which on this data is a
large class (85 of 294 questions): the map is *silent* about 331 of the 1,520
pairs.

**The cross-fit availability** (the map's usability readout, and CS-49's
scoring discipline): the folds are over the **pairs**; each fold's map is built
from the other folds, and every pair is read against a map that never saw it.
A pair whose question has no table in that map, or no word with a positive
log-odds, carries the row's floor and can never be ranked first — so the
availability is the number that says how much of the reviewed set the map can
speak about at all.

Pure core: the standard library only (no model client, no I/O).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.corpus_map import log_odds
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.lexicon import pair_measures, prior_total
from cluster_rag.text_processors.lexical.pos_class import POS_CLASSES
from cluster_rag.text_processors.lexical.word_class import (
    DISTINCTIVE_MAX_TABLES,
    GENERIC_SHARE,
    WORD_CLASSES,
    classify_word,
    generic_threshold,
)

# The unit, as the artifact records it (the descriptions are the block's own
# words: a reader must be able to tell what the "text" and the "label" are).
UNIT_ROW = "the reviewed pair (one judged paragraph-question cell)"
UNIT_TEXT = "the pair's paragraph (the only text a pair has)"
UNIT_LABEL = "the pair's question (the text of the question column)"
UNIT_WORD = "the word is present in the pair's paragraph"
UNIT_N = "the question's reviewed-pair count"
UNIT_DF = "the word's frequency among that question's reviewed pairs"

# The register's threshold: a word that sits in at least a **quarter** of the
# questions' tables is the generic class (CS-33's rule, a share not a magic
# number). The same threshold the UPR map used, so the two readouts are
# comparable.
REGISTER_QUARTER = 0.25


@dataclass(frozen=True)
class QuestionPair:
    """One (question, word) pair of the question map, with the counts behind it.

    Attributes:
        question: The question's text (the map's column key).
        word: The word, as the track's own content-word tokenizer writes it.
        n_with: The question's reviewed pairs that are positive and whose
            paragraph contains the word.
        df: The question's reviewed pairs whose paragraph contains the word at
            all (positives and negatives).
        lift: ``P(positive | word present) / P(positive)`` within the question.
        log_odds: The 0.5-smoothed signed log-odds over the 2×2 table.
        log_odds_informative: The informative-prior log-odds (CS-50): the same
            table with the word's **global** document frequency as the
            pseudo-count (Monroe et al. 2008's Dirichlet prior, adapted to the
            pair unit).
        standard_error: The standard error of that log-odds.
        z_score: The informative log-odds over the standard error.
        g2: The likelihood-ratio G² statistic on the same 2×2 table (1 df).
        p_value: The G² statistic's p-value.
        log_odds_conditional: The log-odds with the 0.5 added only where a cell
            is actually zero.
    """

    question: str
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
class QuestionMap:
    """The question map: the questions' word tables and the reverse tables.

    Attributes:
        n_reviewed_pairs: The reviewed pairs the map was built from.
        n_positive: How many of them are positive.
        n_negative: How many of them are negative.
        questions: The question space (every question, including the ones with
            no positive — they are reported as empty).
        question_counts: How many reviewed pairs each question has.
        question_positive_counts: How many positive reviewed pairs each
            question has.
        per_question: Each question's word table, strongest log-odds first.
        per_word: Each word's question table, strongest log-odds first (the
            same data read the other way).
        word_df: Each word's frequency over the whole reviewed set (the
            informative prior's input in CS-50).
        alpha0: The estimator's prior total (CS-50): Σ over the vocabulary of
            the words' global document frequencies plus the floor.
        word_classes: Each word's class (CS-51): how many of the **labelled**
            tables it sits in — ``distinctive`` (1-2), ``shared`` (3 to a
            quarter of them - 1) or ``generic`` (at least a quarter).
        pos_classes: Each word's grammatical class (CS-52): the class of its
            modal tag in the reviewed paragraphs. Empty when the corpus was
            not tagged.
    """

    n_reviewed_pairs: int
    n_positive: int
    n_negative: int
    questions: tuple[str, ...]
    question_counts: Mapping[str, int]
    question_positive_counts: Mapping[str, int]
    per_question: Mapping[str, tuple[QuestionPair, ...]]
    per_word: Mapping[str, tuple[QuestionPair, ...]]
    word_df: Mapping[str, int]
    alpha0: float = 0.0
    word_classes: Mapping[str, str] = field(default_factory=dict)
    pos_classes: Mapping[str, str] = field(default_factory=dict)

    @property
    def n_questions(self) -> int:
        """How many questions the map holds (the reviewed set's questions)."""
        return len(self.questions)

    @property
    def n_questions_with_table(self) -> int:
        """How many questions have at least one word in their table."""
        return sum(1 for question in self.questions if self.per_question.get(question))

    @property
    def empty_questions(self) -> tuple[str, ...]:
        """The questions with **no positive** — their tables are empty by construction."""
        return tuple(question for question in self.questions if not self.per_question.get(question))

    @property
    def n_words(self) -> int:
        """How many distinct words the map holds."""
        return len(self.per_word)

    @property
    def n_stored_pairs(self) -> int:
        """How many (question, word) pairs the map holds."""
        return sum(len(rows) for rows in self.per_question.values())

    def table_rows(self, question: str) -> tuple[QuestionPair, ...]:
        """One question's word table (empty when the question has no positive)."""
        return tuple(self.per_question.get(question, ()))

    def word_table_counts(self) -> Mapping[str, int]:
        """How many questions' tables each word sits in (the register's raw material)."""
        return {word: len(rows) for word, rows in self.per_word.items()}

    def question_table_sizes(self) -> dict[str, int]:
        """The number of words in each question's table (0 for an empty one)."""
        return {question: len(self.per_question.get(question, ())) for question in self.questions}

    def contrastable_questions(self) -> tuple[str, ...]:
        """The questions whose reviewed pairs carry **both** classes.

        A question whose pairs are all positive (or all negative) has no
        within-question contrast: its table cannot rank the question's own
        pairs, and the map can only remember (or stay silent). Reported as its
        own class, not dropped.
        """
        return tuple(
            question
            for question in self.questions
            if 0
            < self.question_positive_counts.get(question, 0)
            < self.question_counts.get(question, 0)
        )


def build_question_map(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    units: Any = None,
    word_pos: Mapping[str, str] | None = None,
) -> QuestionMap:
    """Build the question map from the **reviewed** pairs.

    The corpus is the reviewed set (subset A) and nothing else: the control
    cells (subset B) are never handed in, so they can never enter the counts.
    A question with no positive pair gets an **empty** table (never a
    fabricated one).

    Args:
        paragraph_texts: Each reviewed pair's paragraph.
        question_texts: Each reviewed pair's question text (the map's column).
        labels: Each reviewed pair's 0/1 label, in the same order.
        stopwords: The stopwords of the track's content-word tokenizer.
        units: The map's **unit**: the callable that turns a paragraph into the
            units its tables are keyed by. The default is the track's own
            content-word tokenizer; the phrase variant (CS-53) passes a bigram
            tokenizer instead.
        word_pos: The words' **grammatical classes** (CS-52), when the
            paragraphs were tagged. The class is a property of the word, so it
            rides on every table the word sits in.

    Returns:
        The question map.

    Raises:
        ValueError: If the inputs have different lengths, or a label is not 0/1.
    """
    if not (len(paragraph_texts) == len(question_texts) == len(labels)):
        raise ValueError(
            "the reviewed pairs' paragraphs, questions and labels must have the same "
            f"length, got {len(paragraph_texts)} / {len(question_texts)} / {len(labels)}"
        )
    bad = sorted({label for label in labels if label not in (0, 1)})
    if bad:
        raise ValueError(f"the pair labels must be 0/1, got {bad}")
    unit_of = units or (lambda text: content_tokens(text, stopwords))

    question_counts: Counter[str] = Counter(question_texts)
    question_positive_counts: Counter[str] = Counter(
        question for question, label in zip(question_texts, labels, strict=True) if label == 1
    )
    positive_words: dict[str, Counter[str]] = {question: Counter() for question in question_counts}
    question_df: dict[str, Counter[str]] = {question: Counter() for question in question_counts}
    word_df: Counter[str] = Counter()
    for paragraph, question, label in zip(paragraph_texts, question_texts, labels, strict=True):
        words = set(unit_of(paragraph))
        word_df.update(words)
        question_df[question].update(words)
        if label == 1:
            positive_words[question].update(words)

    # The estimator's prior total (CS-50): the reviewed set's own (pair, word)
    # incidences plus the floor's contribution — the map-level constant.
    alpha0 = prior_total(word_df.values())
    per_question: dict[str, tuple[QuestionPair, ...]] = {}
    per_word: dict[str, list[QuestionPair]] = {}
    for question in question_counts:
        n_pairs = question_counts[question]
        n_positive = question_positive_counts.get(question, 0)
        if not n_positive:
            # no positive pair: the table is empty by construction
            per_question[question] = ()
            continue
        positive_share = n_positive / n_pairs
        rows: list[QuestionPair] = []
        for word, n_with in positive_words[question].items():
            df = question_df[question].get(word, 0)
            if not df:
                continue
            measures = pair_measures(
                n_with,
                df,
                n_positive,
                n_pairs,
                alpha0=alpha0,
                # the prior's df is the word's GLOBAL frequency (the reviewed
                # set), the table's df is the question's own (CS-50's unit)
                prior_df=word_df.get(word, df),
            )
            rows.append(
                QuestionPair(
                    question=question,
                    word=word,
                    n_with=n_with,
                    df=df,
                    lift=(n_with / df) / positive_share,
                    log_odds=log_odds(n_with=n_with, df=df, n_label=n_positive, n_texts=n_pairs),
                    log_odds_informative=measures.log_odds_informative,
                    standard_error=measures.standard_error,
                    z_score=measures.z_score,
                    g2=measures.g2,
                    p_value=measures.p_value,
                    log_odds_conditional=measures.log_odds_conditional,
                )
            )
        rows.sort(key=lambda row: (-row.log_odds, row.word))
        per_question[question] = tuple(rows)
        for pair in rows:
            per_word.setdefault(pair.word, []).append(pair)
    frozen_per_word = {
        word: tuple(sorted(rows, key=lambda row: (-row.log_odds, row.question)))
        for word, rows in per_word.items()
    }
    # CS-51's word classes: how many of the LABELLED tables each word sits in
    # (a question with no positive has an empty table, so it never counts).
    n_labelled_tables = sum(1 for question in question_counts if positive_words[question])
    word_classes = {
        word: classify_word(len(rows), n_labelled_tables) for word, rows in frozen_per_word.items()
    }
    return QuestionMap(
        n_reviewed_pairs=len(labels),
        n_positive=sum(1 for label in labels if label == 1),
        n_negative=sum(1 for label in labels if label == 0),
        questions=tuple(question_counts),
        question_counts=dict(question_counts),
        question_positive_counts=dict(question_positive_counts),
        per_question=per_question,
        per_word=frozen_per_word,
        word_df=dict(word_df),
        alpha0=alpha0,
        word_classes=word_classes,
        pos_classes=dict(word_pos or {}),
    )


def cross_fit_availability(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    fold_ids: Sequence[int],
    *,
    documents: Sequence[str] | None = None,
    pair_ids: Sequence[str] | None = None,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    units: Any = None,
) -> dict[str, Any]:
    """Read every pair against the out-of-fold map: can the map speak about it?

    The folds are over the **pairs**; each fold's map is built from the other
    folds, so a pair is never read by a map that saw it. For every pair the
    readout carries two flags:

    - ``has_table`` — the fold's map holds a (non-empty) table for the pair's
      question;
    - ``n_evidence`` — how many words of the pair's paragraph appear in that
      table with a **positive** log-odds (the "evidence-for" reading; 0 means
      the pair would carry the row's floor).

    Args:
        paragraph_texts: Each reviewed pair's paragraph.
        question_texts: Each reviewed pair's question text.
        labels: Each reviewed pair's 0/1 label.
        fold_ids: Each pair's fold (an int; the folds are the distinct values).
        documents: Each pair's document (for the per-document breakdown).
        pair_ids: Each pair's id (for the per-pair audit rows).
        stopwords: The tokenizer's stopwords.
        units: The map's unit (the same one the map is built with).

    Returns:
        A JSON-serializable block: the per-pair rows, the per-document /
        per-class / per-fold aggregates and their notes.

    Raises:
        ValueError: If the inputs have different lengths, a label is not 0/1,
            or there are fewer than two folds.
    """
    n_pairs = len(labels)
    if not (len(paragraph_texts) == len(question_texts) == n_pairs == len(fold_ids)):
        raise ValueError(
            "the paragraphs, questions, labels and fold ids must have the same length, "
            f"got {len(paragraph_texts)} / {len(question_texts)} / {n_pairs} / {len(fold_ids)}"
        )
    if documents is not None and len(documents) != n_pairs:
        raise ValueError(f"documents must have {n_pairs} entries, got {len(documents)}")
    if pair_ids is not None and len(pair_ids) != n_pairs:
        raise ValueError(f"pair_ids must have {n_pairs} entries, got {len(pair_ids)}")
    folds = sorted(set(fold_ids))
    if len(folds) < 2:
        raise ValueError(f"a cross-fit needs at least two folds, got {folds}")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("the pair labels must be 0/1")

    unit_of = units or (lambda text: content_tokens(text, stopwords))
    rows: list[dict[str, Any]] = []
    fold_maps: dict[int, dict[str, dict[str, float]]] = {}
    for fold in folds:
        keep = [index for index, value in enumerate(fold_ids) if value != fold]
        fold_map = build_question_map(
            [paragraph_texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels[index] for index in keep],
            stopwords=stopwords,
            units=units,
        )
        fold_maps[fold] = {
            question: {pair.word: pair.log_odds for pair in fold_map.table_rows(question)}
            for question in fold_map.questions
        }
    for index in range(n_pairs):
        table = fold_maps[fold_ids[index]].get(question_texts[index], {})
        evidence = [
            word for word in set(unit_of(paragraph_texts[index])) if table.get(word, 0.0) > 0.0
        ]
        row: dict[str, Any] = {
            "fold": int(fold_ids[index]),
            "label": int(labels[index]),
            "has_table": bool(table),
            "n_evidence": len(evidence),
        }
        if pair_ids is not None:
            row["pair_id"] = pair_ids[index]
        if documents is not None:
            row["document"] = documents[index]
        rows.append(row)

    def _summary(mask: Sequence[bool]) -> dict[str, Any]:
        selected = [row for row, keep in zip(rows, mask, strict=True) if keep]
        n = len(selected)
        with_table = sum(1 for row in selected if row["has_table"])
        with_evidence = sum(1 for row in selected if row["n_evidence"] > 0)
        return {
            "n_pairs": n,
            "with_table": with_table,
            "with_table_share": (with_table / n) if n else None,
            "with_evidence": with_evidence,
            "with_evidence_share": (with_evidence / n) if n else None,
            "n_floor_pairs": n - with_table,
        }

    document_values = sorted({row["document"] for row in rows if "document" in row})
    block: dict[str, Any] = {
        "folds": len(folds),
        "fold_sizes": [sum(1 for fold in fold_ids if fold == value) for value in folds],
        "overall": _summary([True] * n_pairs),
        "by_class": {
            "positive": _summary([row["label"] == 1 for row in rows]),
            "negative": _summary([row["label"] == 0 for row in rows]),
        },
        "by_document": {
            document: _summary([row.get("document") == document for row in rows])
            for document in document_values
        },
        "by_fold": [_summary([row["fold"] == value for row in rows]) for value in folds],
        "per_pair": rows,
        "note": (
            "every pair is read against the map built from the OTHER folds; "
            "has_table = the fold's map has a non-empty table for the pair's "
            "question; n_evidence = the words of the pair's paragraph with a "
            "positive log-odds in that table (0 means the pair carries the row's floor)"
        ),
    }
    return block


def _head(rows: Sequence[QuestionPair], measure: str) -> str | None:
    """The head word of one table under one measure (max; ties by word, as the sort)."""
    if not rows:
        return None
    best = max(getattr(row, measure) for row in rows)
    return min(row.word for row in rows if getattr(row, measure) == best)


def _median(values: Sequence[float]) -> float | None:
    """The median of a non-empty sequence (None for an empty one)."""
    return float(statistics.median(values)) if values else None


def readout_block(question_map: QuestionMap) -> dict[str, Any]:
    """The map's four pre-registered readouts (the shape's description).

    - **the measures readout**: the head word of every question's table under
      the lift and under the log-odds, and how often the two disagree;
    - **the thickness readout**: how many of a table's words occur in two or
      more of the question's positives (a shared word generalizes; a word in
      one positive is a memory of that paragraph);
    - **the register readout**: how many questions' tables each word sits in,
      and whether the "at least a quarter" class exists;
    - **the contrast readout**: the questions whose reviewed pairs are all
      positive / all negative (no within-question contrast).

    Args:
        question_map: The map.

    Returns:
        A JSON-serializable readout block.
    """
    head_rows: list[dict[str, Any]] = []
    n_differ = 0
    n_contrastable = 0
    n_differ_contrastable = 0
    n_only_in_positives = 0
    n_hapax = 0
    shared_word_counts: list[int] = []
    shared_word_counts_multi: list[int] = []
    all_positive: list[str] = []
    all_negative: list[str] = []
    for question in question_map.questions:
        rows = question_map.table_rows(question)
        n_pairs = question_map.question_counts.get(question, 0)
        n_positive = question_map.question_positive_counts.get(question, 0)
        if n_positive == n_pairs:
            all_positive.append(question)
        elif n_positive == 0:
            all_negative.append(question)
        if not rows:
            continue
        lift_head = _head(rows, "lift")
        log_odds_head = _head(rows, "log_odds")
        head_row = next(row for row in rows if row.word == log_odds_head)
        differs = lift_head != log_odds_head
        n_differ += differs
        contrastable = 0 < n_positive < n_pairs
        if contrastable:
            n_contrastable += 1
            n_differ_contrastable += differs
        only_in_positives = head_row.df == head_row.n_with
        n_only_in_positives += only_in_positives
        n_hapax += head_row.df == 1
        shared = sum(1 for row in rows if row.n_with >= 2)
        shared_word_counts.append(shared)
        if n_positive >= 2:
            shared_word_counts_multi.append(shared)
        head_rows.append(
            {
                "question": question,
                "n_pairs": n_pairs,
                "n_positive": n_positive,
                "n_negative": n_pairs - n_positive,
                "contrastable": contrastable,
                "table_words": len(rows),
                "shared_words": shared,
                "lift_head": lift_head,
                "log_odds_head": log_odds_head,
                "head_differs": differs,
                "log_odds_head_df": head_row.df,
                "log_odds_head_n_with": head_row.n_with,
                "log_odds_head_only_in_positives": only_in_positives,
            }
        )

    table_counts = question_map.word_table_counts()
    quarter = max(1, math.ceil(REGISTER_QUARTER * question_map.n_questions))
    in_quarter = [word for word, count in table_counts.items() if count >= quarter]
    top_words = sorted(table_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "measures": {
            "n_with": "the question's positive reviewed pairs whose paragraph has the word",
            "df": "the question's reviewed pairs whose paragraph has the word at all",
            "lift": "P(positive | word present) / P(positive), within the question",
            "log_odds": "log((a+0.5)/(b+0.5)) + log((d+0.5)/(c+0.5)) on the 2x2 table",
        },
        "head_agreement": {
            "n_tables": len(head_rows),
            "n_differ": n_differ,
            "share_differ": (n_differ / len(head_rows)) if head_rows else None,
            "n_contrastable_tables": n_contrastable,
            "n_differ_contrastable": n_differ_contrastable,
            "share_differ_contrastable": (
                (n_differ_contrastable / n_contrastable) if n_contrastable else None
            ),
        },
        "log_odds_head": {
            "n_only_in_positives": n_only_in_positives,
            "share_only_in_positives": (
                (n_only_in_positives / len(head_rows)) if head_rows else None
            ),
            "n_hapax": n_hapax,
            "share_hapax": (n_hapax / len(head_rows)) if head_rows else None,
        },
        "thickness": {
            "n_tables": len(shared_word_counts),
            "n_tables_with_shared_word": sum(1 for count in shared_word_counts if count),
            "share_tables_with_shared_word": (
                (sum(1 for count in shared_word_counts if count) / len(shared_word_counts))
                if shared_word_counts
                else None
            ),
            "median_shared_words": _median(shared_word_counts),
            "n_multi_positive_tables": len(shared_word_counts_multi),
            "median_shared_words_multi_positive": _median(shared_word_counts_multi),
        },
        "register": {
            "quarter": quarter,
            "quarter_share": REGISTER_QUARTER,
            "n_words_in_quarter": len(in_quarter),
            "top": [
                {"word": word, "tables": count, "df": question_map.word_df.get(word, 0)}
                for word, count in top_words
            ],
        },
        "contrast": {
            "n_all_positive_questions": len(all_positive),
            "n_all_positive_pairs": sum(question_map.question_counts[q] for q in all_positive),
            "n_all_negative_questions": len(all_negative),
            "n_all_negative_pairs": sum(question_map.question_counts[q] for q in all_negative),
        },
        "per_question": head_rows,
        "note": (
            "the four readouts the change set's falsification is stated against: the "
            "measures' head agreement, the thickness (words shared by the question's "
            "positives), the register (a word in at least a quarter of the tables) and "
            "the contrast (all-positive / all-negative questions)"
        ),
    }


def estimator_block(question_map: QuestionMap) -> dict[str, Any]:
    """The estimator's readout (CS-50): what the prior and the z-score do.

    The three readings the change set is pre-registered against: the head word
    under each measure (the log-odds, the informative-prior log-odds, the
    z-score, the conditional smoothing) and how often each differs from the
    log-odds' own; the conditional smoothing's sign-flip share; and the
    z-scores' distribution (a prior that swamps a small table gives tiny,
    nearly equal z-scores — the reason the unit is expected to null the
    estimator).

    Args:
        question_map: The map.

    Returns:
        A JSON-serializable readout block.
    """
    heads: list[dict[str, Any]] = []
    n_tables = 0
    n_stored_pairs = 0
    n_differ_informative = 0
    n_differ_z = 0
    n_differ_conditional = 0
    sign_flips = 0
    z_values: list[float] = []
    for question in question_map.questions:
        rows = question_map.table_rows(question)
        if not rows:
            continue
        n_tables += 1
        head_log_odds = _head(rows, "log_odds")
        head_informative = _head(rows, "log_odds_informative")
        head_z = _head(rows, "z_score")
        head_conditional = _head(rows, "log_odds_conditional")
        n_differ_informative += head_log_odds != head_informative
        n_differ_z += head_log_odds != head_z
        n_differ_conditional += head_log_odds != head_conditional
        for row in rows:
            n_stored_pairs += 1
            z_values.append(row.z_score)
            if (row.log_odds > 0.0) != (row.log_odds_conditional > 0.0):
                sign_flips += 1
        heads.append(
            {
                "question": question,
                "n_pairs": question_map.question_counts.get(question, 0),
                "n_positive": question_map.question_positive_counts.get(question, 0),
                "head_log_odds": head_log_odds,
                "head_informative": head_informative,
                "head_z": head_z,
                "head_conditional": head_conditional,
                "z_differs": head_log_odds != head_z,
            }
        )
    z_positive = sum(1 for value in z_values if value > 0.0)
    return {
        "prior_total": question_map.alpha0,
        "prior": (
            "alpha_w = the word's global df over the reviewed pairs + 0.01; "
            "alpha0 = the sum over the vocabulary (the reference implementation, "
            "Monroe et al. 2008, adapted to the pair unit)"
        ),
        "n_tables": n_tables,
        "n_stored_pairs": n_stored_pairs,
        "head_difference": {
            "informative": n_differ_informative,
            "informative_share": (n_differ_informative / n_tables) if n_tables else None,
            "z": n_differ_z,
            "z_share": (n_differ_z / n_tables) if n_tables else None,
            "conditional": n_differ_conditional,
            "conditional_share": (n_differ_conditional / n_tables) if n_tables else None,
        },
        "conditional_sign_flip": {
            "n_pairs": sign_flips,
            "share": (sign_flips / n_stored_pairs) if n_stored_pairs else None,
        },
        "z_score": {
            "min": min(z_values) if z_values else None,
            "median": _median(z_values),
            "max": max(z_values) if z_values else None,
            "share_positive": (z_positive / len(z_values)) if z_values else None,
        },
        "per_question": heads,
        "note": (
            "the informative prior's df is the word's GLOBAL frequency while the "
            "table's df is the question's own: on a 1-3-pair table the prior "
            "dominates, so the z-scores are tiny and nearly equal — the reason "
            "the estimator cannot make two questions' weights comparable"
        ),
    }


def class_block(question_map: QuestionMap) -> dict[str, Any]:
    """The word-class readout (CS-51): CS-33's classes, crossed with the z-score.

    A word's class is how many of the **labelled** tables it sits in (CS-33's
    rule): ``distinctive`` (1-2), ``shared`` (the middle) or ``generic`` (at
    least a quarter). Per class the readout carries the words, the pairs, the
    share of **positive** log-odds, the mean log-odds **and the mean z-score**
    (CS-50's measure — the one the restricted arms are built on), plus the
    registered separation test on the z-score.

    Args:
        question_map: The map.

    Returns:
        A JSON-serializable readout block.
    """
    n_labelled = sum(
        1 for question in question_map.questions if question_map.per_question.get(question)
    )
    per_class: dict[str, dict[str, Any]] = {
        name: {"words": 0, "pairs": 0, "positive_pairs": 0, "log_odds": [], "z_scores": []}
        for name in WORD_CLASSES
    }
    for word, rows in question_map.per_word.items():
        name = question_map.word_classes.get(word, "")
        if name not in per_class:
            continue
        block = per_class[name]
        block["words"] += 1
        for row in rows:
            block["pairs"] += 1
            block["positive_pairs"] += 1 if row.log_odds > 0.0 else 0
            block["log_odds"].append(row.log_odds)
            block["z_scores"].append(row.z_score)
    report: dict[str, Any] = {}
    for name, block in per_class.items():
        pairs = block["pairs"]
        report[name] = {
            "words": block["words"],
            "pairs": pairs,
            "positive_pairs": block["positive_pairs"],
            "positive_share": (block["positive_pairs"] / pairs) if pairs else None,
            "mean_log_odds": (float(np.mean(block["log_odds"])) if block["log_odds"] else None),
            "mean_z_score": (float(np.mean(block["z_scores"])) if block["z_scores"] else None),
        }
    distinctive = report.get("distinctive") or {}
    generic = report.get("generic") or {}
    separation = {
        "mean_z_difference": (
            distinctive["mean_z_score"] - generic["mean_z_score"]
            if distinctive.get("mean_z_score") is not None
            and generic.get("mean_z_score") is not None
            else None
        ),
        "generic_positive_share_lower": (
            (generic.get("positive_share") or 0.0) < (distinctive.get("positive_share") or 1.0)
            if distinctive.get("positive_share") is not None
            and generic.get("positive_share") is not None
            else None
        ),
        "margin": 0.5,
        "note": (
            "the z-measure version of CS-33's separation test (the UPR margin was "
            "1.0 on the log-odds; the z-scores sit on a smaller scale, so the bar "
            "is halved and stated in advance)"
        ),
    }
    return {
        "thresholds": {
            "n_labelled_tables": n_labelled,
            "distinctive_max_tables": DISTINCTIVE_MAX_TABLES,
            "generic_share": GENERIC_SHARE,
            "generic_min_tables": generic_threshold(n_labelled),
        },
        "classes": report,
        "separation": separation,
        "note": (
            "a word's class is the number of LABELLED tables (the 209 questions "
            "with a positive) it sits in; CS-48's register readout used a quarter "
            "of ALL 294 questions, so its threshold (74) differs from this one"
        ),
    }


def pos_block(question_map: QuestionMap) -> dict[str, Any]:
    """The grammatical readout (CS-52): CS-34's classes, crossed with the z-score.

    Per POS class (the class of each word's modal tag): the words, the pairs,
    the share of **positive** log-odds, the mean log-odds and the **mean
    z-score**. The registered test is the **noun - verb z difference** against
    the 0.5 bar (the UPR test's margin was 1.0 on the log-odds; the z-scores
    sit on a smaller scale, so the bar is halved and stated in advance).

    Args:
        question_map: The map.

    Returns:
        A JSON-serializable readout block (with ``available: False`` when the
        corpus was not tagged).
    """
    if not question_map.pos_classes:
        return {"available": False, "note": "the reviewed paragraphs were not tagged"}
    per_class: dict[str, dict[str, Any]] = {
        name: {"words": 0, "pairs": 0, "positive_pairs": 0, "log_odds": [], "z_scores": []}
        for name in POS_CLASSES
    }
    for word, rows in question_map.per_word.items():
        name = question_map.pos_classes.get(word, "")
        if name not in per_class:
            continue
        block = per_class[name]
        block["words"] += 1
        for row in rows:
            block["pairs"] += 1
            block["positive_pairs"] += 1 if row.log_odds > 0.0 else 0
            block["log_odds"].append(row.log_odds)
            block["z_scores"].append(row.z_score)
    report: dict[str, Any] = {}
    for name, block in per_class.items():
        pairs = block["pairs"]
        report[name] = {
            "words": block["words"],
            "pairs": pairs,
            "positive_pairs": block["positive_pairs"],
            "positive_share": (block["positive_pairs"] / pairs) if pairs else None,
            "mean_log_odds": (float(np.mean(block["log_odds"])) if block["log_odds"] else None),
            "mean_z_score": (float(np.mean(block["z_scores"])) if block["z_scores"] else None),
        }
    nouns = report.get("noun") or {}
    verbs = report.get("verb") or {}
    difference = (
        nouns["mean_z_score"] - verbs["mean_z_score"]
        if nouns.get("mean_z_score") is not None and verbs.get("mean_z_score") is not None
        else None
    )
    return {
        "available": True,
        "classes": report,
        "separation": {
            "mean_z_difference": difference,
            "margin": 0.5,
            "holds": (difference is not None and difference >= 0.5),
            "note": (
                "the z-measure version of CS-34's noun-vs-verb test (the UPR "
                "margin was 1.0 on the log-odds; the z-scores sit on a smaller "
                "scale, so the bar is halved and stated in advance)"
            ),
        },
        "note": (
            "a word's class is the class of its modal tag in the reviewed "
            "paragraphs; the tagger is the edge's nltk client, its answers cached"
        ),
    }


def question_map_block(
    question_map: QuestionMap,
    availability: Mapping[str, Any] | None,
    *,
    folds: int,
    seed: int,
    n_paragraphs: int,
    n_documents: int,
) -> dict[str, Any]:
    """Assemble the artifact's `dictionary` block (CS-48).

    Args:
        question_map: The map.
        availability: The cross-fit availability block (None when the run did
            not compute it — the block then says so rather than fabricating).
        folds: The cross-fit's fold count.
        seed: The run's seed (the fold split's seed).
        n_paragraphs: The reviewed set's distinct paragraphs.
        n_documents: The reviewed set's documents.

    Returns:
        A JSON-serializable block: the unit, the corpus, the map's shape, the
        four readouts, the cross-fit and the full per-question tables.
    """
    table_sizes = list(question_map.question_table_sizes().values())
    contrastable = question_map.contrastable_questions()
    block: dict[str, Any] = {
        "unit": {
            "row": UNIT_ROW,
            "text": UNIT_TEXT,
            "label": UNIT_LABEL,
            "word": UNIT_WORD,
            "n_texts": UNIT_N,
            "df": UNIT_DF,
            "note": (
                "the UPR corpus map's unit (a text and the labels it carries) is not "
                "available on a pair dataset: the pair is the row, the paragraph the "
                "text, the question the label. CS-21's word-pair view (paragraph word x "
                "question word) is a DIFFERENT object and is not this map."
            ),
        },
        "corpus": {
            "n_reviewed_pairs": question_map.n_reviewed_pairs,
            "n_positive": question_map.n_positive,
            "n_negative": question_map.n_negative,
            "n_questions": question_map.n_questions,
            "n_paragraphs": int(n_paragraphs),
            "n_documents": int(n_documents),
        },
        "map": {
            "n_questions": question_map.n_questions,
            "n_questions_with_table": question_map.n_questions_with_table,
            "n_questions_empty": len(question_map.empty_questions),
            "empty_questions": list(question_map.empty_questions),
            "n_words": question_map.n_words,
            "n_stored_pairs": question_map.n_stored_pairs,
            "table_size": {
                "min": min(table_sizes) if table_sizes else 0,
                "median": _median(table_sizes),
                "max": max(table_sizes) if table_sizes else 0,
            },
            "n_contrastable_questions": len(contrastable),
            "n_contrastable_pairs": sum(question_map.question_counts[q] for q in contrastable),
        },
        "readout": readout_block(question_map),
        "estimator": estimator_block(question_map),
        "classes": class_block(question_map),
        "pos": pos_block(question_map),
        "cross_fit": {
            "folds": int(folds),
            "seed": int(seed),
            "scheme": (
                "numpy.random.default_rng(seed).permutation(n_pairs) split into "
                "contiguous chunks (the track's cross-fit scheme); each fold's map is "
                "built from the other folds"
            ),
        },
        "availability": dict(availability) if availability is not None else None,
        "tables": {
            question: [pair.to_row() for pair in question_map.table_rows(question)]
            for question in question_map.questions
        },
        "words": {
            word: {"tables": len(rows), "df": question_map.word_df.get(word, 0)}
            for word, rows in question_map.per_word.items()
        },
        "notes": [
            "the map is built from the REVIEWED pairs only (subset A); the unjudged "
            "control (subset B) never enters it and is never a label.",
            "there is no threshold: every (question, word) pair that occurs is stored "
            "with its counts; the noise is displayed, never removed by dropping data.",
            "a question with no positive pair has an EMPTY table by construction; the "
            "count is the map's silence, and the availability block says how many pairs "
            "it costs.",
            "OMCT is auxiliary data: these numbers annotate method choice, they are "
            "never a UPR metric.",
        ],
    }
    return block
