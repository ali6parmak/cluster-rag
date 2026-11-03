"""The **contrastive map** (the track's CS-40): "why A and not B".

The corpus map (CS-23) is a relation between **labels and words**, and the
viewer reads it one label at a time: this label's words, that label's words,
this text's words and the labels they lean to. The question a reader actually
has about a **pair** of labels — *why A and not B?* — is answered nowhere: the
two tables are read separately, never against each other.

This module is the **read** that answers it. It takes **two tables** (in the
artifact's own row shape) and returns the words that **discriminate** them:

- ``lo_A(w)`` and ``lo_B(w)`` — the two log-odds: the **stored** value when the
  pair is in the table and **0** when it is not. The zero is the map's own
  **silence convention**: a word the label's table does not hold contributes
  nothing to that label, exactly as the map's scorer counts it (a missing word
  adds 0 to the label's score);
- ``contrast(w) = lo_A(w) − lo_B(w)`` — above 0 the word pulls A up relative
  to B, below 0 it pulls B up relative to A, 0 it says nothing about the pair;
- the **flags** — ``in_a`` / ``in_b`` (which table holds the word), ``in_text``
  (the word is one of the text's own content words) and ``sign_disagreement``
  (the word is in **both** tables and its two log-odds have **opposite
  signs** — the plan's first reading: the words in A's table with a positive
  log-odds and in B's with a negative one).

**Nothing is rebuilt here.** The map's artifact holds every (label, word) pair,
so this is a read of the map and never a new corpus pass.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from cluster_rag.text_processors.features.lexical_map import cell_coverage

# How many words the corpus's own half of the block shows **in each
# direction** (the positive contrast and the negative one). The map's own head
# size — CS-36's ``HEAD_K``.
CONTRAST_TOP_K = 10

# The map's **silence convention**: a pair that is not in a label's table
# contributes this to that label (nothing).
SILENCE = 0.0

# How many labels of a word's reverse table the viewer prints (the "the labels
# it leans to" block). The novelty measure reads the screen as it is, so this
# is the screen's own limit.
LEAN_LIMIT = 3

# How many rows of a label's table the viewer displays (the corpus block's
# own limit — the screen's ``--top-k``). The novelty measure reads the screen
# as it is, so this is the screen's own limit too.
DISPLAY_TOP_K = 5

# The registered bars of CS-40's measurement (the pre-registration, in
# ``TEXT_PROCESSING_PLAN.md`` section 7.5).
VOCABULARY_BAR = 0.50
RELEVANCE_BAR = 0.50
NOVELTY_BAR = 0.50


@dataclass(frozen=True)
class ContrastRow:
    """One word of a pair's contrast: what it says about each of the two labels.

    Attributes:
        word: The word (the map's own unit: a content word of the corpus).
        log_odds_a: The first label's log-odds for the word (0 when the pair is
            not in that label's table — the map's silence convention).
        log_odds_b: The second label's log-odds for the word (0 when the pair
            is not in that label's table).
        contrast: The difference (the first label's log-odds minus the second
            one's): above 0 the word pulls the first label up relative to the
            second, below 0 the other way round.
        in_a: Whether the first label's table holds the word.
        in_b: Whether the second label's table holds the word.
        in_text: Whether the word is one of **this** text's content words.
        sign_disagreement: Whether the word is in **both** tables with
            **opposite** signs — the plan's first reading, as a flag.
        n_with_a: The corpus texts that have the word **and** the first label.
        n_with_b: The corpus texts that have the word **and** the second label.
        df: The corpus texts that have the word at all.
        word_class: The word's class (CS-33): ``distinctive``, ``shared`` or
            ``generic``.
        is_name_word: Whether the word is one of **either** label's own name
            words (the name-word flag of CS-33).
    """

    word: str
    log_odds_a: float
    log_odds_b: float
    contrast: float
    in_a: bool
    in_b: bool
    in_text: bool
    sign_disagreement: bool
    n_with_a: int
    n_with_b: int
    df: int
    word_class: str
    is_name_word: bool

    def to_row(self) -> dict[str, object]:
        """The row as a JSON-friendly dict (the artifact's own shape)."""
        return {
            "word": self.word,
            "log_odds_a": round(self.log_odds_a, 6),
            "log_odds_b": round(self.log_odds_b, 6),
            "contrast": round(self.contrast, 6),
            "in_a": self.in_a,
            "in_b": self.in_b,
            "in_text": self.in_text,
            "sign_disagreement": self.sign_disagreement,
            "n_with_a": self.n_with_a,
            "n_with_b": self.n_with_b,
            "df": self.df,
            "word_class": self.word_class,
            "is_name_word": self.is_name_word,
        }


def _row_by_word(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    """A table's rows, keyed by the word (the last row of a word wins)."""
    return {str(row.get("word")): row for row in rows}


def contrast_rows(
    a_rows: Sequence[Mapping[str, object]],
    b_rows: Sequence[Mapping[str, object]],
    *,
    text_words: Iterable[str] = (),
) -> tuple[ContrastRow, ...]:
    """Every word of the **union** of two tables, with the pair's numbers.

    The rows are ranked by the size of the contrast (the strongest difference
    first), and the tie-break is the word's own order, so the reading is
    deterministic.

    Args:
        a_rows: The first label's table (the artifact's own rows).
        b_rows: The second label's table (the artifact's own rows).
        text_words: The text's own content words (the ``in_text`` flag).

    Returns:
        The pair's words, strongest contrast first.
    """
    first = _row_by_word(a_rows)
    second = _row_by_word(b_rows)
    in_text = set(text_words)
    names = {str(row.get("word")) for row in list(first.values()) + list(second.values())}
    name_words = {
        str(row.get("word"))
        for row in list(first.values()) + list(second.values())
        if row.get("is_name_word")
    }
    rows: list[ContrastRow] = []
    for word in names:
        row_a = first.get(word)
        row_b = second.get(word)
        log_odds_a = float(row_a.get("log_odds", SILENCE)) if row_a else SILENCE
        log_odds_b = float(row_b.get("log_odds", SILENCE)) if row_b else SILENCE
        contrast = log_odds_a - log_odds_b
        rows.append(
            ContrastRow(
                word=word,
                log_odds_a=log_odds_a,
                log_odds_b=log_odds_b,
                contrast=contrast,
                in_a=row_a is not None,
                in_b=row_b is not None,
                in_text=word in in_text,
                sign_disagreement=(
                    row_a is not None and row_b is not None and log_odds_a * log_odds_b < 0
                ),
                n_with_a=int((row_a or {}).get("n_with", 0)),
                n_with_b=int((row_b or {}).get("n_with", 0)),
                df=int((row_a or row_b or {}).get("df", 0)),
                word_class=str((row_a or row_b or {}).get("word_class", "")),
                is_name_word=word in name_words,
            )
        )
    rows.sort(key=lambda row: (-abs(row.contrast), row.word))
    return tuple(rows)


def corpus_contrast_rows(
    a_rows: Sequence[Mapping[str, object]],
    b_rows: Sequence[Mapping[str, object]],
    *,
    top_k: int = CONTRAST_TOP_K,
) -> tuple[ContrastRow, ...]:
    """The corpus's own discriminators: the **top-k in each direction**.

    The positive direction (the words that pull the first label up relative to
    the second) and the negative one (the words that pull the second up) are
    ranked separately, each by the size of the contrast; a word with no
    contrast at all says nothing about the pair and is not shown.

    Args:
        a_rows: The first label's table (the artifact's own rows).
        b_rows: The second label's table (the artifact's own rows).
        top_k: How many words each direction shows.

    Returns:
        The words, strongest contrast first.

    Raises:
        ValueError: On a negative ``top_k``.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    rows = contrast_rows(a_rows, b_rows)
    positive = [row for row in rows if row.contrast > 0][:top_k]
    negative = [row for row in rows if row.contrast < 0][:top_k]
    kept = {row.word for row in positive} | {row.word for row in negative}
    return tuple(row for row in rows if row.word in kept)


def text_contrast_rows(
    a_rows: Sequence[Mapping[str, object]],
    b_rows: Sequence[Mapping[str, object]],
    *,
    text_words: Iterable[str],
) -> tuple[ContrastRow, ...]:
    """The text's own words and what each of the two labels makes of them.

    Args:
        a_rows: The first label's table (the artifact's own rows).
        b_rows: The second label's table (the artifact's own rows).
        text_words: The text's own content words.

    Returns:
        The text's words that are in either table, strongest contrast first.
    """
    wanted = set(text_words)
    return tuple(
        row for row in contrast_rows(a_rows, b_rows, text_words=wanted) if row.word in wanted
    )


def map_score(
    rows: Sequence[Mapping[str, object]],
    text_words: Iterable[str],
) -> float:
    """The map's own arm for one label, read from its table.

    The arm is the track's ``sum_zscore_positive`` (CS-31's): the sum of the
    **z-scores** of the text's words that are in the label's table, over the
    words whose z-score is **positive**. It is read from the artifact's own
    rows, so the number is the map's, not a rebuild's.

    Args:
        rows: The label's table (the artifact's own rows).
        text_words: The text's own content words.

    Returns:
        The score (0.0 when the text shares no word with the table).
    """
    wanted = set(text_words)
    total = 0.0
    for row in rows:
        word = str(row.get("word"))
        if word not in wanted:
            continue
        z_score = float(row.get("z_score", 0.0))
        if z_score > 0:
            total += z_score
    return total


def default_contrast_label(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    text_words: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    labels: Sequence[str] = (),
) -> str:
    """The map's own runner-up among the labels that are **not** excluded.

    The runner-up is the label with the largest map score (the map's own arm)
    among the labels whose table is not empty. The tie-break is the label
    space's own order, so the reading is deterministic.

    Args:
        tables: Every label's table (the artifact's own rows).
        text_words: The text's own content words.
        exclude: The labels that cannot be the runner-up (the text's own gold
            labels).
        labels: The label space's own order (the tie-break).

    Returns:
        The runner-up's name, or ``""`` when no candidate carries a positive
        score (no evidence, so there is no contrast to show).
    """
    skip = set(exclude)
    wanted = set(text_words)
    order = list(labels) if labels else sorted(tables)
    best = ""
    best_score = 0.0
    for label in order:
        if label in skip:
            continue
        rows = tables.get(label) or ()
        if not rows:
            # an empty table carries no evidence: it never wins
            continue
        score = map_score(rows, wanted)
        if score > best_score:
            best = label
            best_score = score
    return best


def focus_label(
    text: str,
    gold_labels: Sequence[str],
    labels: Sequence[str],
) -> str:
    """The **focus** label: the gold label whose own words the text carries the most.

    The viewer's own rule (unchanged): the topic the text is really about, and
    not the catch-all that happens to be listed first. The tie-break is the
    record's own label order, so the choice is deterministic.

    Args:
        text: The text on screen.
        gold_labels: The text's gold labels, in the record's own order.
        labels: The label space.

    Returns:
        The focus label (the label space's first label when the text carries
        no gold label of the space).
    """
    candidates = [label for label in gold_labels if label in labels]
    if not candidates:
        return labels[0] if labels else ""
    scores = [(label, cell_coverage(text, label)) for label in candidates]
    best = max(score for _, score in scores)
    return next(label for label, score in scores if score == best)


def shown_pairs(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    reverse: Mapping[str, Sequence[Mapping[str, object]]],
    label: str,
    *,
    text_words: Iterable[str],
    top_k: int,
    lean_limit: int = LEAN_LIMIT,
) -> frozenset[str]:
    """The words whose ``(label, word)`` log-odds **the screen prints**.

    The screen (as it is now) prints the corpus map's log-odds for a pair in
    three places: a label's own displayed table rows (the top-k of the table),
    the text's words inside that label's table (the word-class block) and the
    labels a text word leans to (the reverse table's top rows). This function
    is the union of those three, per label.

    Args:
        tables: Every label's table (the artifact's own rows).
        reverse: The reverse table (every word's label rows).
        label: The label to read.
        text_words: The text's own content words.
        top_k: How many rows of the label's table the screen displays.
        lean_limit: How many labels of a word's reverse table the screen shows.

    Returns:
        The words whose log-odds for the label are on the screen.
    """
    wanted = set(text_words)
    shown: set[str] = set()
    rows = tables.get(label) or ()
    for row in rows[:top_k]:
        shown.add(str(row.get("word")))
    for row in rows:
        word = str(row.get("word"))
        if word in wanted:
            shown.add(word)
    for word in wanted:
        for entry in (reverse.get(word) or ())[:lean_limit]:
            if str(entry.get("label")) == label:
                shown.add(word)
    return frozenset(shown)


def novelty_share(
    rows: Sequence[ContrastRow],
    shown_a: frozenset[str],
    shown_b: frozenset[str],
) -> float:
    """The share of the rows for which the screen prints **not both** numbers.

    The reader could not have formed the difference for those rows, so the
    block's numbers are not on the screen already.

    Args:
        rows: The block's rows.
        shown_a: The words whose log-odds for the first label the screen prints.
        shown_b: The words whose log-odds for the second label the screen prints.

    Returns:
        The share (0.0 when there are no rows).
    """
    if not rows:
        return 0.0
    new = sum(1 for row in rows if not (row.word in shown_a and row.word in shown_b))
    return new / len(rows)


def vocabulary_share(
    rows: Sequence[ContrastRow],
    *,
    name_words: Iterable[str] = (),
    text_words: Iterable[str] = (),
) -> float:
    """The share of the rows that the name-overlap maps **cannot** show.

    The topic-side and the text-side maps are the **name-overlap** features:
    their vocabulary is the labels' own name words and the text's own words.
    The rows outside that vocabulary are the corpus's own — the words the map
    knows and the name-overlap features structurally cannot see.

    Args:
        rows: The block's rows.
        name_words: The two labels' own name words (the row's own name-word
            flag counts too).
        text_words: The text's own content words.

    Returns:
        The share (0.0 when there are no rows).
    """
    if not rows:
        return 0.0
    covered = set(name_words) | set(text_words)
    outside = sum(1 for row in rows if row.word not in covered and not row.is_name_word)
    return outside / len(rows)


def relevance_share(rows: Sequence[ContrastRow]) -> float:
    """The share of the rows that are one of **this** text's own words.

    Args:
        rows: The block's rows.

    Returns:
        The share (0.0 when there are no rows).
    """
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.in_text) / len(rows)
