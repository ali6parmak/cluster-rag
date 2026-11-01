"""The **word-class readout** (the track's CS-33): what kind of words affect the
label, in which way.

The reviewer's part-of-speech question — *which kind of word pushes the label
up, and which pushes it down* — answered first **without a new dependency**,
from the **corpus map's own structure** (CS-23). A word's class says how many
of the labelled tables it sits in:

- **distinctive** — the word sits in **one or two** of the labelled tables (the
  corpus map's own wording): the topic's own vocabulary;
- **generic** — the word sits in **at least a quarter** of the labelled tables
  (≥ 15 of the 60): the register ("rights" is in 60 tables, "include" in 59);
- **shared** — the middle (3–14 tables);

and the **name word** as a *flag*: the word is in the label's own name. The
flag **overlaps** the classes — "rights" is the name word of "Rights of the
Child" *and* generic.

Crossed with the **sign** of the map's log-odds (up = with the label, down =
against it), that is the reviewer's question in the map's own vocabulary. The
same split is applied to the **surrogate's** per-topic coefficients, so the
model's own habits are read the same way.

**The readout** (per class): the number of (label, word) pairs, the share with
a **positive** log-odds (the word goes *with* the label), the share with a
**negative** one (it goes *against* it) and the mean log-odds.

**The separation test** (the change set's registered falsification): the
readout is a **null** when the generic and the distinctive classes have the
same profile — the distinctive class's mean log-odds must exceed the generic
class's by **at least 1.0** (one natural-log unit, a factor of *e* in the
odds) **and** the generic class's share of positive pairs must be lower.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens

# The three classes, in the order the readout reports them (from the narrowest
# to the broadest).
WORD_CLASSES: tuple[str, ...] = ("distinctive", "shared", "generic")

# The class's own names (the strings the artifact and the config carry).
DISTINCTIVE = "distinctive"
SHARED = "shared"
GENERIC = "generic"

# The distinctive class: the word sits in at most this many of the labelled
# tables (the corpus map's own wording: "one or two tables").
DISTINCTIVE_MAX_TABLES = 2

# The generic class: the word sits in at least this **share** of the labelled
# tables — a share, not a magic number, so the threshold follows the corpus's
# size (a quarter of 60 tables is 15).
GENERIC_SHARE = 0.25

# The separation test's margin: the distinctive class's mean log-odds must
# exceed the generic class's by at least this many natural-log units (a factor
# of e in the odds).
SEPARATION_MARGIN = 1.0


def generic_threshold(n_labelled_tables: int) -> int:
    """How many labelled tables make a word **generic**.

    At least a quarter of the labelled tables (the plan's wording: "a share, not
    a magic number"). The floor is the distinctive class's ceiling plus one, so
    the classes never overlap on a small corpus.

    Args:
        n_labelled_tables: How many labels have a table (the labelled tables).

    Returns:
        The threshold (the smallest number of tables that makes a word generic).

    Raises:
        ValueError: On a negative number of tables.
    """
    if n_labelled_tables < 0:
        raise ValueError(
            f"the number of labelled tables must be non-negative, got {n_labelled_tables}"
        )
    return max(DISTINCTIVE_MAX_TABLES + 1, math.ceil(n_labelled_tables * GENERIC_SHARE))


def classify_word(n_tables: int, n_labelled_tables: int) -> str:
    """The class of a word that sits in ``n_tables`` of the labelled tables.

    Args:
        n_tables: How many of the labelled tables the word sits in.
        n_labelled_tables: How many labels have a table (the labelled tables).

    Returns:
        ``"distinctive"``, ``"shared"`` or ``"generic"``.

    Raises:
        ValueError: If the word sits in no table (it has no class) or the table
            count is negative.
    """
    if n_tables < 0:
        raise ValueError(f"the number of tables must be non-negative, got {n_tables}")
    if n_tables == 0:
        raise ValueError("a word that sits in no table has no class")
    if n_tables <= DISTINCTIVE_MAX_TABLES:
        return DISTINCTIVE
    if n_tables >= generic_threshold(n_labelled_tables):
        return GENERIC
    return SHARED


def label_name_words(
    label: str,
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> frozenset[str]:
    """A label's own name words (the **name-word flag**'s vocabulary).

    The same tokenizer the corpus map's tables are keyed by, so the flag and
    the tables speak the same unit.

    Args:
        label: The label's name.
        stopwords: The tokenizer's stopwords.

    Returns:
        The label's content words.
    """
    return frozenset(content_tokens(label, stopwords))


@dataclass(frozen=True)
class WordClassIndex:
    """The map's word classes: every word's own class and table count.

    Attributes:
        n_labelled_tables: How many labels have a table (the labelled tables).
        table_counts: How many of the labelled tables each word sits in.
        classes: Each word's class.
    """

    n_labelled_tables: int
    table_counts: Mapping[str, int]
    classes: Mapping[str, str]

    @property
    def generic_threshold(self) -> int:
        """How many labelled tables make a word generic."""
        return generic_threshold(self.n_labelled_tables)

    def class_of(self, word: str) -> str | None:
        """A word's class, or ``None`` when the word sits in no table.

        Args:
            word: The word.

        Returns:
            The class, or ``None``.
        """
        return self.classes.get(word)

    def words_in(self, classes: Iterable[str]) -> frozenset[str]:
        """The words whose class is one of ``classes``.

        Args:
            classes: The classes to keep.

        Returns:
            The words (the map's own vocabulary, restricted).
        """
        wanted = frozenset(classes)
        return frozenset(word for word, name in self.classes.items() if name in wanted)

    def counts_by_class(self) -> dict[str, int]:
        """How many **words** each class holds."""
        counts = dict.fromkeys(WORD_CLASSES, 0)
        for name in self.classes.values():
            counts[name] = counts.get(name, 0) + 1
        return counts


def build_word_class_index(
    table_counts: Mapping[str, int],
    n_labelled_tables: int,
) -> WordClassIndex:
    """The word-class index of a corpus map.

    Args:
        table_counts: How many of the labelled tables each word sits in.
        n_labelled_tables: How many labels have a table.

    Returns:
        The index.

    Raises:
        ValueError: If the table count is negative.
    """
    classes: dict[str, str] = {}
    for word, count in table_counts.items():
        if count < 0:
            raise ValueError(f"the word {word!r} has a negative table count: {count}")
        if count:
            classes[word] = classify_word(count, n_labelled_tables)
    return WordClassIndex(
        n_labelled_tables=n_labelled_tables,
        table_counts=dict(table_counts),
        classes=classes,
    )


@dataclass(frozen=True)
class ClassReadout:
    """One class's readout: how its pairs sit against the sign.

    Attributes:
        n_pairs: How many pairs the class holds.
        share_positive: The share of the pairs whose value is **positive** (the
            word goes *with* the label).
        share_negative: The share of the pairs whose value is **negative** (it
            goes *against* it).
        mean_value: The mean value over the class's pairs.
    """

    n_pairs: int
    share_positive: float
    share_negative: float
    mean_value: float

    def to_row(self, value_name: str = "mean_value") -> dict[str, object]:
        """The readout as a JSON-friendly row.

        Args:
            value_name: The name the mean's key carries (``mean_log_odds`` for
                the corpus map, ``mean_coefficient`` for the surrogate).

        Returns:
            The row.
        """
        return {
            "n_pairs": self.n_pairs,
            "share_positive": round(self.share_positive, 6),
            "share_negative": round(self.share_negative, 6),
            value_name: round(self.mean_value, 6),
        }


def class_readout(
    values: Iterable[tuple[str, float]],
    classes: Sequence[str] = WORD_CLASSES,
) -> dict[str, ClassReadout]:
    """The readout of every class, over ``(class, value)`` pairs.

    Args:
        values: The pairs: the class and the value (the log-odds for the
            corpus map, the coefficient for the surrogate).
        classes: The class names to report (the map's own frequency classes by
            default; CS-34's grammatical classes pass their own names).

    Returns:
        One readout per class (every class is present, empty ones included).

    Raises:
        ValueError: On a class name the readout does not know.
    """
    buckets: dict[str, list[float]] = {name: [] for name in classes}
    for name, value in values:
        if name not in buckets:
            raise ValueError(f"Unknown word class {name!r}; expected one of {list(buckets)}")
        buckets[name].append(float(value))
    return {name: _readout(bucket) for name, bucket in buckets.items()}


def _readout(values: list[float]) -> ClassReadout:
    """One class's readout from its values."""
    if not values:
        return ClassReadout(n_pairs=0, share_positive=0.0, share_negative=0.0, mean_value=0.0)
    positive = sum(1 for value in values if value > 0)
    negative = sum(1 for value in values if value < 0)
    count = len(values)
    return ClassReadout(
        n_pairs=count,
        share_positive=positive / count,
        share_negative=negative / count,
        mean_value=sum(values) / count,
    )


def separation_test(
    readout: Mapping[str, ClassReadout],
    *,
    margin: float = SEPARATION_MARGIN,
    value_name: str = "mean_log_odds",
) -> dict[str, object]:
    """The registered falsification: do the classes separate?

    The test: the **distinctive** class's mean value must exceed the
    **generic** class's by at least ``margin`` (one natural-log unit — a factor
    of *e* in the odds) **and** the generic class's share of positive pairs
    must be lower. If either half fails, the readout is a **null**: the word
    class does not separate the label-carrying words from the register words.

    (CS-34's grammatical test is its own function,
    :func:`cluster_rag.text_processors.lexical.pos_class.pos_separation_test`:
    this block's shape is CS-33's, published, and is never reshaped.)

    Args:
        readout: The readout (from :func:`class_readout`).
        margin: The margin the mean values must differ by.
        value_name: The name the mean's key carries in the block.

    Returns:
        The test's block (JSON-friendly), with the ``holds`` flag.

    Raises:
        ValueError: If the readout does not carry the two classes the test
            reads.
    """
    distinctive = readout.get(DISTINCTIVE)
    generic = readout.get(GENERIC)
    if distinctive is None or generic is None:
        raise ValueError(
            f"the readout must carry the {DISTINCTIVE!r} and {GENERIC!r} classes, "
            f"got {sorted(readout)}"
        )
    if not distinctive.n_pairs or not generic.n_pairs:
        return {
            "holds": False,
            "margin": margin,
            "margin_met": False,
            "generic_share_positive_lower": False,
            "distinctive_mean": distinctive.mean_value,
            "generic_mean": generic.mean_value,
            "difference": 0.0,
            "note": (
                "one of the two classes holds no pair, so the classes cannot be "
                "separated: the readout is a null"
            ),
        }
    difference = distinctive.mean_value - generic.mean_value
    margin_met = difference >= margin
    share_lower = generic.share_positive < distinctive.share_positive
    return {
        "holds": bool(margin_met and share_lower),
        "margin": margin,
        "margin_met": bool(margin_met),
        "generic_share_positive_lower": bool(share_lower),
        "distinctive_mean": round(distinctive.mean_value, 6),
        "generic_mean": round(generic.mean_value, 6),
        "difference": round(difference, 6),
        "distinctive_share_positive": round(distinctive.share_positive, 6),
        "generic_share_positive": round(generic.share_positive, 6),
        "distinctive_n_pairs": distinctive.n_pairs,
        "generic_n_pairs": generic.n_pairs,
        "value_name": value_name,
        "note": (
            "the readout is a null unless the distinctive class's mean exceeds the "
            "generic class's by at least the margin AND the generic class's share "
            "of positive pairs is lower"
        ),
    }
