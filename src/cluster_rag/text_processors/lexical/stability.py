"""The corpus map's **stability** (the track's CS-36): which links survive
resampling.

The corpus map (CS-23) is the **train** split's own word↔label association:
18,707 (label, word) pairs over 3,049 content words. Its bulk is fragile —
two thirds of the pairs rest on a **single** train text — and until now
nothing has measured how much of it is **sampling noise**. CS-31's z-score is
an *asymptotic* repair (a standard error over a table); this module is the
**empirical** one: resample the corpus, rebuild the map on each resample, and
read every pair's own survival.

**The statistic.** For every (label, word) pair of the **full** map, three
shares of the resamples:

- ``present_share`` — the pair is in the label's table at all (the word occurs
  in at least one resampled text that carries the label);
- ``sign_stability`` — the pair is there **and keeps its sign**: the sign of
  its log-odds is the sign it has in the full map. **This is the statistic the
  filter uses** — a link whose sign flips from sample to sample is not a link,
  it is a coin;
- ``head_stability`` — the pair is there **and keeps its place among the
  label's head words**: it is still among the label's top ``head_k`` words by
  log-odds. This one is **reported, not filtered on**: it answers the separate
  question ("does the map's own ordering survive?").

**The resampling.** The ordinary bootstrap: each resample draws ``n`` texts
uniformly with replacement from the corpus's ``n`` texts, and the map is
rebuilt **from scratch** on the resample (the same builder, the same
tokenizer, the same label space). The resample is a *corpus*, so a text drawn
twice counts as two documents. The resamples are **streamed**: one map is
alive at a time, so the memory does not grow with the number of resamples.

**The filter.** A pair survives at a threshold ``tau`` iff its
``sign_stability >= tau``; the dropped pairs leave the label's table
entirely. A label whose every pair is dropped has **no evidence left** and is
treated exactly like a label with no train example (the scorer's floor).

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cluster_rag.text_processors.features.corpus_map import CorpusMap, build_corpus_map
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS

# The head size: how many of a label's top words by log-odds count as "the
# label's head words" (the second statistic). K = 10 is CS-35's top-10
# convention, so the number is the same convention as the mined lexicon's.
HEAD_K = 10

# The survival curve's own grid: the readout reports the share of the map that
# survives at every threshold from 0.00 to 1.00 in steps of 0.05, so the readout
# itself is threshold-free and the registered thresholds are read off it.
SURVIVAL_CURVE_STEP = 0.05


@dataclass(frozen=True)
class PairStability:
    """One (label, word) pair's stability over the resamples.

    Attributes:
        label: The label's name.
        word: The word, as the track's own content-word tokenizer writes it.
        n_with: The corpus texts that have **both** the word and the label (the
            pair's own support — the map's own number).
        df: The corpus texts that have the word at all.
        log_odds: The pair's log-odds in the full map (the sign the stability
            is read against).
        present_share: The share of resamples in which the pair is in the
            label's table.
        sign_stability: The share of resamples in which the pair is present
            **and keeps its sign**.
        head_stability: The share of resamples in which the pair is present
            **and keeps its place among the label's head words**.
    """

    label: str
    word: str
    n_with: int
    df: int
    log_odds: float
    present_share: float
    sign_stability: float
    head_stability: float

    def survives(self, threshold: float) -> bool:
        """Whether the pair survives the filter at ``threshold``.

        Args:
            threshold: The survival threshold (a share of the resamples).

        Returns:
            True when the pair's ``sign_stability`` reaches the threshold.
        """
        return self.sign_stability >= threshold

    def to_row(self) -> dict[str, object]:
        """The pair as a JSON-friendly row (the artifact's own shape)."""
        return {
            "word": self.word,
            "n_with": self.n_with,
            "df": self.df,
            "log_odds": round(self.log_odds, 6),
            "present_share": round(self.present_share, 6),
            "sign_stability": round(self.sign_stability, 6),
            "head_stability": round(self.head_stability, 6),
        }


@dataclass(frozen=True)
class StabilityReport:
    """The map's stability over the resamples: every pair's own numbers.

    Attributes:
        resamples: How many resamples the estimate rests on.
        seed: The resampling's seed.
        head_k: The head size (how many of a label's top words count as its
            head words).
        n_texts: The corpus's size (the resample's size too).
        labels: The map's label space (every label, the ones with no example
            included).
        pairs: Every pair of the full map, with its three shares.
    """

    resamples: int
    seed: int
    head_k: int
    n_texts: int
    labels: tuple[str, ...]
    pairs: tuple[PairStability, ...]

    def survivors(self, threshold: float) -> dict[str, frozenset[str]]:
        """The pairs that survive the filter, per label.

        Args:
            threshold: The survival threshold (a share of the resamples).

        Returns:
            The label -> the surviving words mapping.
        """
        kept: dict[str, set[str]] = {}
        for pair in self.pairs:
            if pair.survives(threshold):
                kept.setdefault(pair.label, set()).add(pair.word)
        return {label: frozenset(words) for label, words in sorted(kept.items())}

    def label_survivors(self, threshold: float) -> dict[str, frozenset[str]]:
        """The surviving pairs of **every** label (the empty ones included).

        Args:
            threshold: The survival threshold (a share of the resamples).

        Returns:
            The label -> the surviving words mapping, with an empty set for
            every label that has no surviving pair.
        """
        kept = self.survivors(threshold)
        return {label: kept.get(label, frozenset()) for label in self.labels}

    def surviving_pairs(self, threshold: float) -> int:
        """How many pairs survive the filter at ``threshold``."""
        return sum(1 for pair in self.pairs if pair.survives(threshold))

    def head_pairs(self) -> tuple[PairStability, ...]:
        """The pairs that are among their label's **head words** in the full map.

        The report's pairs are in the map's own order (each label's table by
        descending log-odds), so the first ``head_k`` pairs of each label are
        its head words.

        Returns:
            The head pairs (the labels with no table contribute none).
        """
        head: list[PairStability] = []
        seen = 0
        label = ""
        for pair in self.pairs:
            if pair.label != label:
                label = pair.label
                seen = 0
            if seen < self.head_k:
                head.append(pair)
                seen += 1
        return tuple(head)


def estimate_stability(
    texts: Sequence[str],
    label_sets: Sequence[Sequence[str]],
    base_map: CorpusMap,
    *,
    resamples: int = 500,
    seed: int = 42,
    head_k: int = HEAD_K,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> StabilityReport:
    """Estimate every pair's stability by resampling the corpus.

    The corpus is whatever the caller passes — the track's **train** split, so
    no val text and no val label enters a resample and ``test`` is not touched
    at all. The map is rebuilt **from scratch** on every resample, so a pair's
    counts, its log-odds and its place in the label's table are all re-earned
    each time.

    Args:
        texts: The corpus's texts.
        label_sets: Each text's labels, in the same order as ``texts``.
        base_map: The map the pairs are read against (the full map, built on
            the whole corpus).
        resamples: How many bootstrap resamples to draw (at least one).
        seed: The resampling's seed (deterministic).
        head_k: The head size (how many of a label's top words count as its
            head words).
        stopwords: The tokenizer's stopwords — the map's own, so the words
            match the tables.

    Returns:
        The stability report: every pair of the base map with its three shares.

    Raises:
        ValueError: If the texts and the label sets have different lengths, or
            if the request is degenerate (no resample, no corpus).
    """
    if len(texts) != len(label_sets):
        raise ValueError(f"the corpus has {len(texts)} texts and {len(label_sets)} label sets")
    if resamples < 1:
        raise ValueError(f"a stability estimate needs at least 1 resample, got {resamples}")
    if not texts:
        raise ValueError("a stability estimate needs a non-empty corpus")
    n_texts = len(texts)
    rng = np.random.default_rng(seed)
    base_rows: dict[str, list[tuple[str, float, int, int]]] = {}
    for label in base_map.labels:
        rows = base_map.topic_rows(label)
        if rows:
            base_rows[label] = [(row.word, row.log_odds, row.n_with, row.df) for row in rows]
    present: dict[tuple[str, str], int] = {}
    signed: dict[tuple[str, str], int] = {}
    headed: dict[tuple[str, str], int] = {}
    for _ in range(resamples):
        index = rng.integers(0, n_texts, size=n_texts)
        resampled = build_corpus_map(
            [texts[int(position)] for position in index],
            [label_sets[int(position)] for position in index],
            base_map.labels,
            stopwords=stopwords,
        )
        for label, rows in base_rows.items():
            table = resampled.topic_rows(label)
            if not table:
                continue
            lookup = {row.word: (row.log_odds, rank) for rank, row in enumerate(table)}
            for word, log_odds, _n_with, _df in rows:
                entry = lookup.get(word)
                if entry is None:
                    continue
                key = (label, word)
                present[key] = present.get(key, 0) + 1
                value, rank = entry
                if _sign(value) == _sign(log_odds):
                    signed[key] = signed.get(key, 0) + 1
                if rank < head_k:
                    headed[key] = headed.get(key, 0) + 1
    pairs = []
    for label, rows in base_rows.items():
        for word, log_odds, n_with, df in rows:
            key = (label, word)
            pairs.append(
                PairStability(
                    label=label,
                    word=word,
                    n_with=n_with,
                    df=df,
                    log_odds=log_odds,
                    present_share=present.get(key, 0) / resamples,
                    sign_stability=signed.get(key, 0) / resamples,
                    head_stability=headed.get(key, 0) / resamples,
                )
            )
    return StabilityReport(
        resamples=resamples,
        seed=seed,
        head_k=head_k,
        n_texts=n_texts,
        labels=tuple(base_map.labels),
        pairs=tuple(pairs),
    )


def _sign(value: float) -> int:
    """The sign of a number (-1, 0 or 1)."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def stability_block(report: StabilityReport) -> dict[str, object]:
    """The map's **stability readout** (CS-36): the artifact's own block.

    The block carries the statistic's definitions, the knobs it was estimated
    with, the **survival curve** (the share of the map that survives at every
    threshold), the survival share **by support** (the pairs resting on one,
    two, three … train texts — the reading of the map's fragility), the
    **per-label** stability (each label's table before and after the filter)
    and **every** pair's three shares, so any threshold can be read off the
    table.

    Args:
        report: The stability report.

    Returns:
        The block (JSON-friendly).
    """
    n_pairs = len(report.pairs)
    thresholds = [round(step * SURVIVAL_CURVE_STEP, 2) for step in range(21)]
    curve = []
    for threshold in thresholds:
        surviving = report.surviving_pairs(threshold)
        curve.append(
            {
                "threshold": threshold,
                "surviving_pairs": surviving,
                "share": round(surviving / n_pairs, 6) if n_pairs else 0.0,
            }
        )
    by_support = _survival_by_support(report, thresholds)
    head_pairs = report.head_pairs()
    per_label = []
    for label in report.labels:
        rows = [pair for pair in report.pairs if pair.label == label]
        label_head = [pair for pair in head_pairs if pair.label == label]
        surviving = {
            str(threshold): sum(1 for row in rows if row.survives(threshold))
            for threshold in thresholds
        }
        per_label.append(
            {
                "label": label,
                "n_pairs": len(rows),
                "has_a_table": bool(rows),
                "mean_present_share": round(sum(row.present_share for row in rows) / len(rows), 6)
                if rows
                else 0.0,
                "mean_sign_stability": round(sum(row.sign_stability for row in rows) / len(rows), 6)
                if rows
                else 0.0,
                "mean_head_stability": round(sum(row.head_stability for row in rows) / len(rows), 6)
                if rows
                else 0.0,
                "head_words": {
                    "n_words": len(label_head),
                    "mean_present_share": round(
                        sum(row.present_share for row in label_head) / len(label_head), 6
                    )
                    if label_head
                    else 0.0,
                    "mean_sign_stability": round(
                        sum(row.sign_stability for row in label_head) / len(label_head), 6
                    )
                    if label_head
                    else 0.0,
                    "mean_head_stability": round(
                        sum(row.head_stability for row in label_head) / len(label_head), 6
                    )
                    if label_head
                    else 0.0,
                },
                "surviving_pairs": surviving,
                "table_size_after": {
                    str(threshold): surviving[str(threshold)] for threshold in thresholds
                },
                "loses_its_table_at": (
                    [threshold for threshold in thresholds if surviving[str(threshold)] == 0]
                    if rows
                    else []
                ),
            }
        )
    return {
        "available": True,
        "definitions": {
            "present_share": (
                "the share of the resamples in which the pair is in the label's "
                "table at all (the word occurs in at least one resampled text "
                "that carries the label)"
            ),
            "sign_stability": (
                "the share of the resamples in which the pair is present AND keeps "
                "its sign (the sign of its log-odds is the sign it has in the full "
                "map). THIS is the statistic the filter uses"
            ),
            "head_stability": (
                "the share of the resamples in which the pair is present AND keeps "
                "its place among the label's head words (the top head_k words by "
                "log-odds). Reported, not filtered on. Read it on the HEAD WORDS "
                "themselves (the head_words block): over all the pairs it is "
                "near zero, because most pairs are never among the label's top "
                "head_k words"
            ),
            "resampling": (
                "the ordinary bootstrap: each resample draws the corpus's n texts "
                "uniformly with replacement and the map is rebuilt from scratch on "
                "it (a text drawn twice counts as two documents)"
            ),
            "note": (
                "the corpus is the train split only: no val text, no val label and "
                "no test text enters a resample. The estimate is Monte-Carlo: a "
                "pair's share carries a standard error of at most 0.5/sqrt(B)."
            ),
        },
        "knobs": {
            "resamples": report.resamples,
            "seed": report.seed,
            "head_k": report.head_k,
            "n_texts": report.n_texts,
            "monte_carlo_standard_error": round(0.5 / (report.resamples**0.5), 6),
        },
        "summary": {
            "n_pairs": n_pairs,
            "mean_present_share": round(
                sum(pair.present_share for pair in report.pairs) / n_pairs, 6
            )
            if n_pairs
            else 0.0,
            "mean_sign_stability": round(
                sum(pair.sign_stability for pair in report.pairs) / n_pairs, 6
            )
            if n_pairs
            else 0.0,
            "mean_head_stability": round(
                sum(pair.head_stability for pair in report.pairs) / n_pairs, 6
            )
            if n_pairs
            else 0.0,
            "share_pairs_never_present": round(
                sum(1 for pair in report.pairs if pair.present_share == 0.0) / n_pairs, 6
            )
            if n_pairs
            else 0.0,
            "head_words": {
                "n_words": len(head_pairs),
                "mean_present_share": round(
                    sum(pair.present_share for pair in head_pairs) / len(head_pairs), 6
                )
                if head_pairs
                else 0.0,
                "mean_sign_stability": round(
                    sum(pair.sign_stability for pair in head_pairs) / len(head_pairs), 6
                )
                if head_pairs
                else 0.0,
                "mean_head_stability": round(
                    sum(pair.head_stability for pair in head_pairs) / len(head_pairs), 6
                )
                if head_pairs
                else 0.0,
                "note": (
                    "the label's own top head_k words in the full map: the mean "
                    "head_stability here is the share of the resamples in which "
                    "those very words keep their place among the head"
                ),
            },
            "survival_curve": curve,
        },
        "survival_by_support": by_support,
        "per_label": per_label,
        "per_pair": [{"label": pair.label, **pair.to_row()} for pair in report.pairs],
    }


def _survival_by_support(
    report: StabilityReport,
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    """The survival share **by support**: the pairs resting on 1, 2, 3 … texts.

    The map's fragility is read off this table: a link that rests on a single
    train text is present in about 63 % of the resamples, one that rests on two
    in about 86 %.

    Args:
        report: The stability report.
        thresholds: The thresholds the survival shares are read at.

    Returns:
        One row per support bucket (the exact counts up to 5, then one bucket
        for everything above).
    """
    buckets: dict[str, list[PairStability]] = {}
    for pair in report.pairs:
        name = str(pair.n_with) if pair.n_with <= 5 else "6+"
        buckets.setdefault(name, []).append(pair)
    rows = []
    for name in sorted(buckets, key=lambda key: (key == "6+", key)):
        pairs = buckets[name]
        rows.append(
            {
                "n_with": name,
                "n_pairs": len(pairs),
                "mean_present_share": round(
                    sum(pair.present_share for pair in pairs) / len(pairs), 6
                ),
                "mean_sign_stability": round(
                    sum(pair.sign_stability for pair in pairs) / len(pairs), 6
                ),
                "mean_head_stability": round(
                    sum(pair.head_stability for pair in pairs) / len(pairs), 6
                ),
                "surviving_pairs": {
                    str(threshold): sum(1 for pair in pairs if pair.survives(threshold))
                    for threshold in thresholds
                },
            }
        )
    return rows


def surviving_filter(
    report: StabilityReport,
    threshold: float,
) -> Mapping[str, frozenset[str]]:
    """The filter the scorer takes: the label -> the surviving words mapping.

    Args:
        report: The stability report.
        threshold: The survival threshold.

    Returns:
        The mapping (every label of the map, the empty ones included).
    """
    return report.label_survivors(threshold)
