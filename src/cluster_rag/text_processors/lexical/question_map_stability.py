"""The question map's **stability** (the track's CS-54): which links survive resampling.

The question map (CS-48) is the reviewed pairs' own word↔question association:
27,979 (question, word) links over 294 questions. CS-36 measured the UPR map's
fragility the same way and found the filter a null and the readout the
deliverable; this module ports the machinery to the **pair unit**, unchanged in
its statistic and its meaning.

**The statistic.** For every (question, unit) link of the **full** map, three
shares of the resamples:

- ``present_share`` — the link is in the question's table at all (the unit
  occurs in at least one resampled pair that carries the question);
- ``sign_stability`` — the link is there **and keeps its sign**: the sign of
  its log-odds is the sign it has in the full map. **This is the statistic the
  filter uses** — a link whose sign flips from sample to sample is not a link;
- ``head_stability`` — the link is there **and keeps its place among the
  question's head words**: it is still among the question's top ``head_k`` by
  log-odds. Reported, never filtered on.

**The resampling.** The ordinary bootstrap over the **reviewed pairs**: each
resample draws ``n`` pairs uniformly with replacement from the ``n`` reviewed
pairs, and the map is rebuilt **from scratch** on the resample (the same
builder, the same tokenizer, the same question space). The resample is a
*corpus*, so a pair drawn twice counts as two pairs. The resamples are
**streamed**: one map is alive at a time.

**The filter.** A link survives at a threshold ``tau`` iff its
``sign_stability >= tau``; the dropped links leave the question's table
entirely. A question whose every link is dropped has **no evidence left** and
is treated exactly like a question with no positive pair (the scorer's floor).

**Why the numbers should look the way they do.** A link resting on ``k``
reviewed pairs survives about ``1 - e^(-k)`` of the resamples — a
single-pair link **≈ 63 %** (1 − e⁻¹), two pairs ≈ 86 %, three ≈ 95 %. On this
map 89.1 % of the links rest on one pair (CS-48's `n_with` readout), so the
registered prediction is that little of the map clears τ = 0.75.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS
from cluster_rag.text_processors.features.question_map import QuestionMap, build_question_map

# The head size: how many of a question's top units by log-odds count as "the
# question's head words" (the second statistic). K = 10 is CS-35's top-10
# convention, so the number is the same convention as the mined lexicon's.
HEAD_K = 10

# The survival curve's own grid: the readout reports the share of the map that
# survives at every threshold from 0.00 to 1.00 in steps of 0.05, so the
# readout itself is threshold-free and the registered thresholds are read off it.
SURVIVAL_CURVE_STEP = 0.05


@dataclass(frozen=True)
class QuestionLinkStability:
    """One (question, unit) link's stability over the resamples.

    Attributes:
        question: The question's text (the map's column key).
        word: The unit, as the map's tokenizer writes it.
        n_with: The question's reviewed pairs that are positive and carry the
            unit (the link's own support).
        df: The question's reviewed pairs that carry the unit at all.
        log_odds: The link's log-odds in the full map (the sign the stability
            is read against).
        z_score: The link's z-score in the full map (CS-50's measure, carried
            so the readout sits beside the arm's own weights).
        present_share: The share of resamples in which the link is in the
            question's table.
        sign_stability: The share of resamples in which the link is present
            **and keeps its sign**.
        head_stability: The share of resamples in which the link is present
            **and keeps its place among the question's head words**.
    """

    question: str
    word: str
    n_with: int
    df: int
    log_odds: float
    z_score: float
    present_share: float
    sign_stability: float
    head_stability: float

    def survives(self, threshold: float) -> bool:
        """Whether the link survives the filter at ``threshold``."""
        return self.sign_stability >= threshold

    def to_row(self) -> dict[str, object]:
        """The link as a JSON-friendly row (the artifact's own shape)."""
        return {
            "question": self.question,
            "word": self.word,
            "n_with": self.n_with,
            "df": self.df,
            "log_odds": round(self.log_odds, 6),
            "z_score": round(self.z_score, 6),
            "present_share": round(self.present_share, 6),
            "sign_stability": round(self.sign_stability, 6),
            "head_stability": round(self.head_stability, 6),
        }


@dataclass(frozen=True)
class QuestionMapStability:
    """The map's stability over the resamples: every link's own numbers.

    Attributes:
        resamples: How many resamples the estimate rests on.
        seed: The resampling's seed.
        head_k: The head size (how many of a question's top units count as its
            head words).
        n_pairs: The reviewed pairs the map was built from (the resample's size
            too).
        questions: The map's question space (the silent ones included).
        links: Every link of the full map, in the map's own order (each
            question's table by descending log-odds).
    """

    resamples: int
    seed: int
    head_k: int
    n_pairs: int
    questions: tuple[str, ...]
    links: tuple[QuestionLinkStability, ...]

    def survivors(self, threshold: float) -> dict[str, frozenset[str]]:
        """The links that survive the filter, per question.

        Args:
            threshold: The survival threshold (a share of the resamples).

        Returns:
            The question -> the surviving units mapping (the questions with no
            surviving link contribute no entry).
        """
        kept: dict[str, set[str]] = {}
        for link in self.links:
            if link.survives(threshold):
                kept.setdefault(link.question, set()).add(link.word)
        return {question: frozenset(words) for question, words in sorted(kept.items())}

    def question_survivors(self, threshold: float) -> dict[str, frozenset[str]]:
        """The surviving links of **every** question (the silent ones included)."""
        kept = self.survivors(threshold)
        return {question: kept.get(question, frozenset()) for question in self.questions}

    def surviving_links(self, threshold: float) -> int:
        """How many links survive the filter at ``threshold``."""
        return sum(1 for link in self.links if link.survives(threshold))

    def survival_curve(self) -> list[dict[str, Any]]:
        """The share of the map that survives at every threshold on the grid."""
        steps = int(round(1.0 / SURVIVAL_CURVE_STEP))
        total = len(self.links)
        rows = []
        for index in range(steps + 1):
            threshold = index * SURVIVAL_CURVE_STEP
            surviving = sum(1 for link in self.links if link.survives(threshold))
            rows.append(
                {
                    "threshold": round(threshold, 2),
                    "share": (surviving / total) if total else None,
                }
            )
        return rows

    def by_support(self) -> list[dict[str, Any]]:
        """The survival readout **by support**: links resting on 1, 2, 3 … pairs.

        The single-pair row is the cliff (the 1 − e⁻¹ prediction): a link seen
        once survives about 63 % of the resamples, so it sits below τ = 0.75.
        """
        buckets: dict[int, list[QuestionLinkStability]] = {}
        for link in self.links:
            buckets.setdefault(link.n_with, []).append(link)
        rows = []
        for support in sorted(buckets):
            group = buckets[support]
            rows.append(
                {
                    "n_with": support,
                    "links": len(group),
                    "mean_present_share": float(np.mean([link.present_share for link in group])),
                    "mean_sign_stability": float(np.mean([link.sign_stability for link in group])),
                    "share_surviving_075": sum(1 for link in group if link.survives(0.75))
                    / len(group),
                    "share_surviving_090": sum(1 for link in group if link.survives(0.90))
                    / len(group),
                }
            )
        return rows

    def per_question(self) -> list[dict[str, Any]]:
        """Each question's own stability: its size, its means, its survivors."""
        groups: dict[str, list[QuestionLinkStability]] = {}
        for link in self.links:
            groups.setdefault(link.question, []).append(link)
        rows = []
        for question in self.questions:
            group = groups.get(question, [])
            rows.append(
                {
                    "question": question,
                    "table_words": len(group),
                    "mean_present_share": (
                        float(np.mean([link.present_share for link in group])) if group else None
                    ),
                    "mean_sign_stability": (
                        float(np.mean([link.sign_stability for link in group])) if group else None
                    ),
                    "surviving_075": sum(1 for link in group if link.survives(0.75)),
                    "surviving_090": sum(1 for link in group if link.survives(0.90)),
                    "share_surviving_075": (
                        sum(1 for link in group if link.survives(0.75)) / len(group)
                        if group
                        else None
                    ),
                }
            )
        return rows

    def head_links(self) -> tuple[QuestionLinkStability, ...]:
        """The links that are among their question's head words in the full map."""
        head: list[QuestionLinkStability] = []
        seen = 0
        question = ""
        for link in self.links:
            if link.question != question:
                question = link.question
                seen = 0
            if seen < self.head_k:
                head.append(link)
                seen += 1
        return tuple(head)


def estimate_question_map_stability(
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    labels: Sequence[int],
    base_map: QuestionMap,
    *,
    resamples: int = 500,
    seed: int = 42,
    head_k: int = HEAD_K,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    units: Any = None,
) -> QuestionMapStability:
    """Estimate every link's stability by resampling the reviewed pairs.

    The corpus is whatever the caller passes — in the run, the **reviewed
    pairs** (subset A) or a fold's training pairs, so the control cells (subset
    B) never enter a resample. The map is rebuilt from scratch on every
    resample, so a link's counts, its log-odds and its place in the question's
    table are all re-earned each time.

    Args:
        paragraph_texts: Each reviewed pair's paragraph.
        question_texts: Each reviewed pair's question text.
        labels: Each pair's 0/1 label.
        base_map: The map the links are read against (the full map, built on
            the whole corpus handed in).
        resamples: How many bootstrap resamples to draw (at least one).
        seed: The resampling's seed (deterministic).
        head_k: How many of a question's top units count as its head words.
        stopwords: The tokenizer's stopwords.
        units: The map's unit (the phrase table passes the bigram tokenizer).

    Returns:
        The stability report: every link of the base map with its three shares.

    Raises:
        ValueError: If the inputs have different lengths, a label is not 0/1,
            or the request is degenerate.
    """
    if not (len(paragraph_texts) == len(question_texts) == len(labels)):
        raise ValueError(
            "the paragraphs, questions and labels must have the same length, got "
            f"{len(paragraph_texts)} / {len(question_texts)} / {len(labels)}"
        )
    if any(int(label) not in (0, 1) for label in labels):
        raise ValueError("the pair labels must be 0/1")
    if resamples < 1:
        raise ValueError(f"a stability estimate needs at least 1 resample, got {resamples}")
    n_pairs = len(labels)
    if not n_pairs:
        raise ValueError("a stability estimate needs a non-empty corpus")

    base_links: dict[str, tuple[tuple[str, float, float, int, int], ...]] = {
        question: tuple(
            (row.word, row.log_odds, row.z_score, row.n_with, row.df)
            for row in base_map.table_rows(question)
        )
        for question in base_map.questions
    }
    present: dict[tuple[str, str], int] = {}
    signed: dict[tuple[str, str], int] = {}
    headed: dict[tuple[str, str], int] = {}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        index = rng.integers(0, n_pairs, size=n_pairs)
        resampled = build_question_map(
            [paragraph_texts[int(position)] for position in index],
            [question_texts[int(position)] for position in index],
            [int(labels[int(position)]) for position in index],
            stopwords=stopwords,
            units=units,
        )
        for question, rows in base_links.items():
            table = resampled.table_rows(question)
            if not table:
                continue
            lookup = {row.word: (row.log_odds, rank) for rank, row in enumerate(table)}
            for word, log_odds, _z, _n_with, _df in rows:
                entry = lookup.get(word)
                if entry is None:
                    continue
                key = (question, word)
                present[key] = present.get(key, 0) + 1
                value, rank = entry
                if _sign(value) == _sign(log_odds):
                    signed[key] = signed.get(key, 0) + 1
                if rank < head_k:
                    headed[key] = headed.get(key, 0) + 1

    links: list[QuestionLinkStability] = []
    for question, rows in base_links.items():
        for word, log_odds, z_score, n_with, df in rows:
            key = (question, word)
            links.append(
                QuestionLinkStability(
                    question=question,
                    word=word,
                    n_with=n_with,
                    df=df,
                    log_odds=log_odds,
                    z_score=z_score,
                    present_share=present.get(key, 0) / resamples,
                    sign_stability=signed.get(key, 0) / resamples,
                    head_stability=headed.get(key, 0) / resamples,
                )
            )
    return QuestionMapStability(
        resamples=resamples,
        seed=seed,
        head_k=head_k,
        n_pairs=n_pairs,
        questions=tuple(base_map.questions),
        links=tuple(links),
    )


def _sign(value: float) -> int:
    """The sign of a number (-1, 0 or 1)."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def head_retention(stability: QuestionMapStability, *, threshold: float = 0.5) -> dict[str, Any]:
    """How many of the map's head links keep their place in the resamples."""
    heads = stability.head_links()
    kept = sum(1 for link in heads if link.head_stability >= threshold)
    return {
        "head_links": len(heads),
        "kept": kept,
        "share": (kept / len(heads)) if heads else None,
        "threshold": threshold,
    }


def stability_block(stability: QuestionMapStability) -> dict[str, Any]:
    """The stability readout (CS-54): the artifact's own block.

    The block carries the statistic's definitions, the knobs, the **survival
    curve**, the survival **by support** (the single-pair cliff), the
    **per-question** stability, the **head retention**, and every link's three
    shares, so any threshold can be read off the table.

    Args:
        stability: The stability report.

    Returns:
        A JSON-serializable block.
    """
    curve = stability.survival_curve()
    by_support = stability.by_support()
    single = next((row for row in by_support if row["n_with"] == 1), None)
    return {
        "resamples": stability.resamples,
        "seed": stability.seed,
        "head_k": stability.head_k,
        "n_pairs": stability.n_pairs,
        "n_links": len(stability.links),
        "n_questions": len(stability.questions),
        "statistics": {
            "present_share": "the share of resamples in which the link is in the question's table",
            "sign_stability": (
                "present AND keeping the full map's log-odds sign (the filter's statistic)"
            ),
            "head_stability": "present AND still among the question's top head_k by log-odds",
        },
        "prediction": {
            "single_pair_survival": 1.0 - np.exp(-1.0),
            "theory": (
                "a link resting on k reviewed pairs survives about 1 - e^(-k) of the resamples"
            ),
        },
        "survival_curve": curve,
        "by_support": by_support,
        "single_pair": single,
        "head_retention": head_retention(stability),
        "per_question": stability.per_question(),
        "links": [link.to_row() for link in stability.links],
        "note": (
            "the bootstrap is over the reviewed PAIRS (subset A) — the pair is the row; "
            "each fold's filter in the run is estimated on that fold's own training "
            "pairs, so a judged pair is scored by a filter that never saw it"
        ),
    }


def stability_filter(
    stability: QuestionMapStability, threshold: float
) -> dict[str, frozenset[str]]:
    """The filter's own shape: question -> the units that survive ``threshold``.

    It is the ``keep`` argument the scorer takes, so a filtered arm is the same
    scoring code reading a reduced map (the same rule as CS-36's filter).
    """
    return stability.question_survivors(threshold)


def kept_share(stability: QuestionMapStability, threshold: float) -> float | None:
    """The share of the map's links that survives at ``threshold``."""
    if not stability.links:
        return None
    return stability.surviving_links(threshold) / len(stability.links)


__all__ = [
    "HEAD_K",
    "SURVIVAL_CURVE_STEP",
    "QuestionLinkStability",
    "QuestionMapStability",
    "estimate_question_map_stability",
    "head_retention",
    "kept_share",
    "stability_block",
    "stability_filter",
]

# The mapping's type is exported for the callers' annotations.
FilterMap = Mapping[str, frozenset[str]]
