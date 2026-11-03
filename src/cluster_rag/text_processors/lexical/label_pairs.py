"""The **label-pair relation** (the track's CS-38): the label space's own geometry.

The corpus map (CS-23) is a relation between **labels and words**. Read the
other way round, it is a relation between **labels**: two labels that share
their vocabulary are near-duplicates, and two labels that keep appearing in the
same train texts are neighbours. This module computes that relation and reads
it out.

**The two halves.** For every pair of labels:

- the **vocabulary** half — the shared words' count, the **Jaccard**
  (``|T₁ ∩ T₂| / |T₁ ∪ T₂|``) and the **containment** (the share of the
  *smaller* table that lies inside the larger one);
- the **co-occurrence** half — the number of **train** texts carrying both
  labels, the **lift** over independence (the observed count over the expected
  one, ``n_a · n_b / N``) and the Jaccard of the two labels' text sets.

**The catalog's relation.** The committed label catalog (CS-11) carries a
**parent** per label — 12 parents over the 68 entries, and none of the 68 is
itself a parent. The parents are therefore *types*, not labels, and the
measurable form of the taxonomy's own claim about the label space is the
**sibling** relation: the pairs of labels that share a parent. The relation
carries that flag per pair, so the readout can compare the corpus's own
geometry with the taxonomy's.

**The readout** answers three questions: which labels are **near-duplicates**,
which pairs are **neighbours**, and which labels have **no vocabulary of their
own** (an empty table, or a table that lies almost entirely inside a
neighbour's).

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cluster_rag.text_processors.features.corpus_map import CorpusMap, build_corpus_map
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS

# The **near-duplicate** threshold: the vocabulary Jaccard at which two labels
# count as lexical twins (the pre-registered value).
NEAR_DUPLICATE_JACCARD = 0.50

# The **containment** threshold: the share of a label's own table that must lie
# inside one partner's table for the label to count as having **no vocabulary
# of its own** (the pre-registered value).
CONTAINMENT_THRESHOLD = 0.80

# How many pairs the readout lists as the **neighbours** (the strongest
# co-occurrence lifts).
N_NEIGHBOURS = 20


@dataclass(frozen=True)
class LabelPair:
    """One pair of labels and the corpus's own relation between them.

    Attributes:
        label_a: The first label.
        label_b: The second label (the pair is unordered; ``label_a`` sorts
            first).
        shared_words: How many words the two tables share.
        jaccard: The vocabulary Jaccard (``|∩| / |∪|``).
        containment: The share of the smaller table that lies inside the larger.
        n_both: The train texts carrying both labels.
        lift: The co-occurrence lift (the observed count over the expected one
            under independence).
        text_jaccard: The Jaccard of the two labels' train-text sets.
        sibling: Whether the two labels **share a catalog parent** (the
            taxonomy's own claim about the label space).
    """

    label_a: str
    label_b: str
    shared_words: int
    jaccard: float
    containment: float
    n_both: int
    lift: float
    text_jaccard: float
    sibling: bool

    def to_row(self) -> dict[str, object]:
        """The pair as a JSON-friendly row (the artifact's own shape)."""
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "shared_words": self.shared_words,
            "jaccard": round(self.jaccard, 6),
            "containment": round(self.containment, 6),
            "n_both": self.n_both,
            "lift": round(self.lift, 6),
            "text_jaccard": round(self.text_jaccard, 6),
            "sibling": self.sibling,
        }


@dataclass(frozen=True)
class LabelSummary:
    """One label's own place in the relation.

    Attributes:
        label: The label.
        n_words: The size of the label's word table (0 when it has none).
        n_corpus_texts: The train texts carrying the label.
        best_partner: The label whose table overlaps this one's the most (the
            Jaccard), or ``""`` when the label has no table.
        best_jaccard: That overlap's Jaccard.
        best_containment: The share of **this label's own** table that lies
            inside one partner's table (the largest over the partners).
        best_containment_partner: The partner that holds that share.
        n_near_duplicates: How many partners reach the near-duplicate
            threshold.
        n_co_occurring: How many partners occur with this label in at least one
            train text.
        sibling: The label's catalog parent (the taxonomy's own type).
    """

    label: str
    n_words: int
    n_corpus_texts: int
    best_partner: str
    best_jaccard: float
    best_containment: float
    best_containment_partner: str
    n_near_duplicates: int
    n_co_occurring: int
    sibling: str

    def to_row(self) -> dict[str, object]:
        """The summary as a JSON-friendly row (the artifact's own shape)."""
        return {
            "label": self.label,
            "n_words": self.n_words,
            "n_corpus_texts": self.n_corpus_texts,
            "best_partner": self.best_partner,
            "best_jaccard": round(self.best_jaccard, 6),
            "best_containment": round(self.best_containment, 6),
            "best_containment_partner": self.best_containment_partner,
            "n_near_duplicates": self.n_near_duplicates,
            "n_co_occurring": self.n_co_occurring,
            "sibling": self.sibling,
        }


@dataclass(frozen=True)
class LabelPairRelation:
    """The label space's own geometry, read off the corpus map.

    Attributes:
        labels: The label space (every label, the ones with no table included).
        pairs: Every pair of labels (unordered), sorted by label.
        per_label: Each label's own place in the relation.
        n_texts: The corpus's size (the train split).
    """

    labels: tuple[str, ...]
    pairs: tuple[LabelPair, ...]
    per_label: tuple[LabelSummary, ...]
    n_texts: int


def label_pair_relation(
    corpus_map: CorpusMap,
    label_sets: Sequence[Sequence[str]],
    *,
    parents: Mapping[str, str] | None = None,
) -> LabelPairRelation:
    """The label-pair relation, read off the corpus map and the train texts.

    Args:
        corpus_map: The corpus map (the train split's word↔label association).
        label_sets: The train records' label sets (the co-occurrence's source).
        parents: The labels' catalog parents (the taxonomy's own relation);
            ``None`` means no label has a parent.

    Returns:
        The relation: every pair of labels and each label's own place in it.

    Raises:
        ValueError: If the corpus map and the label sets disagree in size.
    """
    if corpus_map.n_texts != len(label_sets):
        raise ValueError(
            f"the map's corpus has {corpus_map.n_texts} texts and the label sets have "
            f"{len(label_sets)}"
        )
    catalog = dict(parents or {})
    labels = tuple(corpus_map.labels)
    tables: dict[str, frozenset[str]] = {
        label: frozenset(pair.word for pair in corpus_map.topic_rows(label)) for label in labels
    }
    texts: dict[str, set[int]] = {label: set() for label in labels}
    for index, record_labels in enumerate(label_sets):
        for label in set(record_labels):
            if label in texts:
                texts[label].add(index)
    n_texts = max(len(label_sets), 1)
    counts = {label: len(rows) for label, rows in texts.items()}
    pairs: list[LabelPair] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            label_a, label_b = labels[first], labels[second]
            table_a, table_b = tables[label_a], tables[label_b]
            shared = len(table_a & table_b)
            union = len(table_a | table_b)
            smaller = min(len(table_a), len(table_b))
            both = len(texts[label_a] & texts[label_b])
            union_texts = len(texts[label_a] | texts[label_b])
            expected = counts[label_a] * counts[label_b] / n_texts
            pairs.append(
                LabelPair(
                    label_a=label_a,
                    label_b=label_b,
                    shared_words=shared,
                    jaccard=(shared / union) if union else 0.0,
                    containment=(shared / smaller) if smaller else 0.0,
                    n_both=both,
                    lift=(both / expected) if expected else 0.0,
                    text_jaccard=(both / union_texts) if union_texts else 0.0,
                    sibling=(
                        bool(catalog.get(label_a)) and catalog.get(label_a) == catalog.get(label_b)
                    ),
                )
            )
    per_label: list[LabelSummary] = []
    for label in labels:
        partners = [pair for pair in pairs if label in (pair.label_a, pair.label_b)]
        table = tables[label]
        best = max(partners, key=lambda pair: pair.jaccard, default=None)
        # a partner that shares no word is not a partner: the best one is
        # reported only when the overlap is non-zero.
        if best is not None and best.jaccard <= 0.0:
            best = None
        own_containment = 0.0
        own_partner = ""
        for pair in partners:
            other = pair.label_b if pair.label_a == label else pair.label_a
            share = (pair.shared_words / len(table)) if table else 0.0
            if share > own_containment:
                own_containment = share
                own_partner = other
        per_label.append(
            LabelSummary(
                label=label,
                n_words=len(table),
                n_corpus_texts=counts.get(label, 0),
                best_partner=(
                    (best.label_b if best.label_a == label else best.label_a) if best else ""
                ),
                best_jaccard=best.jaccard if best else 0.0,
                best_containment=own_containment,
                best_containment_partner=own_partner,
                n_near_duplicates=sum(
                    1 for pair in partners if pair.jaccard >= NEAR_DUPLICATE_JACCARD
                ),
                n_co_occurring=sum(1 for pair in partners if pair.n_both > 0),
                sibling=catalog.get(label, ""),
            )
        )
    return LabelPairRelation(
        labels=labels,
        pairs=tuple(pairs),
        per_label=tuple(per_label),
        n_texts=len(label_sets),
    )


def containment_partners(
    corpus_map: CorpusMap,
    *,
    threshold: float = CONTAINMENT_THRESHOLD,
) -> dict[str, tuple[str, float]]:
    """The labels whose table lies (almost) entirely inside **one** partner's.

    The map-only form of CS-38's class (no label sets needed): for every label
    with a non-empty table, the share of its own table that lies inside each
    other label's table; the label joins the class when its **best** share is at
    or above the threshold, and its partner is that best one (the first maximal
    in the map's label order, the same tie-break the relation's readout uses).

    It is the **same computation** as
    :func:`vocabulary_less_labels`' ``subset_of_a_neighbour`` — that readout
    works from the relation's per-label summaries, this one from the map alone,
    so a caller that has the map (the scorer) can ask without building the
    relation.

    Args:
        corpus_map: The corpus map (the train split's word↔label association).
        threshold: The share of a label's own table that must lie inside one
            partner's table (CS-38's pre-registered 0.80).

    Returns:
        The label -> (partner, share) mapping, empty labels excluded.

    Raises:
        ValueError: If the threshold is not in ``(0, 1]``.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"the containment threshold must be in (0, 1], got {threshold!r}")
    labels = tuple(corpus_map.labels)
    tables = {
        label: frozenset(pair.word for pair in corpus_map.topic_rows(label)) for label in labels
    }
    out: dict[str, tuple[str, float]] = {}
    for label in labels:
        table = tables[label]
        if not table:
            continue
        best_share = 0.0
        best_partner = ""
        for other in labels:
            if other == label:
                continue
            share = len(table & tables[other]) / len(table)
            if share > best_share:
                best_share, best_partner = share, other
        if best_partner and best_share >= threshold:
            out[label] = (best_partner, best_share)
    return out


def vocabulary_less_labels(
    relation: LabelPairRelation,
    *,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> dict[str, object]:
    """The labels with **no vocabulary of their own**.

    The class has two members: the labels with an **empty** table (the eight
    held-out ones, by construction) and the labels whose table lies (almost)
    entirely inside **one** partner's table (the pre-registered containment
    threshold).

    Args:
        relation: The label-pair relation.
        containment_threshold: The share of a label's own table that must lie
            inside one partner's table.

    Returns:
        The block: the empty-table labels, the subset labels (with their
        partner and their share) and the union.
    """
    empty = [row.label for row in relation.per_label if row.n_words == 0]
    subset = [
        {
            "label": row.label,
            "partner": row.best_containment_partner,
            "share": round(row.best_containment, 6),
        }
        for row in relation.per_label
        if row.n_words and row.best_containment >= containment_threshold
    ]
    return {
        "definition": (
            "a label has no vocabulary of its own when its table is empty, or "
            "when almost all of its own words lie inside one other label's table"
        ),
        "empty_table": empty,
        "subset_of_a_neighbour": subset,
        "all": sorted(set(empty) | {row["label"] for row in subset}),
    }


def separation_bootstrap(
    texts: Sequence[str],
    label_sets: Sequence[Sequence[str]],
    labels: Sequence[str],
    *,
    parents: Mapping[str, str] | None = None,
    resamples: int = 2000,
    seed: int = 42,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, object]:
    """The **separation** between the catalog's sibling pairs and the rest.

    The resampling unit is the **text**: each resample draws the corpus's
    texts with replacement, the map is rebuilt on it, and the relation's
    vocabulary Jaccards are recomputed. The interval is over the corpus
    itself, so it says how stable the comparison is.

    Args:
        texts: The corpus's texts (the train split).
        label_sets: Each text's labels, in the same order.
        labels: The label space (the map's columns).
        parents: The labels' catalog parents (the sibling relation).
        resamples: How many bootstrap resamples to draw.
        seed: The resampling's seed (deterministic).
        stopwords: The tokenizer's stopwords — the map's own.

    Returns:
        The block: the observed difference and its interval.
    """
    n_texts = len(texts)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, n_texts, size=n_texts)
        corpus_map = build_corpus_map(
            [texts[int(position)] for position in index],
            [label_sets[int(position)] for position in index],
            labels,
            stopwords=stopwords,
        )
        relation = label_pair_relation(corpus_map, label_sets, parents=parents)
        siblings = [pair.jaccard for pair in relation.pairs if pair.sibling]
        others = [pair.jaccard for pair in relation.pairs if not pair.sibling]
        if siblings and others:
            differences.append(sum(siblings) / len(siblings) - sum(others) / len(others))
    if not differences:
        return {"available": False, "note": "the relation has no comparable pairs"}
    differences.sort()
    lower = differences[int(0.025 * (len(differences) - 1))]
    upper = differences[int(0.975 * (len(differences) - 1))]
    return {
        "available": True,
        "metric": "the sibling pairs' mean vocabulary Jaccard minus the other pairs'",
        "resamples": resamples,
        "seed": seed,
        "n_texts": n_texts,
        "point": round(sum(differences) / len(differences), 6),
        "ci_low": round(lower, 6),
        "ci_high": round(upper, 6),
        "note": (
            "the resampling unit is the text: the map is rebuilt on every "
            "resample, so the interval is over the corpus itself"
        ),
    }


def label_pairs_block(
    relation: LabelPairRelation,
    *,
    near_duplicate: float = NEAR_DUPLICATE_JACCARD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
    n_neighbours: int = N_NEIGHBOURS,
    separation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The label-pair readout (CS-38): the artifact's own block.

    Args:
        relation: The label-pair relation.
        near_duplicate: The vocabulary Jaccard at which two labels count as
            near-duplicates.
        containment_threshold: The share of a label's own table that must lie
            inside one partner's table.
        n_neighbours: How many pairs the readout lists as the neighbours.
        separation: The separation bootstrap's block (when it was computed).

    Returns:
        The block (JSON-friendly).
    """
    pairs = relation.pairs
    near = sorted(
        (pair for pair in pairs if pair.jaccard >= near_duplicate),
        key=lambda pair: (-pair.jaccard, pair.label_a, pair.label_b),
    )
    neighbours = sorted(
        pairs,
        key=lambda pair: (-pair.lift, -pair.n_both, pair.label_a, pair.label_b),
    )[:n_neighbours]
    siblings = [pair for pair in pairs if pair.sibling]
    others = [pair for pair in pairs if not pair.sibling]
    # the ranking's top N (the pairs the corpus's vocabulary likes most), not
    # the near-duplicate list: the pre-registration's test is about the
    # ranking, and the near-duplicate list can be empty.
    ranking = sorted(pairs, key=lambda pair: (-pair.jaccard, pair.label_a, pair.label_b))
    top_sibling_hits = sum(1 for pair in ranking[:n_neighbours] if pair.sibling)
    chance = (len(siblings) / len(pairs)) if pairs else 0.0

    def _mean(values: Sequence[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    return {
        "definitions": {
            "shared_words": "how many words the two labels' tables share",
            "jaccard": "|T1 & T2| / |T1 | T2| (the vocabulary overlap)",
            "containment": ("the share of the SMALLER table that lies inside the larger one"),
            "n_both": "the train texts carrying both labels",
            "lift": (
                "the co-occurrence lift: the observed count over the expected one "
                "under independence (n_a * n_b / N)"
            ),
            "text_jaccard": "the Jaccard of the two labels' train-text sets",
            "sibling": (
                "the two labels share a catalog parent: the taxonomy's own claim "
                "about the label space (the parents are types, not labels, so the "
                "sibling relation is the measurable form of it)"
            ),
            "note": (
                "the corpus is the train split only: no val text, no val label and "
                "no test text enters the relation"
            ),
        },
        "knobs": {
            "near_duplicate_jaccard": near_duplicate,
            "containment_threshold": containment_threshold,
            "n_neighbours": n_neighbours,
        },
        "summary": {
            "n_labels": len(relation.labels),
            "n_pairs": len(pairs),
            "n_labels_with_a_table": sum(1 for row in relation.per_label if row.n_words),
            "n_sibling_pairs": len(siblings),
            "mean_jaccard": _mean([pair.jaccard for pair in pairs]),
            "mean_containment": _mean([pair.containment for pair in pairs]),
            "mean_lift": _mean([pair.lift for pair in pairs]),
        },
        "near_duplicates": [pair.to_row() for pair in near],
        "closest_pairs": [pair.to_row() for pair in ranking[:n_neighbours]],
        "neighbours": [pair.to_row() for pair in neighbours],
        "vocabulary_less": vocabulary_less_labels(
            relation, containment_threshold=containment_threshold
        ),
        "catalog_comparison": {
            "sibling_pairs": len(siblings),
            "other_pairs": len(others),
            "chance_rate": round(chance, 6),
            "siblings": {
                "mean_jaccard": _mean([pair.jaccard for pair in siblings]),
                "mean_containment": _mean([pair.containment for pair in siblings]),
                "mean_lift": _mean([pair.lift for pair in siblings]),
            },
            "others": {
                "mean_jaccard": _mean([pair.jaccard for pair in others]),
                "mean_containment": _mean([pair.containment for pair in others]),
                "mean_lift": _mean([pair.lift for pair in others]),
            },
            "top_n_sibling_hits": top_sibling_hits,
            "top_n": n_neighbours,
            "separation_bootstrap": dict(separation) if separation else None,
            "note": (
                "the comparison says whether the corpus's own vocabulary relation "
                "is the taxonomy's: the catalog's 12 parents are types, so the "
                "measurable form of the taxonomy's claim is the sibling relation "
                "(the pairs of labels that share a parent)"
            ),
        },
        "per_label": [row.to_row() for row in relation.per_label],
        "per_pair": [pair.to_row() for pair in pairs],
    }
