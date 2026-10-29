"""The decision model's ablation map (the track's CS-18, Version B).

Version A maps the **label side** (the lexical feature's own decomposition);
this module maps the **text side**, and it is the only version of the three
that **perturbs the input** — so it is the only one that says anything causal
("attribution is not causation unless you perturb").

**The mechanism.** The decision model reads a paragraph as its state and asks
about all 68 labels in one batched call, so the cost is per **text**, not per
cell: removing one unit of the text and re-asking costs the same ~2 calls per
text. The unit is either the text's **sentence** (the track's own splitter) or
its **clause** (the track's own atom chunker, which splits a sentence at its
clause separators) — both are deterministic rule-based splitters already in the
repo, and the round's two arms differ only in the unit.

**What the ablation measures.** For every variant (a text with one unit
removed) the model's probability for every label is compared with its
probability for the intact text. The **drop** is the difference. Three reads
come out of it, all pre-registered in the plan archive's §7.1.4:

- the **label-linked unit** — the unit whose content words overlap the gold
  label's name the most — against the other units: does the probability fall
  when the words that name the label are taken away? (A tie keeps every tied
  unit and the linked side is the **mean** over them, so the two sides of the
  comparison are read the same way.)
- the **formulaic** units (a unit that opens with one of the config's
  ``preprocess.strip_openers``) against the **topical** ones: does the
  probability fall as much when the boilerplate is taken away?
- the **gold** cells against the **background** ones: is the fall concentrated
  on the labels that are actually true?

The **label-blind counterpart** (the same ablation with the wrong pairing) runs
beside every read: if its pattern is the same as the real arm's, the attribution
is the model's own prior and says nothing about the labels.

Pure core: numpy and the standard library only — no model client, no I/O. The
model client (``cluster_rag.decision``) lives at the edge and the CLI wires it
in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cluster_rag.eval.multilabel import paired_values_bootstrap
from cluster_rag.text_processors.chunkers.atoms import (
    AtomChunker,
    AtomChunkerConfig,
    split_sentences,
)
from cluster_rag.text_processors.decision.probe import DecisionCell
from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    content_tokens,
)

# The ablation units of the round's two arms.
UNIT_SENTENCE = "sentence"
UNIT_CLAUSE = "clause"
HIT_MAP_UNITS: tuple[str, ...] = (UNIT_SENTENCE, UNIT_CLAUSE)

# The clause separators the removal takes with the unit it removes (the track's
# atom chunker's own separators).
CLAUSE_SEPARATORS: tuple[str, ...] = (",", ";")


@dataclass(frozen=True)
class AblationUnit:
    """One unit of a text (the removable piece) and its character span.

    Attributes:
        index: The unit's index inside its text.
        text: The unit's own text.
        begin: The unit's first character in the source text.
        end: The unit's last character (exclusive) in the source text.
        formulaic: Whether the unit opens with one of the config's
            ``preprocess.strip_openers`` (the boilerplate flag).
    """

    index: int
    text: str
    begin: int
    end: int
    formulaic: bool


@dataclass(frozen=True)
class AblationVariant:
    """One ablation variant: a text with one of its units removed.

    Attributes:
        text_row: The text's row in the grid.
        unit_index: The removed unit's index inside the text.
        text: The ablated text (the source text without the unit).
        unit: The removed unit.
    """

    text_row: int
    unit_index: int
    text: str
    unit: AblationUnit


def is_formulaic(text: str, openers: Sequence[str]) -> bool:
    """Whether a unit is **boilerplate**: it opens with one of the openers.

    The openers are the repo's own ``preprocess.strip_openers`` list ("continue
    to ensure that …", "take all necessary measures to …"), so the boilerplate
    flag is the pipeline's own notion of a formulaic opening, not a hand-made
    one.

    Args:
        text: The unit's text.
        openers: The opener prefixes (the longest match wins).

    Returns:
        Whether the unit opens with one of them.
    """
    lowered = text.strip().lower()
    for opener in sorted(openers, key=len, reverse=True):
        if lowered.startswith(opener.lower()):
            return True
    return False


def text_units(
    text: str,
    *,
    unit: str = UNIT_CLAUSE,
    openers: Sequence[str] = (),
) -> tuple[AblationUnit, ...]:
    """One text's ablation units, with their character spans.

    The **sentence** unit is the track's own sentence splitter (the same one
    the chunk header and the chunk profile use); the **clause** unit is the
    track's atom chunker with the clause splitting on (the CS-2/CS-3 atoms).
    Both are deterministic and rule-based.

    Args:
        text: The text.
        unit: ``"sentence"`` or ``"clause"``.
        openers: The opener prefixes (the boilerplate flag).

    Returns:
        The units, in the text's order.

    Raises:
        ValueError: If the unit kind is unknown.
    """
    if unit == UNIT_SENTENCE:
        pieces = split_sentences(text)
    elif unit == UNIT_CLAUSE:
        chunker = AtomChunker(AtomChunkerConfig(split_clauses=True))
        pieces = chunker.segment([text])[0]
    else:
        raise ValueError(f"unknown ablation unit {unit!r}; expected one of {list(HIT_MAP_UNITS)}")
    spans = locate_spans(text, pieces)
    return tuple(
        AblationUnit(
            index=index,
            text=piece,
            begin=begin,
            end=end,
            formulaic=is_formulaic(piece, openers),
        )
        for index, (piece, (begin, end)) in enumerate(zip(pieces, spans, strict=True))
    )


def locate_spans(text: str, pieces: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """The character span of every piece inside its text.

    The pieces are located in order, each searched from the previous piece's
    end (the same convention as the chunkers' ``segment_spans`` fallback), so a
    repeated piece resolves in order.

    Args:
        text: The text the pieces came from.
        pieces: The pieces (substrings of the text, in order).

    Returns:
        The spans, in the pieces' order.

    Raises:
        ValueError: If a piece cannot be located in the text.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        position = text.find(piece, cursor)
        if position < 0:
            raise ValueError(f"the piece {piece[:40]!r} is not a substring of its text")
        spans.append((position, position + len(piece)))
        cursor = position + len(piece)
    return tuple(spans)


def remove_unit(text: str, unit: AblationUnit) -> str:
    """The text without one of its units (the ablated text).

    The unit is removed **with one adjacent separator** (the clause splitter
    drops the separators, so removing the clause alone would leave ", ,"), and
    the remaining whitespace is normalised. Everything else is kept verbatim,
    so the only difference between the intact text and the ablated one is the
    removed unit.

    Args:
        text: The intact text.
        unit: The unit to remove.

    Returns:
        The ablated text.
    """
    start, stop = unit.begin, unit.end
    if stop < len(text) and text[stop] in CLAUSE_SEPARATORS:
        stop += 1
    head = " ".join(text[:start].split())
    tail = " ".join(text[stop:].split())
    # a separator left dangling at the junction goes with the removed unit
    while head and head[-1] in CLAUSE_SEPARATORS:
        head = head[:-1].rstrip()
    while tail and tail[0] in CLAUSE_SEPARATORS:
        tail = tail[1:].lstrip()
    return " ".join(part for part in (head, tail) if part)


def build_variants(
    texts: Sequence[str],
    units: Sequence[Sequence[AblationUnit]],
) -> tuple[AblationVariant, ...]:
    """The round's ablation variants: one per **removable** unit.

    A text with *k* units yields *k − 1* variants (removing its only unit
    leaves nothing to ask about, so a one-unit text is **not ablatable** — it is
    reported as such, never silently dropped).

    Args:
        texts: The texts (the grid's rows).
        units: Every text's units (from :func:`text_units`).

    Returns:
        The variants, ordered by (text row, unit index) — the order the
        variant scores are expected in.
    """
    variants: list[AblationVariant] = []
    for row, text in enumerate(texts):
        row_units = units[row]
        if len(row_units) < 2:
            continue  # not ablatable: reported, never silently dropped
        for unit in row_units:
            variants.append(
                AblationVariant(
                    text_row=row,
                    unit_index=unit.index,
                    text=remove_unit(text, unit),
                    unit=unit,
                )
            )
    return tuple(variants)


def label_linked_units(
    units: Sequence[AblationUnit],
    label_name: str,
    *,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> tuple[int, ...]:
    """The units that overlap a label's name the most (the label-linked units).

    The overlap is the number of **content words** the unit and the label's
    name share. Every unit that reaches the maximum is linked (a tie keeps them
    all, so a text whose every unit mentions the label is not forced into one).

    Args:
        units: The text's units.
        label_name: The label's name.
        stopwords: The stopwords of the tokenizer.

    Returns:
        The indices of the label-linked units (empty when no unit shares a word
        with the name).
    """
    label_tokens = set(content_tokens(label_name, stopwords))
    if not label_tokens:
        return ()
    overlaps = [len(set(content_tokens(unit.text, stopwords)) & label_tokens) for unit in units]
    if not overlaps:
        return ()
    best = max(overlaps)
    if best == 0:
        return ()
    return tuple(index for index, value in enumerate(overlaps) if value == best)


def measure_ablation(
    *,
    texts: Sequence[str],
    labels: Sequence[str],
    gold: np.ndarray,
    units: Sequence[Sequence[AblationUnit]],
    variants: Sequence[AblationVariant],
    full_scores: np.ndarray,
    ablated_scores: np.ndarray,
    blind_full_scores: np.ndarray | None = None,
    blind_ablated_scores: np.ndarray | None = None,
    resamples: int = 2000,
    seed: int = 42,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
) -> dict[str, object]:
    """Measure the ablation: the per-text drops and the three paired reads.

    Args:
        texts: The texts (the grid's rows).
        labels: The label space (the grid's columns).
        gold: The bool gold matrix of the grid.
        units: Every text's units.
        variants: The variants (from :func:`build_variants`), in the order the
            variant scores are given in.
        full_scores: The intact texts' probability matrix ``(n_texts, n_labels)``.
        ablated_scores: The variants' probability matrix
            ``(n_variants, n_labels)``.
        blind_full_scores: The **label-blind counterpart's** intact-text matrix.
        blind_ablated_scores: The label-blind counterpart's variant matrix.
        resamples: The paired bootstrap's resamples.
        seed: The paired bootstrap's seed.
        stopwords: The stopwords of the tokenizer.

    Returns:
        The measurement block (JSON-friendly).

    Raises:
        ValueError: If the score matrices' shapes do not match the grid.
    """
    truth = np.asarray(gold, dtype=bool)
    full = np.asarray(full_scores, dtype=np.float64)
    ablated = np.asarray(ablated_scores, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != len(labels):
        raise ValueError(f"the gold matrix must have {len(labels)} columns, got {truth.shape}")
    if full.shape != truth.shape:
        raise ValueError(f"the intact scores {full.shape} must match the gold {truth.shape}")
    if ablated.shape != (len(variants), len(labels)):
        raise ValueError(
            f"the ablated scores {ablated.shape} must match the variants "
            f"({len(variants)}, {len(labels)})"
        )
    blind_full = (
        None if blind_full_scores is None else np.asarray(blind_full_scores, dtype=np.float64)
    )
    blind_ablated = (
        None if blind_ablated_scores is None else np.asarray(blind_ablated_scores, dtype=np.float64)
    )

    variant_index: dict[tuple[int, int], int] = {
        (variant.text_row, variant.unit_index): index for index, variant in enumerate(variants)
    }

    per_text: list[dict[str, object]] = []
    linked_means: list[float] = []
    other_means: list[float] = []
    gold_means: list[float] = []
    background_means: list[float] = []
    formulaic_means: list[float] = []
    topical_means: list[float] = []
    blind_linked_means: list[float] = []
    blind_other_means: list[float] = []
    blind_gold_means: list[float] = []
    blind_background_means: list[float] = []
    blind_formulaic_means: list[float] = []
    blind_topical_means: list[float] = []

    for row in range(len(texts)):
        row_units = units[row]
        if len(row_units) < 2:
            continue
        indexes = [variant_index[(row, unit.index)] for unit in row_units]
        drops = full[row] - ablated[indexes]  # (n_units, n_labels)
        blind_drops = (
            None
            if blind_full is None or blind_ablated is None
            else blind_full[row] - blind_ablated[indexes]
        )
        gold_columns = np.flatnonzero(truth[row])
        background_columns = np.flatnonzero(~truth[row])
        unit_gold_drops = (
            drops[:, gold_columns].mean(axis=1) if gold_columns.size else np.zeros(len(row_units))
        )
        unit_background_drops = (
            drops[:, background_columns].mean(axis=1)
            if background_columns.size
            else np.zeros(len(row_units))
        )
        blind_unit_gold_drops = (
            None
            if blind_drops is None
            else (
                blind_drops[:, gold_columns].mean(axis=1)
                if gold_columns.size
                else np.zeros(len(row_units))
            )
        )
        blind_unit_background_drops = (
            None
            if blind_drops is None
            else (
                blind_drops[:, background_columns].mean(axis=1)
                if background_columns.size
                else np.zeros(len(row_units))
            )
        )
        formulaic = np.asarray([unit.formulaic for unit in row_units], dtype=bool)

        # the label-linked unit, per gold cell
        linked_drops: list[float] = []
        other_drops: list[float] = []
        blind_linked_drops: list[float] = []
        blind_other_drops: list[float] = []
        for column in gold_columns:
            linked = label_linked_units(row_units, labels[int(column)], stopwords=stopwords)
            if not linked:
                continue
            other_indices = [index for index in range(len(row_units)) if index not in linked]
            if not other_indices:
                # every unit is label-linked: the "other" side is undefined for
                # this cell, so the text carries no linked-vs-other pair
                continue
            linked_drops.append(float(np.mean(unit_gold_drops[list(linked)])))
            other_drops.append(float(np.mean([unit_gold_drops[index] for index in other_indices])))
            if blind_unit_gold_drops is not None:
                blind_linked_drops.append(float(np.mean(blind_unit_gold_drops[list(linked)])))
                blind_other_drops.append(
                    float(np.mean([blind_unit_gold_drops[index] for index in other_indices]))
                )
        linked_mean = float(np.mean(linked_drops)) if linked_drops else float("nan")
        other_mean = float(np.mean(other_drops)) if other_drops else float("nan")
        blind_linked_mean = (
            float(np.mean(blind_linked_drops)) if blind_linked_drops else float("nan")
        )
        blind_other_mean = float(np.mean(blind_other_drops)) if blind_other_drops else float("nan")
        gold_mean = float(unit_gold_drops.mean())
        background_mean = float(unit_background_drops.mean())
        blind_gold_mean = (
            float(blind_unit_gold_drops.mean())
            if blind_unit_gold_drops is not None
            else float("nan")
        )
        blind_background_mean = (
            float(blind_unit_background_drops.mean())
            if blind_unit_background_drops is not None
            else float("nan")
        )
        formulaic_mean = (
            float(unit_gold_drops[formulaic].mean()) if formulaic.any() else float("nan")
        )
        topical_mean = (
            float(unit_gold_drops[~formulaic].mean()) if (~formulaic).any() else float("nan")
        )
        blind_formulaic_mean = (
            float(blind_unit_gold_drops[formulaic].mean())
            if blind_unit_gold_drops is not None and formulaic.any()
            else float("nan")
        )
        blind_topical_mean = (
            float(blind_unit_gold_drops[~formulaic].mean())
            if blind_unit_gold_drops is not None and (~formulaic).any()
            else float("nan")
        )
        if not np.isnan(linked_mean) and not np.isnan(other_mean):
            linked_means.append(linked_mean)
            other_means.append(other_mean)
        if not np.isnan(gold_mean):
            gold_means.append(gold_mean)
            background_means.append(background_mean)
        if not np.isnan(formulaic_mean) and not np.isnan(topical_mean):
            formulaic_means.append(formulaic_mean)
            topical_means.append(topical_mean)
        if not np.isnan(blind_linked_mean) and not np.isnan(blind_other_mean):
            blind_linked_means.append(blind_linked_mean)
            blind_other_means.append(blind_other_mean)
        if not np.isnan(blind_gold_mean):
            blind_gold_means.append(blind_gold_mean)
            blind_background_means.append(blind_background_mean)
        if not np.isnan(blind_formulaic_mean) and not np.isnan(blind_topical_mean):
            blind_formulaic_means.append(blind_formulaic_mean)
            blind_topical_means.append(blind_topical_mean)

        per_text.append(
            {
                "text_row": row,
                "n_units": len(row_units),
                "n_gold_cells": int(gold_columns.size),
                "gold_drop_mean": round(gold_mean, 6),
                "background_drop_mean": round(background_mean, 6),
                "linked_drop_mean": None if np.isnan(linked_mean) else round(linked_mean, 6),
                "other_drop_mean": None if np.isnan(other_mean) else round(other_mean, 6),
                "formulaic_drop_mean": (
                    None if np.isnan(formulaic_mean) else round(formulaic_mean, 6)
                ),
                "topical_drop_mean": None if np.isnan(topical_mean) else round(topical_mean, 6),
                "blind_gold_drop_mean": (
                    None if np.isnan(blind_gold_mean) else round(blind_gold_mean, 6)
                ),
                "blind_background_drop_mean": (
                    None if np.isnan(blind_background_mean) else round(blind_background_mean, 6)
                ),
                "blind_linked_drop_mean": (
                    None if np.isnan(blind_linked_mean) else round(blind_linked_mean, 6)
                ),
                "blind_other_drop_mean": (
                    None if np.isnan(blind_other_mean) else round(blind_other_mean, 6)
                ),
                "units": [
                    {
                        "index": int(unit.index),
                        "formulaic": bool(unit.formulaic),
                        "gold_drop": round(float(unit_gold_drops[position]), 6),
                        "background_drop": round(float(unit_background_drops[position]), 6),
                        "text": unit.text,
                    }
                    for position, unit in enumerate(row_units)
                ],
            }
        )

    reads = {
        "label_linked_vs_other": _paired_read(
            linked_means,
            other_means,
            resamples=resamples,
            seed=seed,
            left_name="the drop when the label-linked unit is removed",
            right_name="the drop when another unit is removed",
        ),
        "formulaic_vs_topical": _paired_read(
            formulaic_means,
            topical_means,
            resamples=resamples,
            seed=seed,
            left_name="the drop when a formulaic unit is removed",
            right_name="the drop when a topical unit is removed",
        ),
        "gold_vs_background": _paired_read(
            gold_means,
            background_means,
            resamples=resamples,
            seed=seed,
            left_name="the gold cells' drop",
            right_name="the background cells' drop",
        ),
    }
    blind_reads = {
        "label_linked_vs_other": _paired_read(
            blind_linked_means,
            blind_other_means,
            resamples=resamples,
            seed=seed,
            left_name="the blind drop when the label-linked unit is removed",
            right_name="the blind drop when another unit is removed",
        ),
        "formulaic_vs_topical": _paired_read(
            blind_formulaic_means,
            blind_topical_means,
            resamples=resamples,
            seed=seed,
            left_name="the blind drop when a formulaic unit is removed",
            right_name="the blind drop when a topical unit is removed",
        ),
        "gold_vs_background": _paired_read(
            blind_gold_means,
            blind_background_means,
            resamples=resamples,
            seed=seed,
            left_name="the blind gold cells' drop",
            right_name="the blind background cells' drop",
        ),
    }
    linked = reads["label_linked_vs_other"]
    formulaic = reads["formulaic_vs_topical"]
    gold = reads["gold_vs_background"]
    blind_linked = blind_reads["label_linked_vs_other"]
    falsification = {
        "label_linked_drop_greater": {
            "condition": (
                "the label-linked unit's drop is larger than the other units' "
                "drop (the model keys on the words that name the label)"
            ),
            "value": linked.get("difference"),
            "holds": bool(
                linked.get("difference") is not None and float(linked["difference"]) > 0.0
            ),
        },
        "formulaic_drop_smaller": {
            "condition": (
                "a formulaic unit's drop is smaller than a topical unit's drop "
                "(the model does not key on the boilerplate)"
            ),
            "value": formulaic.get("difference"),
            "holds": bool(
                formulaic.get("difference") is not None and float(formulaic["difference"]) < 0.0
            ),
        },
        "label_driven": {
            "condition": (
                "the real arm's label-linked drop is larger than the "
                "label-blind counterpart's (the attribution is label-driven)"
            ),
            "value": (
                float(linked["difference"]) - float(blind_linked["difference"])
                if linked.get("difference") is not None
                and blind_linked.get("difference") is not None
                else None
            ),
            "holds": bool(
                linked.get("difference") is not None
                and blind_linked.get("difference") is not None
                and float(linked["difference"]) > float(blind_linked["difference"])
            ),
        },
        "note": (
            "the pre-registered falsification for question 3, read out one by "
            "one: the version is falsified when the label-linked drop is not "
            "larger, or when the formulaic drop is not smaller, or when the "
            "label-blind counterpart's pattern is the same as the real arm's"
        ),
    }
    return {
        "reads": reads,
        "label_blind_counterpart": blind_reads,
        "falsification": falsification,
        "n_ablatable_texts": len(per_text),
        "per_text": per_text,
    }


def _paired_read(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    left_name: str,
    right_name: str,
) -> dict[str, object]:
    """One paired read: the two means and their paired bootstrap interval."""
    left_array = np.asarray(left_values, dtype=np.float64)
    right_array = np.asarray(right_values, dtype=np.float64)
    if left_array.size == 0 or left_array.size != right_array.size:
        return {
            "left": left_name,
            "right": right_name,
            "n_texts": int(left_array.size),
            "left_mean": None,
            "right_mean": None,
            "difference": None,
            "interval": None,
            "note": "no text carried both sides of the comparison",
        }
    block = paired_values_bootstrap(left_array, right_array, resamples=resamples, seed=seed)
    return {
        "left": left_name,
        "right": right_name,
        "n_texts": int(left_array.size),
        "left_mean": round(float(np.nanmean(left_array)), 6),
        "right_mean": round(float(np.nanmean(right_array)), 6),
        "difference": round(float(block["delta"]), 6),
        "interval": [
            round(float(block["ci_low"]), 6),
            round(float(block["ci_high"]), 6),
        ],
        "share_positive": block.get("share_positive"),
        "metric": block.get("metric"),
    }


def unit_histogram(units: Sequence[Sequence[AblationUnit]]) -> dict[str, int]:
    """The units' count histogram (the text set's shape, per unit count)."""
    histogram: dict[str, int] = {}
    for row_units in units:
        key = str(len(row_units))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def ablatable_summary(
    units: Sequence[Sequence[AblationUnit]],
) -> dict[str, object]:
    """The text set's shape: the unit histogram and the not-ablatable share."""
    histogram = unit_histogram(units)
    n_texts = len(units)
    n_ablatable = sum(1 for row_units in units if len(row_units) >= 2)
    n_variants = sum(max(0, len(row_units) - 1) for row_units in units)
    return {
        "n_texts": n_texts,
        "unit_histogram": histogram,
        "n_ablatable": n_ablatable,
        "n_not_ablatable": n_texts - n_ablatable,
        "share_not_ablatable": round((n_texts - n_ablatable) / n_texts, 6) if n_texts else None,
        "n_variants": n_variants,
        "note": (
            "a text with one unit has no ablation (removing it leaves "
            "nothing to ask about): it is reported as 'not ablatable', never "
            "silently dropped"
        ),
    }


def label_bucket_map(buckets: Mapping[str, str], labels: Sequence[str]) -> dict[str, str]:
    """The ``label -> bucket`` mapping, restricted to the label space."""
    return {label: buckets.get(label, "unknown") for label in labels}


def build_ablation_cells(
    variants: Sequence[AblationVariant],
    labels: Sequence[str],
    *,
    mismatched: bool = False,
) -> tuple[DecisionCell, ...]:
    """The cells of the ablation variants: the ablated text as the state.

    The **label-blind counterpart** (``mismatched=True``) asks the shifted label
    of every cell (the deterministic cyclic shift the track uses), so the
    ablation keeps its shape, its cost and its calibration and loses only the
    pairing — the yardstick that says whether the attribution is label-driven.

    Args:
        variants: The variants (from :func:`build_variants`).
        labels: The label space.
        mismatched: Whether the cells carry the label-blind counterpart.

    Returns:
        The cells, in the variants' order (every variant's labels in order).
    """
    n_labels = len(labels)
    cells: list[DecisionCell] = []
    for variant in variants:
        for column in range(n_labels):
            asked = (column + 1) % n_labels if mismatched else column
            cells.append(
                DecisionCell(
                    cell_id=f"ablation-{variant.text_row}-{variant.unit_index}-{column}",
                    group="ablation",
                    document="upr",
                    paragraph_row=variant.text_row,
                    question_col=column,
                    paragraph_text=variant.text,
                    question_text=labels[asked],
                    label=None,
                )
            )
    return tuple(cells)
