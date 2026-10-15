"""Record data model, JSONL loading/validation, descriptive helpers and splits.

The raw dataset is JSONL, one JSON object per line (PLAN.md section 2):

    {"id": "...", "text": "...", "language": "en", "labels": ["a", "b"]}

``load_jsonl`` validates every line against the :class:`Record` model and
returns a :class:`LoadReport`: valid records are kept, malformed lines are
reported (with line numbers and reasons) and dropped, so a single bad row
never aborts the whole load. Files in ``data/raw/`` are read-only.

The split functions implement the Phase 1 fixed splits: multi-label
iterative stratification into train/val/test, a deterministic coverage
rebalance, and a label-holdout selection whose records are kept out of train
(see ``build_splits``). All are pure and take thresholds from config.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from cluster_rag.config import SplitsConfig

logger = logging.getLogger(__name__)

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


class Record(BaseModel):
    """A single labeled text record.

    Multi-label per PLAN.md section 2: ``labels`` has 1..n entries. Unknown
    fields are rejected so schema drift in the data fails loudly instead of
    being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("labels")
    @classmethod
    def _labels_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not label.strip() for label in value):
            raise ValueError("labels must not contain blank strings")
        return value


@dataclass
class LoadReport:
    """Outcome of loading a JSONL dataset.

    Attributes:
        path: The file the records were loaded from.
        records: All valid :class:`Record` objects, in file order.
        errors: Mapping of 1-based line number to a human-readable reason for
            every malformed (dropped) line. Blank lines are not errors.
    """

    path: Path
    records: list[Record] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        """Number of valid records loaded."""
        return len(self.records)

    @property
    def n_invalid(self) -> int:
        """Number of malformed lines dropped."""
        return len(self.errors)


def load_jsonl(path: str | Path) -> LoadReport:
    """Load a JSONL dataset, dropping and reporting malformed lines.

    Args:
        path: Path to the JSONL file.

    Returns:
        A :class:`LoadReport` with the valid records and per-line errors.
    """
    source = Path(path)
    report = LoadReport(path=source)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors[line_number] = f"invalid JSON: {exc.msg}"
                continue
            try:
                report.records.append(Record.model_validate(payload))
            except ValidationError as exc:
                report.errors[line_number] = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()
                )
    logger.info(
        "Loaded %d records from %s (%d malformed lines dropped)",
        report.n_valid,
        source,
        report.n_invalid,
    )
    return report


def summarize_records(records: Sequence[Record], top_labels: int = 10) -> dict[str, Any]:
    """Compute a small descriptive summary of records.

    Fast overview builder, reused inside the full report from
    :mod:`cluster_rag.dataset_stats`.

    Args:
        records: Valid records to summarize.
        top_labels: How many most-frequent labels to include.

    Returns:
        A JSON-serializable dict with record/label/language statistics.
    """
    label_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    labels_per_record: list[int] = []
    text_lengths: list[int] = []
    for record in records:
        label_counts.update(record.labels)
        language_counts[record.language] += 1
        labels_per_record.append(len(record.labels))
        text_lengths.append(len(record.text))

    n = len(records)
    return {
        "n_records": n,
        "n_unique_labels": len(label_counts),
        "n_language_values": len(language_counts),
        "language_counts": dict(sorted(language_counts.items())),
        "labels_per_record": {
            "min": min(labels_per_record, default=0),
            "mean": statistics.fmean(labels_per_record) if labels_per_record else 0.0,
            "max": max(labels_per_record, default=0),
        },
        "text_length_chars": {
            "min": min(text_lengths, default=0),
            "mean": statistics.fmean(text_lengths) if text_lengths else 0.0,
            "max": max(text_lengths, default=0),
        },
        "top_labels": label_counts.most_common(top_labels),
    }


def build_label_matrix(records: Sequence[Record], labels: Sequence[str]) -> np.ndarray:
    """Build the multi-hot indicator matrix ``(n_records, n_labels)`` of dtype bool.

    ``labels`` must be the complete, ordered label set (the convention is the
    sorted unique labels), so column order is stable across runs.
    """
    label_pos = {label: pos for pos, label in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=bool)
    for row, record in enumerate(records):
        for label in record.labels:
            matrix[row, label_pos[label]] = True
    return matrix


def stratified_split_indices(
    label_matrix: np.ndarray, ratios: Sequence[float], seed: int
) -> np.ndarray:
    """Assign every sample a split id (0=train, 1=val, 2=test), deterministically.

    Multi-label iterative stratification (Sechidis et al. 2011) via
    ``iterative-stratification``. Two stages with fixed seeds: the test share
    is carved from the full dataset (``seed``), then the val share from the
    remainder (``seed + 1``), so per-label proportions follow ``ratios``
    (train, val, test) as closely as the algorithm can manage.
    """
    train_ratio, val_ratio, test_ratio = ratios
    n_samples = label_matrix.shape[0]
    split_of = np.full(n_samples, -1, dtype=np.int8)
    dummy_x = np.zeros((n_samples, 1))

    test_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, train_size=1 - test_ratio, random_state=seed
    )
    rest_idx, test_idx = next(test_splitter.split(dummy_x, label_matrix))
    split_of[test_idx] = 2

    val_share = val_ratio / (val_ratio + train_ratio)
    val_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_share, train_size=1 - val_share, random_state=seed + 1
    )
    train_rest, val_idx = next(
        val_splitter.split(np.zeros((rest_idx.size, 1)), label_matrix[rest_idx])
    )
    split_of[rest_idx[val_idx]] = 1
    split_of[rest_idx[train_rest]] = 0
    if (split_of < 0).any():
        raise ValueError("internal error: stratification left samples unassigned")
    return split_of


def rebalance_split_coverage(
    label_matrix: np.ndarray,
    split_of: np.ndarray,
    *,
    min_total: int,
    targets: Sequence[int] = (2, 1, 0),
) -> list[dict[str, Any]]:
    """Move samples so that every label with >= min_total examples hits every split.

    Iterative stratification maximizes per-label balance but cannot guarantee
    coverage for rare labels. This pass fixes what it can: for each label
    missing from one of the ``targets`` splits one containing sample is moved
    over, chosen deterministically by preferring a donor split that (a) does
    not lose its last example of the label, (b) has the most examples of the
    label, (c) whose sample carries the fewest labels, then (d) the smallest
    row index. Labels rarer than ``min_total`` are left alone (reported as
    uncovered by the caller) instead of being churned.

    The default order fixes test first, then val, then train. ``targets`` may
    exclude train (id 0) — e.g. after label-holdout redistribution, where
    moving anything back into train would break the "no holdout label in
    train" invariant.

    Args:
        label_matrix: Multi-hot indicator matrix (not modified).
        split_of: Split id per sample, modified in place.
        min_total: Minimum total examples for a label to be force-corrected.
        targets: Target split ids to fix, in priority order.

    Returns:
        One dict per move: ``{"row", "label_index", "from", "to"}``.
    """
    moves: list[dict[str, Any]] = []
    for target in targets:
        for label_idx in range(label_matrix.shape[1]):
            column = label_matrix[:, label_idx]
            if column[split_of == target].sum() > 0:
                continue
            if column.sum() < min_total:
                continue
            candidates = np.flatnonzero(column & (split_of != target))
            if candidates.size == 0:
                continue
            best: dict[str, Any] | None = None
            for row in candidates:
                row = int(row)
                source = int(split_of[row])
                source_count = int(column[split_of == source].sum())
                key = (
                    0 if source_count >= 2 else 1,  # never empty a split if avoidable
                    -source_count,
                    int(label_matrix[row].sum()),
                    row,
                )
                if best is None or key < best["key"]:
                    best = {"key": key, "row": row, "source": source}
            row, source = best["row"], best["source"]
            split_of[row] = target
            moves.append(
                {
                    "row": row,
                    "label_index": label_idx,
                    "from": SPLIT_NAMES[source],
                    "to": SPLIT_NAMES[target],
                }
            )
    return moves


def select_label_holdout(
    label_matrix: np.ndarray,
    split_of: np.ndarray,
    labels: Sequence[str],
    *,
    n_labels: int,
    min_examples: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    """Choose labels to hold out of train entirely, minimizing collateral damage.

    Eligibility: labels whose total example count (whole dataset) is within
    ``[min_examples, max_examples]`` — rare enough to be interesting for
    zero-shot discovery, frequent enough to evaluate on val/test.

    Selection is greedy and deterministic (no RNG): candidates are scored by
    (a) collateral = how many *other* labels would disappear from train
    entirely if this label's train records were removed, (b) more total
    examples preferred (a stronger zero-shot evaluation sample), (c) label
    name for a stable tie-break. Collateral is recomputed after every pick.

    Args:
        label_matrix: Multi-hot indicator matrix over all records.
        split_of: Current split id per sample (train is id 0).
        labels: Ordered label names matching the matrix columns.

    Returns:
        One dict per selected label, in pick order:
        ``{"label", "index", "total_examples", "train_records_removed",
        "collateral_labels_removed_from_train"}``.
    """
    total_counts = label_matrix.sum(axis=0)
    eligible = [
        idx for idx in range(len(labels)) if min_examples <= total_counts[idx] <= max_examples
    ]
    train_matrix = label_matrix[split_of == 0]
    active = np.ones(train_matrix.shape[0], dtype=bool)

    picked: list[dict[str, Any]] = []
    for _ in range(n_labels):
        rows = train_matrix[active]
        if rows.shape[0] == 0:
            break
        present = rows.any(axis=0)
        best: dict[str, Any] | None = None
        for idx in eligible:
            if not present[idx]:
                continue  # already gone from train (collateral of an earlier pick)
            would_vanish = (rows & ~rows[:, idx][:, None]).sum(axis=0) == 0
            collateral = [
                other
                for other in range(len(labels))
                if other != idx and present[other] and would_vanish[other]
            ]
            key = (len(collateral), -int(total_counts[idx]), labels[idx])
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "index": idx,
                    "collateral": [labels[other] for other in collateral],
                    "train_records_removed": int(rows[:, idx].sum()),
                }
        if best is None:
            break  # no eligible label left in train
        active &= ~train_matrix[:, best["index"]]
        picked.append(
            {
                "label": labels[best["index"]],
                "index": best["index"],
                "total_examples": int(total_counts[best["index"]]),
                "train_records_removed": best["train_records_removed"],
                "collateral_labels_removed_from_train": best["collateral"],
            }
        )
    return picked


def build_splits(records: Sequence[Record], splits: SplitsConfig, seed: int) -> dict[str, Any]:
    """Run the full Phase 1 split pipeline and return a manifest-ready dict.

    Steps, in order:
    1. Stratified train/val/test assignment (:func:`stratified_split_indices`).
    2. Deterministic coverage rebalance (:func:`rebalance_split_coverage`).
    3. Label-holdout selection (:func:`select_label_holdout`).
    4. Records still in train that contain a held-out label are redistributed
       into val/test (stratified on their label vectors, seed + 2), so the
       train split contains no held-out label and both evaluation splits keep
       their share of held-out examples.

    Returns:
        A dict with ``ratios``, ``seed``, ``sizes``, ``ids`` (per split,
        sorted), ``label_holdout``, ``holdout_details``, ``coverage_moves``,
        ``per_split_label_counts`` and ``labels_missing_from_train``. After
        step 4 ``labels_missing_from_train`` should equal the holdout set;
        any extra name there is accidental collateral (reported, not hidden).
    """
    labels = sorted({label for record in records for label in record.labels})
    label_matrix = build_label_matrix(records, labels)
    split_of = stratified_split_indices(label_matrix, splits.ratios, seed)
    moves = rebalance_split_coverage(label_matrix, split_of, min_total=splits.coverage_min_total)

    holdout_details = select_label_holdout(
        label_matrix,
        split_of,
        labels,
        n_labels=splits.label_holdout.n_labels,
        min_examples=splits.label_holdout.min_examples,
        max_examples=splits.label_holdout.max_examples,
    )

    holdout_indices = [detail["index"] for detail in holdout_details]
    moved_rows = np.array([], dtype=int)
    if holdout_indices:
        holdout_columns = np.zeros(label_matrix.shape[1], dtype=bool)
        holdout_columns[holdout_indices] = True
        in_train = split_of == 0
        moved_rows = np.flatnonzero(in_train & label_matrix[:, holdout_columns].any(axis=1))
        if moved_rows.size > 0:
            test_share = splits.ratios[2] / (splits.ratios[1] + splits.ratios[2])
            moved_splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=test_share,
                train_size=1 - test_share,
                random_state=seed + 2,
            )
            val_part, test_part = next(
                moved_splitter.split(np.zeros((moved_rows.size, 1)), label_matrix[moved_rows])
            )
            split_of[moved_rows[val_part]] = 1
            split_of[moved_rows[test_part]] = 2
        # Second coverage pass, eval splits only: redistribution above may have
        # emptied a label from one of them. Never moves anything into train,
        # so the "no holdout label in train" invariant is preserved.
        moves += rebalance_split_coverage(
            label_matrix, split_of, min_total=splits.coverage_min_total, targets=(2, 1)
        )

    record_ids = [record.id for record in records]
    ids_by_split = {
        name: sorted(record_ids[row] for row in np.flatnonzero(split_of == pos))
        for pos, name in enumerate(SPLIT_NAMES)
    }
    per_split_label_counts = {
        name: {
            label: int(label_matrix[split_of == pos, label_idx].sum())
            for label_idx, label in enumerate(labels)
        }
        for pos, name in enumerate(SPLIT_NAMES)
    }
    logger.info(
        "Split built: %s (holdout labels: %d, coverage moves: %d)",
        {name: len(ids) for name, ids in ids_by_split.items()},
        len(holdout_details),
        len(moves),
    )
    return {
        "ratios": {"train": splits.ratios[0], "val": splits.ratios[1], "test": splits.ratios[2]},
        "seed": seed,
        "n_records": len(records),
        "n_labels_total": len(labels),
        "sizes": {name: len(ids) for name, ids in ids_by_split.items()},
        "ids": ids_by_split,
        "label_holdout": [detail["label"] for detail in holdout_details],
        "holdout_details": holdout_details,
        "coverage_moves": [
            {
                "record_id": record_ids[move["row"]],
                "label": labels[move["label_index"]],
                "from": move["from"],
                "to": move["to"],
            }
            for move in moves
        ],
        "per_split_label_counts": per_split_label_counts,
        "labels_missing_from_train": sorted(
            label for label, count in per_split_label_counts["train"].items() if count == 0
        ),
    }
