"""Record data model and JSONL loading with validation.

The raw dataset is JSONL, one JSON object per line (PLAN.md section 2):

    {"id": "...", "text": "...", "language": "en", "labels": ["a", "b"]}

``load_jsonl`` validates every line against the :class:`Record` model and
returns a :class:`LoadReport`: valid records are kept, malformed lines are
reported (with line numbers and reasons) and dropped, so a single bad row
never aborts the whole load. Files in ``data/raw/`` are read-only.
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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


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

    This is the fast CLI-facing preview used by ``stats``; the full
    exploratory analysis (frequency buckets, co-occurrence, prefix analysis)
    is a Phase 1 deliverable.

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
