"""Auxiliary pairwise relevance data (Phase 1.5).

The collaborator-provided ``data/external/omct_pairs.json`` contains
human-labeled paragraph ↔ GTI-question pairs with binary "same topic?"
labels. It is strictly auxiliary (PLAN.md section 2): model selection,
threshold calibration and optional reranker/fine-tune training. It is never
merged into UPR train artifacts and its runs get the ``omct-`` dataset prefix
(the command adds it at run time).
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


class PairRecord(BaseModel):
    """One manually judged (paragraph, question) pair.

    Fields mirror the collaborator dataset exactly; extra keys are rejected so
    schema drift fails loudly:

    - ``pair_id``: stable pair identifier (e.g. "AI-11-Q70")
    - ``source_csv``: which source table the pair was judged in
    - ``document``: the report the paragraph belongs to
    - ``paragraph_number``: string position of the paragraph inside the document
    - ``paragraph_text``: the scored text
    - ``question_number`` / ``question_text``: the GTI question id and wording
    - ``label``: 1 = same topic, 0 = reviewer-confirmed non-match
    - ``label_origin``: which labeling pass produced the judgment
    """

    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(min_length=1)
    source_csv: str = Field(min_length=1)
    document: str = Field(min_length=1)
    paragraph_number: str = Field(min_length=1)
    paragraph_text: str = Field(min_length=1)
    question_number: int
    question_text: str = Field(min_length=1)
    label: int
    label_origin: str = Field(min_length=1)

    @field_validator("paragraph_text", "question_text")
    @classmethod
    def _texts_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("label")
    @classmethod
    def _label_must_be_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("label must be 0 or 1")
        return value


@dataclass
class PairLoadReport:
    """Outcome of loading the pairwise dataset (parallel to ``data.LoadReport``).

    Attributes:
        path: The file the pairs were loaded from.
        metadata: All top-level sibling keys of ``"pairs"`` (name, counts, ...),
            passed through unchanged.
        records: Valid pairs, in file order.
        errors: 0-based pair index -> reason for every malformed (dropped) entry.
    """

    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    records: list[PairRecord] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        """Number of valid pairs loaded."""
        return len(self.records)

    @property
    def n_invalid(self) -> int:
        """Number of malformed entries dropped."""
        return len(self.errors)


def load_pairs(path: str | Path) -> PairLoadReport:
    """Load the pairwise dataset: one JSON object with a top-level ``pairs`` list.

    Every entry is validated against :class:`PairRecord`; malformed entries are
    reported by 0-based index and dropped. Top-level metadata passes through.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or lacks a ``pairs`` list.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError(f"{source} must be a JSON object with a top-level 'pairs' list")

    report = PairLoadReport(
        path=source, metadata={k: v for k, v in payload.items() if k != "pairs"}
    )
    for index, entry in enumerate(payload["pairs"]):
        try:
            report.records.append(PairRecord.model_validate(entry))
        except ValidationError as exc:
            report.errors[index] = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            )
    logger.info(
        "Loaded %d pairs from %s (%d malformed entries dropped)",
        report.n_valid,
        source,
        report.n_invalid,
    )
    return report


def summarize_pairs(records: Sequence[PairRecord]) -> dict[str, Any]:
    """Descriptive statistics for the pairwise dataset (analogous to ``summarize_records``).

    Pure function, JSON-serializable output; thresholds are not needed here,
    so everything is parameter-free.
    """
    label_counts: Counter[int] = Counter(record.label for record in records)
    document_counts: Counter[str] = Counter(record.document for record in records)
    source_counts: Counter[str] = Counter(record.source_csv for record in records)
    paragraph_documents: dict[str, set[str]] = {}
    question_documents: dict[str, set[str]] = {}
    positive_in_document: dict[str, int] = {}
    for record in records:
        paragraph_documents.setdefault(record.paragraph_text, set()).add(record.document)
        question_documents.setdefault(record.question_text, set()).add(record.document)
        if record.label == 1:
            positive_in_document[record.document] = positive_in_document.get(record.document, 0) + 1
    paragraph_lengths = [len(record.paragraph_text) for record in records]
    question_lengths = [len(record.question_text) for record in records]
    pair_ids = [record.pair_id for record in records]

    n = len(records)
    return {
        "n_pairs": n,
        "label_counts": {"1": label_counts.get(1, 0), "0": label_counts.get(0, 0)},
        "positive_share": label_counts.get(1, 0) / n if n else 0.0,
        "n_distinct_question_numbers": len({record.question_number for record in records}),
        "n_distinct_question_texts": len({record.question_text for record in records}),
        "n_paragraph_texts_in_multiple_documents": sum(
            1 for docs in paragraph_documents.values() if len(docs) > 1
        ),
        "n_question_texts_in_multiple_documents": sum(
            1 for docs in question_documents.values() if len(docs) > 1
        ),
        "n_documents": len(document_counts),
        "document_counts": dict(sorted(document_counts.items())),
        "positive_by_document": {
            document: {"n": count, "positive": positive_in_document.get(document, 0)}
            for document, count in sorted(document_counts.items())
        },
        "source_csv_counts": dict(sorted(source_counts.items())),
        "n_duplicate_pair_ids": len(pair_ids) - len(set(pair_ids)),
        "paragraph_text_chars": {
            "min": min(paragraph_lengths, default=0),
            "mean": statistics.fmean(paragraph_lengths) if paragraph_lengths else 0.0,
            "max": max(paragraph_lengths, default=0),
        },
        "question_text_chars": {
            "min": min(question_lengths, default=0),
            "mean": statistics.fmean(question_lengths) if question_lengths else 0.0,
            "max": max(question_lengths, default=0),
        },
    }
