"""The optional **label catalog**: a name / description / parent per label.

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8d) names the label catalog as the
UPR feature pack's missing piece: the label-description coverage and the
label-type features (the label's parent category) need a description and a
parent per label, and the UPR dataset itself carries neither — a record's
labels are bare taxonomy terms.

The catalog is an **optional, human-readable JSONL file** (``paths.label_catalog``)
with one object per label::

    {"name": "Detention", "description": "arrest and detention", "parent": "Liberties"}

The file is the edge (it is read from disk), the data it carries is passed to
the track's pure pack as plain values. A run **without** a catalog is a
perfectly valid run: the pack's two catalog-gated feature groups are then
unavailable, the config refuses a selection of them, and the artifact records
the gap.

The catalog is **not** part of the dataset and never changes the raw data
(AGENTS.md: the raw data is read-only). It is a description of the taxonomy,
and a description of the taxonomy is label-side knowledge — exactly the kind
of thing the zero-shot label vectors are allowed to use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cluster_rag.data import Record


@dataclass(frozen=True)
class LabelEntry:
    """One label's catalog entry.

    Attributes:
        name: The label's name (it must match a dataset label).
        description: The label's description (a topical description; ``None``
            when the entry carries none).
        parent: The label's parent category (its **type** in the taxonomy;
            ``None`` when the entry carries none).
    """

    name: str
    description: str | None = None
    parent: str | None = None


def load_label_catalog(path: Path) -> dict[str, LabelEntry]:
    """The catalog's entries, keyed by the label's name.

    Args:
        path: The catalog's JSONL file (one object per line, with a ``name``
            and optionally a ``description`` and a ``parent``).

    Returns:
        The entries, keyed by the label's name.

    Raises:
        FileNotFoundError: If the catalog does not exist.
        ValueError: If a line is malformed, carries no name, or repeats a
            label.
    """
    if not path.is_file():
        raise FileNotFoundError(f"the label catalog does not exist: {path}")
    entries: dict[str, LabelEntry] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: Any = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"the label catalog's line {number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"the label catalog's line {number} is not a JSON object")
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"the label catalog's line {number} carries no 'name'")
        if name in entries:
            raise ValueError(f"the label catalog repeats the label {name!r} (line {number})")
        description = payload.get("description")
        parent = payload.get("parent")
        entries[name] = LabelEntry(
            name=name,
            description=description if isinstance(description, str) and description else None,
            parent=parent if isinstance(parent, str) and parent else None,
        )
    return entries


def catalog_entries_for(
    records: tuple[Record, ...] | list[Record],
    labels: tuple[str, ...] | list[str],
    catalog: dict[str, LabelEntry],
) -> dict[str, LabelEntry]:
    """The catalog's entries restricted to the dataset's label space.

    A catalog entry for a label the dataset does not have is ignored (the
    catalog may cover more than one snapshot of the taxonomy); a dataset label
    without an entry simply has no description and no parent.
    """
    known = set(labels)
    return {name: entry for name, entry in catalog.items() if name in known}
