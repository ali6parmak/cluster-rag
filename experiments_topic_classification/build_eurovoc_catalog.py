"""Build the EURLEX-4K label catalog from the pinned official EUROVOC SKOS dump (TC-19).

Reads the core-concepts RDF, extracts each concept's immediate ``skos:broader``
terms, its English ``skos:prefLabel`` and its English ``skos:definition`` /
``skos:scopeNote``, and joins them onto the dataset's label space (the 4,271
distinct labels across the three splits; **only the label metadata is read from
the splits — no document text**).  Writes ``catalogs/eurlex_labels.jsonl`` and
``catalogs/eurlex_labels.meta.json``.

The source is pinned in ``docs/topic_classification.md`` section 23 (URL and
both sha256 hashes are verified here before parsing).

Usage:
    python experiments_topic_classification/build_eurovoc_catalog.py \
        --rdf .cache/tmp/eurovoc/eurovoc_in_skos_core_concepts.rdf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eurlex_4k"
CATALOG = ROOT / "catalogs" / "eurlex_labels.jsonl"
META = ROOT / "catalogs" / "eurlex_labels.meta.json"

# The pre-registration's pinned source (docs/topic_classification.md section 23).
SOURCE_URL = (
    "https://op.europa.eu/o/opportal-service/euvoc-download-handler?"
    "cellarURI=http%3A%2F%2Fpublications.europa.eu%2Fresource%2Fcellar%2F"
    "9f2bd600-ae7b-11e7-837e-01aa75ed71a1.0001.09%2FDOC_1"
    "&fileName=eurovoc_in_skos_core_concepts.zip"
)
EXPECTED_RDF_SHA256 = "6c362f79ad03e325ba1b4818f1ca3a847bb6167c2a8f7167e2e4df91305b6620"
EXPECTED_RDF_BYTES = 60_691_531

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SKOS = "{http://www.w3.org/2004/02/skos/core#}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
LABEL_FIELD = re.compile(r'"eurovoc_concepts"\s*:\s*\[([^\]]*)\]')
LABEL_VALUE = re.compile(r'"([^"]+)"')


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_label_ids(path: Path) -> set[str]:
    """The split's distinct label IDs, extracted without parsing document text."""
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = LABEL_FIELD.search(line)
            if match is None:
                continue
            ids.update(LABEL_VALUE.findall(match.group(1)))
    return ids


def concept_id(uri: str) -> str:
    """The EUROVOC id: the last path segment of the concept's URI."""
    return uri.rstrip("/").rsplit("/", 1)[-1]


def parse_skos(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the RDF/XML into ``{id: {name, parents, definition, definition_kind}}``."""
    concepts: dict[str, dict[str, Any]] = {}
    for _event, element in ET.iterparse(str(path), events=("end",)):
        if element.tag != RDF + "Description":
            continue
        identifier = concept_id(element.get(RDF + "about", ""))
        if not identifier:
            element.clear()
            continue
        parents: list[str] = []
        english_name: str | None = None
        definition: str | None = None
        scope_note: str | None = None
        for child in element:
            if child.tag == SKOS + "broader":
                parent = concept_id(child.get(RDF + "resource", ""))
                if parent:
                    parents.append(parent)
            elif child.tag == SKOS + "prefLabel" and child.get(XML_LANG) == "en":
                english_name = (child.text or "").strip() or None
            elif child.tag == SKOS + "definition" and child.get(XML_LANG) == "en":
                definition = (child.text or "").strip() or None
            elif child.tag == SKOS + "scopeNote" and child.get(XML_LANG) == "en":
                scope_note = (child.text or "").strip() or None
        if identifier not in concepts:
            kind = None
            text = None
            if definition:
                kind, text = "definition", definition
            elif scope_note:
                kind, text = "scope_note", scope_note
            concepts[identifier] = {
                "name": english_name,
                "parents": sorted(set(parents)),
                "definition": text,
                "definition_kind": kind,
            }
        element.clear()
    return concepts


def ancestors_of(
    identifier: str,
    concepts: dict[str, dict[str, Any]],
    memo: dict[str, set[str]],
    stack: set[str],
) -> set[str]:
    """The transitive ``skos:broader`` chain (cycle-guarded, memoised)."""
    if identifier in memo:
        return memo[identifier]
    if identifier in stack:
        return set()
    stack.add(identifier)
    result: set[str] = set()
    for parent in concepts.get(identifier, {}).get("parents", []):
        result.add(parent)
        result |= ancestors_of(parent, concepts, memo, stack)
    stack.discard(identifier)
    memo[identifier] = result
    return result


def main() -> None:
    """Build the catalog and its provenance file, validating the pre-registered join."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdf", type=Path, required=True, help="the EUROVOC SKOS RDF/XML")
    args = parser.parse_args()
    started = time.perf_counter()

    if not args.rdf.exists():
        raise SystemExit(f"RDF not found: {args.rdf}")
    digest = sha256_of(args.rdf)
    size = args.rdf.stat().st_size
    print(f"RDF {args.rdf} ({size:,} bytes, sha256 {digest})", flush=True)
    if digest != EXPECTED_RDF_SHA256 or size != EXPECTED_RDF_BYTES:
        raise SystemExit(
            f"RDF does not match the pinned source in section 23 (sha256 {digest}, {size} bytes)"
        )

    label_ids = set()
    for split in ("train", "dev", "test"):
        ids = split_label_ids(DATA / f"{split}.jsonl")
        print(f"  {split}: {len(ids):,} distinct label ids", flush=True)
        label_ids |= ids
    print(f"  union: {len(label_ids):,} label ids", flush=True)

    vocabulary = {
        str(record["id"]): record["title"]
        for record in (
            json.loads(line) for line in (DATA / "eurovoc_concepts.jsonl").open(encoding="utf-8")
        )
    }

    print("parsing the SKOS dump ...", flush=True)
    concepts = parse_skos(args.rdf)
    print(f"  {len(concepts):,} concepts parsed", flush=True)

    memo: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for identifier in sorted(label_ids):
        concept = concepts.get(identifier)
        if concept is None:
            missing.append(identifier)
            continue
        parents = concept["parents"]
        parent_names = [
            concepts.get(parent, {}).get("name") or " ".join(parent.split("_"))
            for parent in parents
        ]
        rows.append(
            {
                "id": identifier,
                "name": vocabulary.get(identifier),
                "parents": parents,
                "ancestors": sorted(ancestors_of(identifier, concepts, memo, set())),
                "parent_names": parent_names,
                "definition": concept["definition"],
                "definition_kind": concept["definition_kind"],
            }
        )

    with_parent = sum(1 for row in rows if row["parents"])
    multi_parent = sum(1 for row in rows if 2 <= len(row["parents"]) <= 9)
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[str(row["definition_kind"])] = kinds.get(str(row["definition_kind"]), 0) + 1
    with_definition = kinds.get("definition", 0)
    with_scope_note = kinds.get("scope_note", 0)
    missing_titles = sum(1 for row in rows if not row["name"])
    counts = {
        "labels": len(rows),
        "matched": len(rows),
        "unmatched": missing,
        "with_parent": with_parent,
        "parentless": len(rows) - with_parent,
        "multi_parent_2_9": multi_parent,
        "with_definition": with_definition,
        "with_scope_note": with_scope_note,
        "with_definition_or_scope_note": with_definition + with_scope_note,
        "definition_kinds": kinds,
        "dataset_title_missing": missing_titles,
        "max_parents": max((len(row["parents"]) for row in rows), default=0),
    }
    print(json.dumps(counts, indent=1), flush=True)

    # The pre-registration's measured join (section 23), checked exactly.
    expected = {
        "labels": 4_271,
        "matched": 4_271,
        "with_parent": 3_850,
        "parentless": 421,
        "multi_parent_2_9": 214,
        "with_definition": 660,
        "with_scope_note": 119,
    }
    failures = {
        key: {"expected": value, "measured": counts[key]}
        for key, value in expected.items()
        if counts[key] != value
    }
    if failures:
        raise SystemExit(f"the join does not reproduce the pre-registered counts: {failures}")

    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    with CATALOG.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "what": "EURLEX-4K label catalog: EUROVOC immediate broader terms (TC-19)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "downloaded_at": "2026-11-14",
        "pre_registration": "docs/topic_classification.md section 23",
        "reviewer_decision": "pre-registered and approved before the run (2026-11-14)",
        "source": {
            "url": SOURCE_URL,
            "zip_sha256": "91bdb24e833ba431707f3980a19f475434ea8dcddb2b4d5e32e79e9fc1a0ca2f",
            "rdf_sha256": EXPECTED_RDF_SHA256,
            "rdf_bytes": EXPECTED_RDF_BYTES,
            "zip_entry_date": "2026-07-07",
        },
        "parser": "experiments_topic_classification/build_eurovoc_catalog.py",
        "output": str(CATALOG.relative_to(ROOT)),
        "fields": {
            "id": "the dataset's label id",
            "name": "the dataset vocabulary title",
            "parents": "immediate skos:broader ids (0-9)",
            "ancestors": "the full transitive skos:broader chain",
            "parent_names": "the English skos:prefLabel of each parent",
            "definition": "English skos:definition, else skos:scopeNote, else null",
            "definition_kind": "definition / scope_note / null",
        },
        "split_labels": "the 4,271 distinct labels across train/dev/test (label metadata only)",
        "counts": counts,
        "rules": {
            "parent": "the immediate skos:broader only (no full ancestry as a feature)",
            "definition": "recorded only; NOT a feature in this change set",
            "join": "label id == the last path segment of the concept URI",
        },
        "expected_counts": expected,
        "seconds_total": round(time.perf_counter() - started, 1),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {CATALOG} and {META} in {meta['seconds_total']}s")


if __name__ == "__main__":
    sys.exit(main())
