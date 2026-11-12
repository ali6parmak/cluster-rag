"""Data audit for the EURLEX-4K dataset.

Reads ``data/eurlex_4k/{train,dev,test}.jsonl`` and the label vocabulary
``data/eurlex_4k/eurovoc_concepts.jsonl`` (read-only) and writes
``results/eurlex-stats/stats.json``.  The numbers in the track's
``docs/topic_classification.md`` are copied from that artifact.

The audit covers: the schema and per-split counts, the label space (including
the vocabulary join and the support distribution), cardinality, text lengths,
and leakage (exact and near duplicates, citation links).
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as normalize_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eurlex_4k"
OUT = ROOT / "results" / "eurlex-stats" / "stats.json"
SPLITS = ("train", "dev", "test")
FIELDS = ("title", "header", "recitals", "main_body")

# Label categories: train support -> bucket.  The thresholds are the audit's
# working definition; they are recorded in the artifact and in the doc.
FREQUENT_MIN = 10
HEAD_MIN = 100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for exact-duplicate hashing."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text: str) -> list[str]:
    """Word tokens (unicode word characters)."""
    return re.findall(r"\w+", text)


def field_text(record: dict[str, Any], field: str) -> str:
    """Return a field as text; ``main_body`` is a list of paragraphs."""
    value = record[field]
    if field == "main_body":
        return "\n".join(value)
    return value or ""


def whole_text(record: dict[str, Any]) -> str:
    """The whole document, in field order."""
    return "\n".join(field_text(record, field) for field in FIELDS)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    """Mean, median, p95, max of a numeric sequence (rounded)."""
    data = sorted(values)
    return {
        "n": len(data),
        "mean": round(statistics.mean(data), 2),
        "median": round(statistics.median(data), 2),
        "p95": data[int(0.95 * (len(data) - 1))],
        "max": data[-1],
    }


def length_stats(values: Iterable[str]) -> dict[str, Any]:
    """Character and word length statistics for a sequence of texts."""
    texts = list(values)
    return {
        "chars": describe(len(text) for text in texts),
        "words": describe(len(words(text)) for text in texts),
        "empty": sum(1 for text in texts if not text.strip()),
    }


def schema_report(flat: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, Any]:
    """Keys, field types, document types and empty-field counts."""
    key_sets: collections.Counter[tuple[str, ...]] = collections.Counter()
    field_types: dict[str, set[str]] = {field: set() for field in FIELDS}
    empty: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    doc_types: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in SPLITS
    }
    for record in flat:
        split = record["_split"]
        key_sets[tuple(sorted(record))] += 1
        for field in FIELDS:
            field_types[field].add(type(record[field]).__name__)
        doc_types[split][record["document_type"]] += 1
        for field in FIELDS:
            text = field_text(record, field)
            if not text.strip():
                empty[split][field] = empty[split].get(field, 0) + 1
    return {
        "documents": counts,
        "key_sets": [list(keys) for keys in key_sets],
        "field_types": {field: sorted(types) for field, types in field_types.items()},
        "document_types": {
            split: dict(sorted(counter.items(), key=lambda item: -item[1]))
            for split, counter in doc_types.items()
        },
        "empty_fields": empty,
    }


def label_report(
    flat: list[dict[str, Any]], vocab: dict[str, str], counts: dict[str, int], vocab_rows: int
) -> dict[str, Any]:
    """Join the split labels to the vocabulary and describe their support."""
    split_ids = {str(label) for record in flat for label in record["eurovoc_concepts"]}
    unmatched = sorted(split_ids - set(vocab))
    unused = sorted(set(vocab) - split_ids)

    support: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in SPLITS
    }
    for record in flat:
        support[record["_split"]].update(set(map(str, record["eurovoc_concepts"])))

    def category(label_id: str) -> str:
        train_support = support["train"][label_id]
        if train_support == 0:
            return "zero_shot"
        if train_support < FREQUENT_MIN:
            return "few_shot"
        return "frequent"

    categories = {label_id: category(label_id) for label_id in split_ids}
    category_counts = collections.Counter(categories.values())

    per_split: dict[str, Any] = {}
    for split in SPLITS:
        present = {label_id for label_id in split_ids if support[split][label_id] > 0}
        labels_by_category = collections.Counter(categories[label_id] for label_id in present)
        docs_with_category: collections.Counter[str] = collections.Counter()
        assignments_by_category: collections.Counter[str] = collections.Counter()
        for record in flat:
            if record["_split"] != split:
                continue
            labels = {str(label) for label in record["eurovoc_concepts"]}
            for label_id in labels:
                assignments_by_category[categories[label_id]] += 1
            for label_category in {categories[label_id] for label_id in labels}:
                docs_with_category[label_category] += 1
        per_split[split] = {
            "documents": counts[split],
            "distinct_labels": len(present),
            "labels_by_category": {
                key: labels_by_category.get(key, 0) for key in ("frequent", "few_shot", "zero_shot")
            },
            "documents_with_category": {
                key: docs_with_category.get(key, 0) for key in ("frequent", "few_shot", "zero_shot")
            },
            "assignments_by_category": {
                key: assignments_by_category.get(key, 0)
                for key in ("frequent", "few_shot", "zero_shot")
            },
            "assignments_total": sum(assignments_by_category.values()),
        }

    bins = [
        (0, 0, "0"),
        (1, 1, "1"),
        (2, 4, "2-4"),
        (5, 9, "5-9"),
        (10, 19, "10-19"),
        (20, 49, "20-49"),
        (50, 99, "50-99"),
        (100, 499, "100-499"),
        (500, 10**9, "500+"),
    ]
    support_bins = [
        {
            "bin": name,
            "labels": sum(1 for label_id in split_ids if low <= support["train"][label_id] <= high),
        }
        for low, high, name in bins
    ]

    top = sorted(split_ids, key=lambda label_id: (-support["train"][label_id], label_id))
    top_labels = [
        {
            "id": label_id,
            "title": vocab.get(label_id, ""),
            "train_support": support["train"][label_id],
            "dev_support": support["dev"][label_id],
            "test_support": support["test"][label_id],
        }
        for label_id in top[:20]
    ]
    zero_shot = [
        {
            "id": label_id,
            "title": vocab.get(label_id, ""),
            "dev_support": support["dev"][label_id],
            "test_support": support["test"][label_id],
        }
        for label_id in sorted(split_ids)
        if categories[label_id] == "zero_shot"
    ]
    examples = {}
    for name in ("frequent", "few_shot", "zero_shot"):
        ids = [label_id for label_id in sorted(split_ids) if categories[label_id] == name]
        examples[name] = [
            {
                "id": label_id,
                "title": vocab.get(label_id, ""),
                "train_support": support["train"][label_id],
            }
            for label_id in ids[:10]
        ]

    alphanumeric = sorted(
        label_id for label_id in split_ids if any(ch.isalpha() for ch in label_id)
    )
    return {
        "vocabulary": {
            "path": "data/eurlex_4k/eurovoc_concepts.jsonl",
            "rows": vocab_rows,
            "distinct_ids": len(vocab),
            "empty_titles": sum(1 for title in vocab.values() if not title.strip()),
            "alphanumeric_ids": sum(
                1 for label_id in vocab if any(ch.isalpha() for ch in label_id)
            ),
        },
        "join": {
            "split_distinct_ids": len(split_ids),
            "matched": len(split_ids) - len(unmatched),
            "unmatched": unmatched,
            "titles_available": sum(1 for label_id in split_ids if vocab.get(label_id, "").strip()),
            "vocabulary_unused": len(unused),
            "alphanumeric_ids_used": alphanumeric,
        },
        "categories": {
            "frequent": {
                "definition": f"train support >= {FREQUENT_MIN}",
                "labels": category_counts["frequent"],
            },
            "few_shot": {
                "definition": f"train support 1-{FREQUENT_MIN - 1}",
                "labels": category_counts["few_shot"],
            },
            "zero_shot": {
                "definition": "train support 0, present in dev/test",
                "labels": category_counts["zero_shot"],
            },
        },
        "head_torso_tail_unseen": {
            "head": sum(1 for label_id in split_ids if support["train"][label_id] >= HEAD_MIN),
            "torso": sum(
                1 for label_id in split_ids if FREQUENT_MIN <= support["train"][label_id] < HEAD_MIN
            ),
            "tail": sum(
                1 for label_id in split_ids if 0 < support["train"][label_id] < FREQUENT_MIN
            ),
            "unseen": category_counts["zero_shot"],
        },
        "support_bins": support_bins,
        "per_split": per_split,
        "top_train_labels": top_labels,
        "zero_shot_ids": zero_shot,
        "examples": examples,
    }


def cardinality_report(flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Labels-per-document distribution, overall and per split."""

    def one(values: list[int]) -> dict[str, Any]:
        values = sorted(values)
        histogram = collections.Counter(values)
        return {
            "documents": len(values),
            "mean": round(statistics.mean(values), 3),
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
            "p90": values[int(0.90 * (len(values) - 1))],
            "p95": values[int(0.95 * (len(values) - 1))],
            "p99": values[int(0.99 * (len(values) - 1))],
            "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        }

    overall = one([len(record["eurovoc_concepts"]) for record in flat])
    per_split = {
        split: one(
            [len(record["eurovoc_concepts"]) for record in flat if record["_split"] == split]
        )
        for split in SPLITS
    }
    return {"overall": overall, "per_split": per_split}


def text_report(flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Field lengths, the whole-document length, and the fields' char share."""
    fields: dict[str, Any] = {}
    total_chars = collections.Counter()
    for field in FIELDS:
        texts = [field_text(record, field) for record in flat]
        total_chars[field] = sum(len(text) for text in texts)
        fields[field] = {
            "overall": length_stats(texts),
            "per_split": {
                split: length_stats(
                    field_text(record, field) for record in flat if record["_split"] == split
                )
                for split in SPLITS
            },
        }
    whole = [whole_text(record) for record in flat]
    paragraphs = [len(record["main_body"]) for record in flat]
    header_contains_title = 0
    for record in flat:
        title = normalize(record["title"])
        header = normalize(record["header"])
        if title and title in header:
            header_contains_title += 1
    denominator = sum(total_chars.values())
    return {
        "fields": fields,
        "whole_document": {
            "overall": length_stats(whole),
            "per_split": {
                split: length_stats(
                    whole_text(record) for record in flat if record["_split"] == split
                )
                for split in SPLITS
            },
        },
        "char_share_percent": {
            field: round(100 * total_chars[field] / denominator, 1) for field in FIELDS
        },
        "main_body_paragraphs": {
            "overall": describe(paragraphs),
            "per_split": {
                split: describe(
                    len(record["main_body"]) for record in flat if record["_split"] == split
                )
                for split in SPLITS
            },
        },
        "header_contains_title_share": round(header_contains_title / len(flat), 4),
    }


def exact_duplicates(flat: list[dict[str, Any]], name: str, text_fn: Any) -> dict[str, Any]:
    """Exact normalized-string duplicates across splits, for one field."""
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, record in enumerate(flat):
        digest = hashlib.sha1(normalize(text_fn(record)).encode("utf-8")).hexdigest()
        groups[digest].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    cross_groups = [
        indices
        for indices in duplicate_groups
        if len({flat[index]["_split"] for index in indices}) > 1
    ]
    examples = []
    for indices in cross_groups[:3]:
        examples.append(
            [
                {"split": flat[index]["_split"], "celex_id": flat[index]["celex_id"]}
                for index in indices[:4]
            ]
        )
    return {
        "field": name,
        "duplicate_groups": len(duplicate_groups),
        "cross_split_groups": len(cross_groups),
        "documents_in_duplicate_groups": sum(len(indices) for indices in duplicate_groups),
        "documents_with_cross_split_twin": sum(len(indices) for indices in cross_groups),
        "examples": examples,
    }


def near_duplicates(
    flat: list[dict[str, Any]], n_train: int, text_fn: Any, label: str
) -> dict[str, Any]:
    """TF-IDF cosine near-duplicates: each dev/test doc's best train match."""
    train_texts = [text_fn(record) for record in flat[:n_train]]
    all_texts = [text_fn(record) for record in flat]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(train_texts)
    matrix = normalize_rows(vectorizer.transform(all_texts))
    train_matrix = matrix[:n_train]
    query = matrix[n_train:]

    maxima: list[np.ndarray] = []
    argmaxima: list[np.ndarray] = []
    for start in range(0, query.shape[0], 500):
        similarity = query[start : start + 500] @ train_matrix.T
        maxima.append(np.asarray(similarity.max(axis=1).toarray()).ravel())
        argmaxima.append(np.asarray(similarity.argmax(axis=1)).ravel())
    best = np.concatenate(maxima)
    best_index = np.concatenate(argmaxima)

    thresholds = {}
    for threshold in (0.95, 0.90, 0.80, 0.70):
        hits = int((best >= threshold).sum())
        thresholds[f"{threshold:.2f}"] = {
            "documents": hits,
            "share": round(hits / len(best), 4),
        }
    examples = []
    for position in np.argsort(-best)[:5]:
        query_record = flat[n_train + int(position)]
        train_record = flat[int(best_index[position])]
        examples.append(
            {
                "cosine": round(float(best[position]), 4),
                "query": {
                    "split": query_record["_split"],
                    "celex_id": query_record["celex_id"],
                    "title": query_record["title"].strip(),
                },
                "neighbor": {
                    "split": train_record["_split"],
                    "celex_id": train_record["celex_id"],
                    "title": train_record["title"].strip(),
                },
            }
        )
    return {
        "method": f"TF-IDF word unigrams, fit on train, cosine; {label}",
        "thresholds": thresholds,
        "examples": examples,
    }


def title_skeleton(title: str) -> str:
    """A coarse template key: digits -> '#', punctuation removed."""
    text = title.lower()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[^a-z# ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def skeleton_report(flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Title-template families and how many cross splits."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in flat:
        groups[title_skeleton(record["title"])].append(record)
    multi = [records for records in groups.values() if len(records) > 1]
    cross = [records for records in multi if len({record["_split"] for record in records}) > 1]
    top = []
    for records in sorted(multi, key=len, reverse=True)[:10]:
        top.append(
            {
                "documents": len(records),
                "splits": sorted({record["_split"] for record in records}),
                "example_title": records[0]["title"].strip(),
                "skeleton": title_skeleton(records[0]["title"]),
            }
        )
    return {
        "families": len(groups),
        "families_with_more_than_one_document": len(multi),
        "documents_in_multi_document_families": sum(len(records) for records in multi),
        "cross_split_families": len(cross),
        "documents_in_cross_split_families": sum(len(records) for records in cross),
        "top_families": top,
    }


REFERENCE_PATTERNS = (
    re.compile(
        r"\b(Regulation|Decision|Directive)\s*(?:\([A-Za-z]+\)\s*)?No\.?\s*(\d+)\s*/\s*(\d{4})\b",
        re.I,
    ),
    re.compile(
        r"\b(Regulation|Decision|Directive)\s+(?:\([A-Za-z]+\)\s*)?(\d{4})\s*/\s*(\d+)\b", re.I
    ),
)
TYPE_LETTER = {"Regulation": "R", "Decision": "D", "Directive": "L"}


def citation_report(flat: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-split citations by instrument number/year in title and header."""
    split_of = {record["celex_id"]: record["_split"] for record in flat}
    duplicated_celex = len(flat) - len(split_of)

    def references(record: dict[str, Any]) -> set[tuple[str, int, int]]:
        found: set[tuple[str, int, int]] = set()
        for field in ("title", "header"):
            text = record[field]
            for pattern in REFERENCE_PATTERNS:
                for match in pattern.finditer(text):
                    kind = TYPE_LETTER[match.group(1).capitalize()]
                    if pattern is REFERENCE_PATTERNS[0]:
                        found.add((kind, int(match.group(3)), int(match.group(2))))
                    else:
                        found.add((kind, int(match.group(2)), int(match.group(3))))
        return found

    def celex_id(reference: tuple[str, int, int]) -> str:
        kind, year, number = reference
        return f"3{year:04d}{kind}{number:04d}"

    documents_with_cross = 0
    edges: collections.Counter[str] = collections.Counter()
    examples = []
    language = collections.Counter()
    language_cross = collections.Counter()
    for record in flat:
        references_here = references(record)
        cross = [
            reference
            for reference in references_here
            if celex_id(reference) in split_of and split_of[celex_id(reference)] != record["_split"]
        ]
        if cross:
            documents_with_cross += 1
            for reference in cross:
                edges[f"{record['_split']}->{split_of[celex_id(reference)]}"] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "source": {"split": record["_split"], "celex_id": record["celex_id"]},
                        "targets": [
                            {
                                "split": split_of[celex_id(reference)],
                                "celex_id": celex_id(reference),
                            }
                            for reference in cross[:3]
                        ],
                    }
                )
        title_lower = record["title"].lower()
        for keyword in ("amending", "implementing", "repealing"):
            if keyword in title_lower:
                language[keyword] += 1
                if cross:
                    language_cross[keyword] += 1
    return {
        "celex_ids": len(split_of),
        "duplicated_celex_ids": duplicated_celex,
        "documents_with_cross_split_citation": documents_with_cross,
        "share_documents_with_cross_split_citation": round(documents_with_cross / len(flat), 4),
        "edges_by_direction": dict(sorted(edges.items())),
        "amendment_language": {
            "amending_titles": language["amending"],
            "implementing_titles": language["implementing"],
            "repealing_titles": language["repealing"],
            "amending_with_cross_split_citation": language_cross["amending"],
            "implementing_with_cross_split_citation": language_cross["implementing"],
            "repealing_with_cross_split_citation": language_cross["repealing"],
        },
        "examples": examples,
    }


def sample_documents(
    flat: list[dict[str, Any]], vocab: dict[str, str], support: dict[str, collections.Counter[str]]
) -> list[dict[str, Any]]:
    """Five illustrative documents, including the one alphanumeric label."""
    chosen: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            chosen.append(index)

    add(0)
    for index, record in enumerate(flat):
        if any(any(ch.isalpha() for ch in str(label)) for label in record["eurovoc_concepts"]):
            add(index)
            break
    for split in ("dev", "test"):
        for index, record in enumerate(flat):
            if record["_split"] == split:
                add(index)
                break
    for index, record in enumerate(flat):
        if any(support["train"][str(label)] == 0 for label in record["eurovoc_concepts"]):
            add(index)
            break

    samples = []
    for index in chosen[:5]:
        record = flat[index]
        labels = [
            {"id": str(label), "title": vocab.get(str(label), "")}
            for label in record["eurovoc_concepts"]
        ]
        samples.append(
            {
                "split": record["_split"],
                "celex_id": record["celex_id"],
                "document_type": record["document_type"],
                "title": record["title"].strip(),
                "labels": labels,
                "cardinality": len(record["eurovoc_concepts"]),
                "lengths": {field: len(field_text(record, field)) for field in FIELDS},
            }
        )
    return samples


def main() -> None:
    """Run the audit and write the artifact."""
    counts = {
        split: sum(1 for _ in (DATA / f"{split}.jsonl").open(encoding="utf-8")) for split in SPLITS
    }
    flat: list[dict[str, Any]] = []
    for split in SPLITS:
        for record in read_jsonl(DATA / f"{split}.jsonl"):
            record["_split"] = split
            flat.append(record)
    n_train = counts["train"]
    vocab_records = read_jsonl(DATA / "eurovoc_concepts.jsonl")
    vocab = {str(record["id"]): record.get("title") or "" for record in vocab_records}
    vocab_counter = collections.Counter(str(record["id"]) for record in vocab_records)

    support: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in SPLITS
    }
    for record in flat:
        support[record["_split"]].update(set(map(str, record["eurovoc_concepts"])))

    stats: dict[str, Any] = {
        "dataset": "eurlex_4k",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source_files": {
            "train": "data/eurlex_4k/train.jsonl",
            "dev": "data/eurlex_4k/dev.jsonl",
            "test": "data/eurlex_4k/test.jsonl",
            "vocabulary": "data/eurlex_4k/eurovoc_concepts.jsonl",
        },
        "schema": schema_report(flat, counts),
        "labels": label_report(flat, vocab, counts, len(vocab_records)),
        "cardinality": cardinality_report(flat),
        "text": text_report(flat),
        "leakage": {
            "exact_duplicates": {
                field: exact_duplicates(
                    flat, field, lambda record, field=field: field_text(record, field)
                )
                for field in FIELDS
            }
            | {"whole_document": exact_duplicates(flat, "whole_document", whole_text)},
            "near_duplicates": {
                "full_text": near_duplicates(
                    flat,
                    n_train,
                    lambda record: " ".join(
                        [
                            record["title"],
                            record["header"],
                            record["recitals"][:2500],
                            " ".join(record["main_body"])[:2500],
                        ]
                    ),
                    "title+header+first 2500 chars of recitals+main_body",
                ),
                "title": near_duplicates(
                    flat, n_train, lambda record: record["title"], "title only"
                ),
            },
            "title_skeleton_families": skeleton_report(flat),
            "citations": citation_report(flat),
        },
        "samples": sample_documents(flat, vocab, support),
    }
    stats["labels"]["vocabulary"]["duplicate_ids"] = len(vocab_records) - len(vocab_counter)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    join = stats["labels"]["join"]
    print(f"wrote {OUT}")
    print(
        f"documents {counts}; labels used {join['split_distinct_ids']}; "
        f"matched {join['matched']}; unmatched {len(join['unmatched'])}; "
        f"titles {join['titles_available']}"
    )
    print(
        "categories:",
        {
            key: stats["labels"]["categories"][key]["labels"]
            for key in ("frequent", "few_shot", "zero_shot")
        },
    )


if __name__ == "__main__":
    main()
