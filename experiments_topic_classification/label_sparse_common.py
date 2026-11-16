"""TC-21 shared harness: the label-sparse regime on EURLEX-4K.

The deployment regime is **240 labelled documents over the 4,197-label EURLEX
space** (120/2,000 ~ 240/4,197 ~ 0.057 documents per label).  This module holds
the parts every TC-21 arm shares: the five seeded 240-document samples, the
seen/unseen label classes, the fixed slices, the label-text builders, the cached
embedding bank, the lexical / embedding k-NN, the per-seed evaluation, the
cross-seed pooling and the paired bootstrap.

Pre-registered in ``docs/topic_classification.md`` section 27 before any run.
The labelled corpus is the sample only: the other 44,760 train documents' labels
are never read.  The word-cluster arm (sub-task 3) is the single pre-authorised
exception and may read train **text** for the label-free clustering.

Artifacts are written by the three sub-task scripts, one per commit:
``results/eurlex-label-sparse``, ``-transfer`` and ``-clusters``.  Test is never
read.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from embed_chunkmean import knn_scores  # noqa: E402
from lexical_rrf import TFIDF_CONFIG, lexical_scores  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    _embedder_config,
    full_text,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranked,
    label_recall_at_k,
    paired_bootstrap_ci,
    per_document_ndcg_at_k,
    per_document_r_precision,
    per_document_recall_at_k,
    per_document_subset_recall_at_k,
    ranked_labels,
)

SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
LABELLED_DOCUMENTS = 240
SOURCES_PER_SEED = 45_000
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_V1 = "eurlex-4k-label-v1"
LABEL_CACHE_V2 = "eurlex-4k-label-v2"
BOOTSTRAP_RESAMPLES = 2_000
SLICE_NAMES: tuple[str, ...] = ("all_dev", "family_new", "family_new_clean")
SUPPORT_BUCKETS: tuple[str, ...] = ("1", "2-4", "5+", "unseen")
LABEL_TEXT_VARIANTS: tuple[str, ...] = (
    "title",
    "title_parents",
    "title_parents_definition",
)
PRIMARY_LABEL_VARIANT = "title_parents"
CATALOG = ROOT / "catalogs" / "eurlex_labels.jsonl"
DATA = ROOT / "data" / "eurlex_4k"
LABEL_VOCABULARY = DATA / "eurovoc_concepts.jsonl"
READOUT_KS: tuple[int, ...] = (5, 10, 20, 50, 100)


def load_dataset() -> dict[str, Any]:
    """Load the train/dev splits, the 4,197-label space and the label catalog."""
    train_records = load_split("train")
    dev_records = load_split("dev")
    all_records = train_records + dev_records
    labels = sorted({str(label) for record in all_records for label in record["eurovoc_concepts"]})
    label_index = {label: index for index, label in enumerate(labels)}
    train_labels = [
        [str(label) for label in record["eurovoc_concepts"]] for record in train_records
    ]
    dev_labels = [[str(label) for label in record["eurovoc_concepts"]] for record in dev_records]
    gold = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    vocabulary = {str(record["id"]): record["title"] for record in read_jsonl(LABEL_VOCABULARY)}
    missing_titles = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing_titles:
        raise ValueError(f"{len(missing_titles)} labels have no title: {missing_titles[:5]}")
    catalog = {
        str(record["id"]): record
        for record in (json.loads(line) for line in CATALOG.open(encoding="utf-8"))
    }
    missing_catalog = [label for label in labels if label not in catalog]
    if missing_catalog:
        raise ValueError(
            f"{len(missing_catalog)} labels missing from the catalog: {missing_catalog[:5]}"
        )
    return {
        "train_records": train_records,
        "dev_records": dev_records,
        "labels": labels,
        "label_index": label_index,
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "gold": gold,
        "vocabulary": vocabulary,
        "catalog": catalog,
        "train_texts": [full_text(record) for record in train_records],
        "dev_texts": [full_text(record) for record in dev_records],
    }


def build_slices(
    train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """The track's fixed document slices (the 240 never redefine them)."""
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    masks = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_new_clean": family_new & clean,
    }
    return {
        "masks": masks,
        "documents": {name: int(mask.sum()) for name, mask in masks.items()},
        "clean_documents": int(clean.sum()),
        "family_new_documents": int(family_new.sum()),
        "max_cosines": max_cosines,
    }


def sample_documents(seed: int, n_documents: int = SOURCES_PER_SEED) -> np.ndarray:
    """The seeded 240-document labelled corpus, sorted ascending."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_documents, size=LABELLED_DOCUMENTS, replace=False))


def label_classes(
    seed_rows: np.ndarray, train_labels: list[list[str]], labels: list[str]
) -> dict[str, Any]:
    """Seen/unseen label classes for one seeded corpus."""
    support = Counter(label for row in seed_rows for label in set(train_labels[int(row)]))
    seen = {index for index, label in enumerate(labels) if support[label] >= 1}
    unseen = set(range(len(labels))) - seen
    class_of_label = {
        index: ("seen" if index in seen else "unseen") for index in range(len(labels))
    }
    return {
        "seen": seen,
        "unseen": unseen,
        "class_of_label": class_of_label,
        "seen_support": support,
        "seen_labels": len(seen),
        "unseen_labels": len(unseen),
    }


def support_bucket_map(
    seed_rows: np.ndarray, train_labels: list[list[str]], labels: list[str]
) -> tuple[dict[int, str], dict[str, int]]:
    """The 240-support bucket of every label: 1 / 2-4 / 5+ / unseen."""
    support = Counter(label for row in seed_rows for label in set(train_labels[int(row)]))
    mapping: dict[int, str] = {}
    counts = {"1": 0, "2-4": 0, "5+": 0, "unseen": 0}
    for index, label in enumerate(labels):
        value = support[label]
        if value == 0:
            bucket = "unseen"
        elif value == 1:
            bucket = "1"
        elif value <= 4:
            bucket = "2-4"
        else:
            bucket = "5+"
        mapping[index] = bucket
        counts[bucket] += 1
    return mapping, counts


def label_texts(
    labels: list[str],
    vocabulary: dict[str, str],
    catalog: dict[str, dict[str, Any]],
    variant: str,
) -> list[str]:
    """The label text for one of the three pre-registered variants."""
    if variant not in LABEL_TEXT_VARIANTS:
        raise ValueError(f"unknown label-text variant {variant!r}")
    texts: list[str] = []
    for label in labels:
        parts = [vocabulary[label]]
        if variant != "title":
            parents = list(dict.fromkeys(catalog[label]["parent_names"]))
            parts.extend(parents)
        if variant == "title_parents_definition" and catalog[label]["definition"]:
            parts.append(str(catalog[label]["definition"]))
        texts.append(" | ".join(parts))
    return texts


class EmbeddingBank:
    """One MiniLM backend behind the document / label caches of TC-21."""

    def __init__(self) -> None:
        """Load the model once and open the three cache views."""
        self.backend = HFEmbedder(_embedder_config())
        self.documents = CachedEmbedder(
            self.backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION
        )
        self.titles = CachedEmbedder(
            self.backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_V1
        )
        self.enriched = CachedEmbedder(
            self.backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_V2
        )

    def label_vectors(self, variant: str, texts: list[str]) -> np.ndarray:
        """Embed one label-text variant through the cache view its texts belong to."""
        embedder = self.titles if variant == "title" else self.enriched
        return embedder.encode(texts, "label")

    def stats(self) -> dict[str, int]:
        """Hit/miss counts for the run metadata."""
        return {
            "document_cache_hits": self.documents.cache_hits,
            "document_cache_misses": self.documents.cache_misses,
            "title_cache_hits": self.titles.cache_hits,
            "title_cache_misses": self.titles.cache_misses,
            "enriched_cache_hits": self.enriched.cache_hits,
            "enriched_cache_misses": self.enriched.cache_misses,
        }


def embedding_knn(
    index_vectors: np.ndarray, index_label_matrix: np.ndarray, query_vectors: np.ndarray
) -> np.ndarray:
    """The track's embedding k-NN (k = 100 clamped, power 1.0) over the index."""
    return knn_scores(index_vectors, index_label_matrix, query_vectors)


def lexical_knn(
    index_texts: list[str], index_label_matrix: np.ndarray, query_texts: list[str]
) -> tuple[np.ndarray, int]:
    """The track's TF-IDF k-NN (k = 100 clamped, power 1.0) over the index."""
    vectorizer = TfidfVectorizer(
        lowercase=TFIDF_CONFIG["lowercase"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )
    index_matrix = vectorizer.fit_transform(index_texts)
    query_matrix = vectorizer.transform(query_texts)
    scores = lexical_scores(query_matrix, index_matrix, index_label_matrix)
    return scores, int(len(vectorizer.vocabulary_))


def _nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def _flatten_slice(slice_name: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for group in ("primary", "secondary"):
        for key, value in metrics[group].items():
            flat[f"{slice_name}/{key}"] = float(value)
    flat[f"{slice_name}/coverage@50"] = float(metrics["diagnostics"]["coverage@50"])
    for bucket, row in metrics["diagnostics"]["label_recall@50"].items():
        flat[f"{slice_name}/label_recall@50/{bucket}"] = float(row["recall"])
        flat[f"{slice_name}/label_recall@50/{bucket}_n"] = float(row["labels"])
    return flat


def evaluate_ranked_output(
    ranked: np.ndarray,
    gold: list[set[int]],
    seen: set[int],
    unseen: set[int],
    support_map: dict[int, str],
    slices: dict[str, Any],
) -> dict[str, Any]:
    """Flat per-slice metrics plus the per-document vectors a bootstrap needs."""
    n_labels = int(ranked.shape[1])
    class_of_label = {index: ("seen" if index in seen else "unseen") for index in range(n_labels)}
    metrics: dict[str, dict[str, float]] = {}
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name in SLICE_NAMES:
        indexes = np.flatnonzero(slices["masks"][name])
        slice_gold = [gold[index] for index in indexes]
        slice_ranked = ranked[indexes]
        flat = _flatten_slice(name, evaluate_ranked(slice_ranked, slice_gold, class_of_label))
        for k in READOUT_KS:
            if k == 50:
                continue
            recall = label_recall_at_k(slice_ranked, slice_gold, class_of_label, k)
            flat[f"{name}/label_recall@{k}/unseen"] = float(recall["unseen"]["recall"])
            flat[f"{name}/label_recall@{k}/seen"] = float(recall["seen"]["recall"])
        by_support = label_recall_at_k(slice_ranked, slice_gold, support_map, 50)
        for bucket in SUPPORT_BUCKETS:
            if bucket not in by_support:
                continue
            flat[f"{name}/support_recall@50/{bucket}"] = float(by_support[bucket]["recall"])
            flat[f"{name}/support_recall@50/{bucket}_n"] = float(by_support[bucket]["labels"])
        document_vectors = {
            "r50": per_document_recall_at_k(slice_ranked, slice_gold, 50),
            "r_precision": per_document_r_precision(slice_ranked, slice_gold),
            "ndcg10": per_document_ndcg_at_k(slice_ranked, slice_gold, 10),
            "unseen_r50": per_document_subset_recall_at_k(slice_ranked, slice_gold, unseen, 50),
            "seen_r50": per_document_subset_recall_at_k(slice_ranked, slice_gold, seen, 50),
        }
        flat[f"{name}/unseen_macro_recall@50"] = _nanmean(document_vectors["unseen_r50"])
        flat[f"{name}/seen_macro_recall@50"] = _nanmean(document_vectors["seen_r50"])
        flat[f"{name}/unseen_documents"] = float(np.isfinite(document_vectors["unseen_r50"]).sum())
        metrics[name] = flat
        vectors[name] = document_vectors
    return {"metrics": metrics, "vectors": vectors}


def evaluate_scores(
    scores: np.ndarray,
    gold: list[set[int]],
    seen: set[int],
    unseen: set[int],
    support_map: dict[int, str],
    slices: dict[str, Any],
) -> dict[str, Any]:
    """Rank a score matrix and evaluate it (all arms output a full ranking)."""
    return evaluate_ranked_output(ranked_labels(scores), gold, seen, unseen, support_map, slices)


def flat_across_slices(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """One flat metric dict for the artefact's per-seed row.

    ``evaluate_ranked_output`` already prefixes every key with its slice name,
    so the slices are simply merged.
    """
    flat: dict[str, float] = {}
    for slice_name in SLICE_NAMES:
        flat.update(metrics[slice_name])
    return flat


def pooled_metrics(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean +- sample SD (ddof = 1), min and max across the five seeds."""
    keys = sorted(per_seed[0])
    pooled: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([row[key] for row in per_seed], dtype=np.float64)
        pooled[key] = {
            "mean": round(float(values.mean()), 6),
            "sd": round(float(values.std(ddof=1)), 6) if values.size > 1 else 0.0,
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return pooled


def vector_comparison(
    baseline: dict[str, np.ndarray],
    treatment: dict[str, np.ndarray],
    metric: str,
    seed: int,
) -> dict[str, Any]:
    """One paired per-document difference on one slice, with its bootstrap CI."""
    base = baseline[metric]
    treat = treatment[metric]
    finite = np.isfinite(base) & np.isfinite(treat)
    base_values = base[finite]
    treat_values = treat[finite]
    bootstrap = paired_bootstrap_ci(
        treat_values - base_values,
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=seed,
    )
    return {
        "baseline": round(float(base_values.mean()), 6),
        "treatment": round(float(treat_values.mean()), 6),
        "delta": round(float(treat_values.mean() - base_values.mean()), 6),
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "documents": int(finite.sum()),
    }


def aggregate_comparison(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """The cross-seed reading: mean +- SD and the count of CIs excluding zero."""
    deltas = np.array([row["delta"] for row in per_seed], dtype=np.float64)
    return {
        "per_seed": per_seed,
        "mean_delta": round(float(deltas.mean()), 6),
        "sd_delta": round(float(deltas.std(ddof=1)), 6) if deltas.size > 1 else 0.0,
        "seeds_ci_positive": int(sum(row["ci_low"] > 0.0 for row in per_seed)),
        "seeds_ci_negative": int(sum(row["ci_high"] < 0.0 for row in per_seed)),
        "seeds": len(per_seed),
    }


def compare_arms(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    seed: int,
    *,
    slices: tuple[str, ...] = ("all_dev", "family_new"),
    metrics: tuple[str, ...] = ("r50", "r_precision", "ndcg10", "unseen_r50"),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Paired per-slice, per-metric comparison of two evaluated arms."""
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for slice_name in slices:
        result[slice_name] = {
            metric: vector_comparison(
                baseline["vectors"][slice_name],
                treatment["vectors"][slice_name],
                metric,
                seed,
            )
            for metric in metrics
        }
    return result


def merge_comparisons(
    store: dict[str, Any], comparison: dict[str, dict[str, dict[str, Any]]]
) -> None:
    """Append one seed's comparisons into the artefact's accumulator."""
    for slice_name, by_metric in comparison.items():
        for metric, row in by_metric.items():
            store.setdefault(slice_name, {}).setdefault(metric, []).append(row)


def finalize_comparisons(store: dict[str, Any]) -> dict[str, Any]:
    """Aggregate every accumulated comparison across seeds."""
    return {
        slice_name: {metric: aggregate_comparison(rows) for metric, rows in by_metric.items()}
        for slice_name, by_metric in store.items()
    }


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write the run artifact (indent 2, UTF-8, one trailing newline)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_header(
    run_id: str,
    *,
    change_set: str,
    extra_config: dict[str, Any],
    slices: dict[str, Any],
) -> dict[str, Any]:
    """The common artifact header for a TC-21 run."""
    return {
        "run_id": run_id,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": change_set,
        "protocol": {
            "seeds": list(SEEDS),
            "labelled_documents_per_seed": LABELLED_DOCUMENTS,
            "label_count": 4_197,
            "label_space": "the sorted union of the train and dev label IDs; test never read",
            "seen_definition": "the label occurs in at least one of the 240 sampled documents",
            "slices": {name: int(slices["masks"][name].sum()) for name in SLICE_NAMES},
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "document, paired within a seed; seed = the sampling seed",
        },
        "config": extra_config,
    }
