"""Label-name embeddings: the zero-shot arm for EURLEX-4K shortlists.

Both the dictionary and the document k-NN are seen-label methods: a label with
no train document can never enter their shortlist.  The vocabulary gives every
label a title, so this script embeds the 4,197 train+dev label titles with the
same cached MiniLM model, scores dev documents by cosine to each title, and
evaluates three arms: label-name alone, the k-NN with the zero-shot labels
rescued by their title score, and an untuned equal-weight fusion.

Artifact: ``results/eurlex-labelname/metrics.json``.  Pre-registered in
``docs/topic_classification.md`` section 10 before this script runs.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    KNN_NEIGHBORS,
    VOTE_POWER,
    _embedder_config,
    full_text,
    label_buckets,
    load_split,
)

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.retrieve.knn import knn_vote_scores  # noqa: E402
from cluster_rag.topic_classification.metrics import evaluate_ranking  # noqa: E402

OUT = ROOT / "results" / "eurlex-labelname"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"


def row_zscore(scores: np.ndarray) -> np.ndarray:
    """Per-row z-score (a zero-variance row becomes all zeros)."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (scores - mean) / std


def main() -> None:
    """Run the label-name arms and write the artifact."""
    started = time.perf_counter()
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
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)
    train_support = Counter(label for record_labels in train_labels for label in set(record_labels))
    unseen_columns = [index for index, label in enumerate(labels) if train_support[label] == 0]
    unseen_set = set(unseen_columns)

    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    missing = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} labels have no title: {missing[:5]}")
    label_titles = [vocabulary[label] for label in labels]

    print(
        f"labels {len(labels)}, zero-shot labels {len(unseen_columns)}, "
        f"zero-shot assignments {sum(1 for g in gold for label in g if label in unseen_set)}",
        flush=True,
    )

    print("embedding the label titles ...", flush=True)
    label_started = time.perf_counter()
    backend = HFEmbedder(_embedder_config())
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_VERSION
    )
    label_vectors = label_embedder.encode(label_titles, "label")
    print(
        f"titles embedded in {time.perf_counter() - label_started:.1f}s "
        f"(hits {label_embedder.cache_hits}, misses {label_embedder.cache_misses})",
        flush=True,
    )

    print("loading the cached dev document vectors ...", flush=True)
    doc_started = time.perf_counter()
    document_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION
    )
    dev_texts = [full_text(record) for record in dev_records]
    dev_vectors = document_embedder.encode(dev_texts, "document")
    print(
        f"documents loaded in {time.perf_counter() - doc_started:.1f}s "
        f"(hits {document_embedder.cache_hits}, misses {document_embedder.cache_misses})",
        flush=True,
    )

    print("scoring documents against label names ...", flush=True)
    label_scores = (dev_vectors @ label_vectors.T).astype(np.float32)

    print("computing the k-NN scores ...", flush=True)
    knn_started = time.perf_counter()
    train_texts = [full_text(record) for record in train_records]
    train_vectors = document_embedder.encode(train_texts, "document")
    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True
    knn_scores = np.zeros((len(dev_records), len(labels)), dtype=np.float32)
    for start in range(0, len(dev_records), 256):
        stop = min(start + 256, len(dev_records))
        knn_scores[start:stop] = knn_vote_scores(
            train_vectors,
            train_label_matrix,
            dev_vectors[start:stop],
            k=KNN_NEIGHBORS,
            power=VOTE_POWER,
        )
    print(f"k-NN in {time.perf_counter() - knn_started:.1f}s", flush=True)

    prior = np.zeros(len(labels), dtype=np.float32)
    for label, count in train_support.items():
        prior[label_index[label]] = count
    prior_scores = np.tile(prior, (len(dev_records), 1))

    rescue_scores = knn_scores.copy()
    rescue_scores[:, unseen_columns] = label_scores[:, unseen_columns]
    fusion_scores = row_zscore(knn_scores) + row_zscore(label_scores)

    # The zero-shot-only ranking: the 89 unseen labels alone, for the 87 dev
    # documents that carry one.  This separates "the title signal exists" from
    # "it survives the full 4,197-label ranking".
    zero_shot_documents = [
        index for index, labels_set in enumerate(gold) if labels_set & unseen_set
    ]
    zero_shot_gold = [
        {unseen_columns.index(label) for label in gold[index] if label in unseen_set}
        for index in zero_shot_documents
    ]
    zero_shot_scores = label_scores[zero_shot_documents][:, unseen_columns]
    zero_shot_buckets = dict.fromkeys(range(len(unseen_columns)), "unseen")

    methods: dict[str, Any] = {
        "frequency_prior": {
            "kind": "frequency_prior",
            "arm": "train document frequency",
            "overall": evaluate_ranking(prior_scores, gold, bucket_of_label),
        },
        "embedding_knn": {
            "kind": "embedding_knn",
            "arm": f"MiniLM, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
            "overall": evaluate_ranking(knn_scores, gold, bucket_of_label),
        },
        "labelname_alone": {
            "kind": "label_name",
            "arm": "MiniLM label-title cosine",
            "overall": evaluate_ranking(label_scores, gold, bucket_of_label),
            "zero_shot_only": evaluate_ranking(zero_shot_scores, zero_shot_gold, zero_shot_buckets),
        },
        "knn_plus_labelname_rescue": {
            "kind": "combination",
            "arm": "k-NN for seen labels, label-title cosine for the zero-shot labels",
            "overall": evaluate_ranking(rescue_scores, gold, bucket_of_label),
        },
        "knn_plus_labelname_fusion_untuned": {
            "kind": "combination",
            "arm": "per-row z-score k-NN + per-row z-score label-name (weight 1, untuned)",
            "overall": evaluate_ranking(fusion_scores, gold, bucket_of_label),
        },
    }

    knn = methods["embedding_knn"]["overall"]
    labelname = methods["labelname_alone"]["overall"]
    rescue = methods["knn_plus_labelname_rescue"]["overall"]
    fusion = methods["knn_plus_labelname_fusion_untuned"]["overall"]
    comparison = {
        "unseen_labels": len(unseen_columns),
        "unseen_documents": len(zero_shot_documents),
        "unseen_assignments": sum(len(gold[index] & unseen_set) for index in range(len(gold))),
        "knn_unseen_recall@50": knn["diagnostics"]["label_recall@50"]
        .get("unseen", {})
        .get("recall", 0.0),
        "labelname_unseen_recall@50": labelname["diagnostics"]["label_recall@50"]
        .get("unseen", {})
        .get("recall", 0.0),
        "rescue_unseen_recall@50": rescue["diagnostics"]["label_recall@50"]
        .get("unseen", {})
        .get("recall", 0.0),
        "fusion_unseen_recall@50": fusion["diagnostics"]["label_recall@50"]
        .get("unseen", {})
        .get("recall", 0.0),
        "knn_r50": knn["primary"]["recall@50"],
        "labelname_r50": labelname["primary"]["recall@50"],
        "rescue_r50": rescue["primary"]["recall@50"],
        "fusion_r50": fusion["primary"]["recall@50"],
        "rescue_delta_r50": round(rescue["primary"]["recall@50"] - knn["primary"]["recall@50"], 6),
        "fusion_delta_r50": round(fusion["primary"]["recall@50"] - knn["primary"]["recall@50"], 6),
        "labelname_zero_shot_only_recall@50": methods["labelname_alone"]["zero_shot_only"][
            "primary"
        ]["recall@50"],
    }

    artifact = {
        "run_id": "eurlex-labelname",
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "label_count": len(labels),
            "label_titles": "eurovoc_concepts.jsonl, raw title",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "label_cache_version": LABEL_CACHE_VERSION,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "label_buckets": bucket_counts,
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "label_cache_hits": label_embedder.cache_hits,
            "label_cache_misses": label_embedder.cache_misses,
            "document_cache_hits": document_embedder.cache_hits,
            "document_cache_misses": document_embedder.cache_misses,
            "llm_calls": 0,
        },
        "comparison": comparison,
        "methods": methods,
        "notes": {
            "metrics": (
                "recall@k and R-precision are macro over documents; nDCG uses binary "
                "relevance; label_recall@50 is micro over gold assignments."
            ),
            "zero_shot_only": (
                "the 89 unseen labels alone, for the 87 dev documents with an unseen "
                "gold label; this is the title signal before the 4,197-label ranking."
            ),
            "rescue": (
                "the k-NN's seen-label scores are unchanged; the 89 zero-shot labels' "
                "scores are replaced by their label-title cosine."
            ),
            "fusion": "untuned equal weight; no weight was selected on dev.",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
