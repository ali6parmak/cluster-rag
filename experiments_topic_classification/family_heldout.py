"""Family-held-out evaluation for the EURLEX-4K shortlist baselines.

The random dev split is not a clean "new document" test: families (the audit's
title skeleton) and near duplicates cross the split.  This script evaluates the
same dictionary arms and the same cached MiniLM k-NN on the dev documents whose
family never appears in the train index, and reports the family effect.

Artifact: ``results/eurlex-family-heldout/metrics.json``.  Pre-registered in
``docs/topic_classification.md`` section 9 before this script runs.
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

from audit_eurlex import title_skeleton  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    DICTIONARY_ARMS,
    KNN_NEIGHBORS,
    NEAR_DUP_THRESHOLD,
    VOTE_POWER,
    _embedder_config,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.retrieve.knn import knn_vote_scores  # noqa: E402
from cluster_rag.text_processors.features.corpus_map import build_corpus_map  # noqa: E402
from cluster_rag.text_processors.lexical.corpus_map_score import score_cells  # noqa: E402
from cluster_rag.topic_classification.metrics import evaluate_ranking  # noqa: E402

OUT = ROOT / "results" / "eurlex-family-heldout"
EMBEDDING_CACHE_VERSION = "eurlex-4k-v1"


def evaluate_slice(
    scores: np.ndarray,
    gold: list[set[int]],
    bucket_of_label: dict[int, str],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one score matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranking(scores[indexes], [gold[index] for index in indexes], bucket_of_label)


def main() -> None:
    """Run the family-held-out evaluation and write the artifact."""
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
    seen_labels = {label_index[label] for label, count in train_support.items() if count > 0}

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]

    print("computing the family split ...", flush=True)
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    family_seen = ~family_new
    print(f"family_new {int(family_new.sum())}, family_seen {int(family_seen.sum())}", flush=True)

    print("computing the near-duplicate clean slice ...", flush=True)
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    family_new_clean = family_new & clean
    print(
        f"family_new_clean {int(family_new_clean.sum())}; clean overall {int(clean.sum())}",
        flush=True,
    )

    # The zero-shot labels are the gold assignments with no train example; the
    # seen-label view removes them so family generalisation is not confused
    # with label absence.
    gold_seen = [{label for label in labels_set if label in seen_labels} for labels_set in gold]
    seen_indexes = [index for index in np.flatnonzero(family_new) if gold_seen[int(index)]]
    family_new_assignments = sum(len(gold[index]) for index in np.flatnonzero(family_new))
    family_new_zero_shot = sum(
        len(gold[index] - gold_seen[index]) for index in np.flatnonzero(family_new)
    )

    slices = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_seen": family_seen,
        "family_new_clean": family_new_clean,
    }

    print("building the train-only corpus dictionary ...", flush=True)
    corpus_map = build_corpus_map(train_texts, train_labels, labels)
    methods: dict[str, Any] = {}
    for arm in DICTIONARY_ARMS:
        arm_started = time.perf_counter()
        scores = score_cells(dev_texts, corpus_map, scoring=arm)
        method: dict[str, Any] = {
            "kind": "dictionary",
            "arm": arm,
            "seconds": round(time.perf_counter() - arm_started, 1),
        }
        for slice_name, mask in slices.items():
            method[slice_name] = evaluate_slice(scores, gold, bucket_of_label, mask)
        method["family_new_seen_labels"] = evaluate_ranking(
            scores[seen_indexes],
            [gold_seen[int(index)] for index in seen_indexes],
            bucket_of_label,
        )
        methods[f"dictionary_{arm}"] = method
        print(
            f"  {arm}: R@50 family_new "
            f"{method['family_new']['primary']['recall@50']:.4f} "
            f"({method['seconds']:.0f}s)",
            flush=True,
        )
        del scores

    print("loading the cached MiniLM vectors ...", flush=True)
    embedding_started = time.perf_counter()
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=EMBEDDING_CACHE_VERSION
    )
    train_vectors = embedder.encode(train_texts, "document")
    dev_vectors = embedder.encode(dev_texts, "document")
    print(
        f"loaded in {time.perf_counter() - embedding_started:.1f}s "
        f"(hits {embedder.cache_hits}, misses {embedder.cache_misses})",
        flush=True,
    )

    print("computing the k-NN ...", flush=True)
    knn_started = time.perf_counter()
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
    methods["embedding_knn"] = {
        "kind": "embedding_knn",
        "arm": f"MiniLM, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
        "seconds": round(time.perf_counter() - knn_started, 1),
        **{
            name: evaluate_slice(knn_scores, gold, bucket_of_label, mask)
            for name, mask in slices.items()
        },
        "family_new_seen_labels": evaluate_ranking(
            knn_scores[seen_indexes],
            [gold_seen[int(index)] for index in seen_indexes],
            bucket_of_label,
        ),
    }
    print(
        f"  k-NN: R@50 family_new "
        f"{methods['embedding_knn']['family_new']['primary']['recall@50']:.4f}",
        flush=True,
    )
    del knn_scores

    print("computing the frequency prior ...", flush=True)
    prior = np.zeros(len(labels), dtype=np.float32)
    for label, count in train_support.items():
        prior[label_index[label]] = count
    prior_scores = np.tile(prior, (len(dev_records), 1))
    methods["frequency_prior"] = {
        "kind": "frequency_prior",
        "arm": "train document frequency",
        "seconds": 0.0,
        **{
            name: evaluate_slice(prior_scores, gold, bucket_of_label, mask)
            for name, mask in slices.items()
        },
        "family_new_seen_labels": evaluate_ranking(
            prior_scores[seen_indexes],
            [gold_seen[int(index)] for index in seen_indexes],
            bucket_of_label,
        ),
    }

    knn = methods["embedding_knn"]
    dictionary_names = [f"dictionary_{arm}" for arm in DICTIONARY_ARMS]
    best_dictionary = max(
        dictionary_names, key=lambda name: methods[name]["family_new"]["primary"]["recall@50"]
    )
    comparison = {
        "knn_r50_all_dev": knn["all_dev"]["primary"]["recall@50"],
        "knn_r50_family_new": knn["family_new"]["primary"]["recall@50"],
        "knn_r50_family_seen": knn["family_seen"]["primary"]["recall@50"],
        "knn_r50_family_new_clean": knn["family_new_clean"]["primary"]["recall@50"],
        "family_effect_r50": round(
            knn["all_dev"]["primary"]["recall@50"] - knn["family_new"]["primary"]["recall@50"], 6
        ),
        "best_dictionary_arm": best_dictionary,
        "best_dictionary_r50_family_new": methods[best_dictionary]["family_new"]["primary"][
            "recall@50"
        ],
        "knn_minus_best_dictionary_family_new": round(
            knn["family_new"]["primary"]["recall@50"]
            - methods[best_dictionary]["family_new"]["primary"]["recall@50"],
            6,
        ),
        "knn_minus_best_dictionary_family_new_clean": round(
            knn["family_new_clean"]["primary"]["recall@50"]
            - methods[best_dictionary]["family_new_clean"]["primary"]["recall@50"],
            6,
        ),
    }

    artifact = {
        "run_id": "eurlex-family-heldout",
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "family_definition": (
                "the audit's title skeleton: lowercase, digits -> '#', punctuation removed, "
                "whitespace collapsed"
            ),
            "family_new": "dev documents whose skeleton is absent from train",
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
            "embedding_cache_version": EMBEDDING_CACHE_VERSION,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "family_new_assignments": int(family_new_assignments),
            "family_new_zero_shot_assignments": int(family_new_zero_shot),
            "family_new_zero_shot_share": round(family_new_zero_shot / family_new_assignments, 4),
            "family_new_seen_label_documents": len(seen_indexes),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "embedding_cache_hits": embedder.cache_hits,
            "embedding_cache_misses": embedder.cache_misses,
            "dictionary_pairs": corpus_map.n_pairs,
            "llm_calls": 0,
        },
        "comparison": comparison,
        "methods": methods,
        "notes": {
            "metrics": (
                "recall@k and R-precision are macro over documents; nDCG uses binary "
                "relevance; label_recall@50 is micro over gold assignments."
            ),
            "family_new_clean": (
                "family_new AND the audit's clean near-duplicate slice (best train TF-IDF "
                "cosine < 0.90); the strongest new-document view."
            ),
            "family_new_seen_labels": (
                "family_new restricted to gold labels with at least one train example; "
                "the zero-shot labels are removed."
            ),
            "dictionary": (
                "train-only corpus map; labels with no train example carry the row floor. "
                "sum_zscore_positive was the pre-registered primary in section 8."
            ),
            "embedding": (
                "MiniLM-L6-v2, whole-document text, 256-token context, cosine k-NN, "
                "similarity-weighted votes."
            ),
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
