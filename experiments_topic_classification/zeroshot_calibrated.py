"""Calibrated zero-shot channel for EURLEX-4K shortlists.

The label-name cosine is on a different scale from the k-NN's vote score.
For each dev document, map each zero-shot label's title-cosine percentile
(among all 4,197 labels) to the k-NN score distribution of the seen labels,
then blend that calibrated score into the k-NN with a strength ``alpha``.

Artifact: ``results/eurlex-zeroshot-calibrated/metrics.json``.  Pre-registered
in ``docs/topic_classification.md`` section 12 before this run.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from dictionary_fusion import bootstrap_difference, per_document_recall  # noqa: E402
from shortlist_baselines import (  # noqa: E402
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
from cluster_rag.topic_classification.metrics import evaluate_ranking  # noqa: E402

OUT = ROOT / "results" / "eurlex-zeroshot-calibrated"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def calibrated_matrix(
    label_scores: np.ndarray,
    knn_scores: np.ndarray,
    seen_columns: list[int],
    unseen_columns: list[int],
) -> np.ndarray:
    """Map each zero-shot label's title-cosine percentile to the k-NN score scale."""
    calibrated = np.zeros_like(knn_scores, dtype=np.float32)
    n_labels = label_scores.shape[1]
    seen_scores = knn_scores[:, seen_columns]
    for row in range(label_scores.shape[0]):
        sorted_titles = np.sort(label_scores[row])
        percentiles = (
            np.searchsorted(sorted_titles, label_scores[row, unseen_columns], side="right")
            / n_labels
        )
        calibrated[row, unseen_columns] = np.quantile(seen_scores[row], percentiles)
    return calibrated


def eval_slice(
    scores: np.ndarray, gold: list[set[int]], bucket_of_label: dict[int, str], mask: np.ndarray
) -> dict[str, Any]:
    """Evaluate one score matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranking(scores[indexes], [gold[index] for index in indexes], bucket_of_label)


def select_alpha(
    knn_scores: np.ndarray,
    calibrated: np.ndarray,
    gold: list[set[int]],
    indexes: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """The alpha that maximises R@50 on ``indexes`` (ties go to the smaller alpha)."""
    scores = {}
    for alpha in ALPHAS:
        fused = knn_scores[indexes] + alpha * calibrated[indexes]
        scores[alpha] = float(
            per_document_recall(fused, [gold[index] for index in indexes], 50).mean()
        )
    best = max(ALPHAS, key=lambda alpha: (scores[alpha], -alpha))
    return best, {str(alpha): round(value, 6) for alpha, value in scores.items()}


def main() -> None:
    """Run the calibration, the grouped CV and the comparison."""
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
    seen_columns = [index for index, label in enumerate(labels) if train_support[label] > 0]
    unseen_columns = [index for index, label in enumerate(labels) if train_support[label] == 0]

    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    label_titles = [vocabulary[label] for label in labels]
    dev_texts = [full_text(record) for record in dev_records]
    train_texts = [full_text(record) for record in train_records]

    print("loading the cached MiniLM vectors ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    document_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION
    )
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_VERSION
    )
    dev_vectors = document_embedder.encode(dev_texts, "document")
    train_vectors = document_embedder.encode(train_texts, "document")
    label_vectors = label_embedder.encode(label_titles, "label")
    label_scores = (dev_vectors @ label_vectors.T).astype(np.float32)

    print("computing the k-NN scores ...", flush=True)
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
    print(f"k-NN in {time.perf_counter() - knn_started:.1f}s", flush=True)

    print("calibrating the zero-shot scores ...", flush=True)
    calibrated = calibrated_matrix(label_scores, knn_scores, seen_columns, unseen_columns)

    dev_families = [title_skeleton(record["title"]) for record in dev_records]
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array([family not in train_families for family in dev_families])
    clean = near_duplicate_cosines(train_records, dev_records) < NEAR_DUP_THRESHOLD
    family_new_clean = family_new & clean

    grid = {}
    for alpha in ALPHAS:
        grid[str(alpha)] = evaluate_ranking(knn_scores + alpha * calibrated, gold, bucket_of_label)

    folds = list(
        GroupKFold(n_splits=5).split(np.arange(len(dev_records)), groups=np.asarray(dev_families))
    )
    cv_scores = np.empty_like(knn_scores)
    selected = []
    for fold_index, (tune_index, validation_index) in enumerate(folds):
        best, tune_scores = select_alpha(knn_scores, calibrated, gold, tune_index)
        cv_scores[validation_index] = (
            knn_scores[validation_index] + best * calibrated[validation_index]
        )
        selected.append(
            {
                "fold": fold_index,
                "documents": int(validation_index.size),
                "alpha": best,
                "tune_r50": tune_scores,
            }
        )

    knn_metrics = evaluate_ranking(knn_scores, gold, bucket_of_label)
    cv_metrics = evaluate_ranking(cv_scores, gold, bucket_of_label)
    bootstrap = {
        "recall@50": bootstrap_difference(
            per_document_recall(cv_scores, gold, 50), per_document_recall(knn_scores, gold, 50)
        ),
    }
    unseen_knn = knn_metrics["diagnostics"]["label_recall@50"]["unseen"]
    unseen_cv = cv_metrics["diagnostics"]["label_recall@50"]["unseen"]
    comparison = {
        "knn_r50": knn_metrics["primary"]["recall@50"],
        "cv_r50": cv_metrics["primary"]["recall@50"],
        "cv_delta_r50": bootstrap["recall@50"],
        "knn_unseen_recall@50": unseen_knn["recall"],
        "knn_unseen_hits": unseen_knn["hits"],
        "cv_unseen_recall@50": unseen_cv["recall"],
        "cv_unseen_hits": unseen_cv["hits"],
        "unseen_assignments": unseen_cv["labels"],
        "selected_alphas": [entry["alpha"] for entry in selected],
        "best_grid_alpha": max(
            ALPHAS, key=lambda alpha: (grid[str(alpha)]["primary"]["recall@50"], -alpha)
        ),
    }

    artifact = {
        "run_id": "eurlex-zeroshot-calibrated",
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "alphas": list(ALPHAS),
            "label_count": len(labels),
            "seen_labels": len(seen_columns),
            "zero_shot_labels": len(unseen_columns),
            "label_buckets": bucket_counts,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "folds": 5,
            "fold_grouping": "title skeleton (whole families)",
            "calibration": (
                "per-document percentile of a zero-shot label's title cosine among all "
                "4,197 label-title cosines, mapped to the k-NN score quantile of the "
                "seen labels"
            ),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "document_cache_misses": document_embedder.cache_misses,
            "label_cache_hits": label_embedder.cache_hits,
            "label_cache_misses": label_embedder.cache_misses,
            "llm_calls": 0,
        },
        "comparison": comparison,
        "grid": grid,
        "cv": {
            "metrics": cv_metrics,
            "family_new": eval_slice(cv_scores, gold, bucket_of_label, family_new),
            "family_new_clean": eval_slice(cv_scores, gold, bucket_of_label, family_new_clean),
            "selected": selected,
        },
        "notes": {
            "metrics": (
                "recall@k and R-precision are macro over documents; nDCG uses binary "
                "relevance; label_recall@50 is micro over gold assignments."
            ),
            "alpha": "final scores = k-NN + alpha * calibrated, on the zero-shot columns only.",
            "bootstrap": "paired bootstrap over documents, CV calibration minus k-NN.",
            "gate": "unseen R@50 >= 0.10 with overall R@50 loss <= 0.01; overall win >= +0.01.",
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
