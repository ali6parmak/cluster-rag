"""One-parameter dictionary + k-NN fusion for EURLEX-4K shortlists.

Per document, z-score the k-NN and dictionary score matrices and combine them
as ``z(k-NN) + w * z(dictionary)``.  The weight is selected by 5-fold group
cross-validation on dev (families held out), and the fused ranking is compared
with the k-NN alone.  Artifact: ``results/eurlex-fusion/metrics.json``.

Pre-registered in ``docs/topic_classification.md`` section 11 before this run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

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
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranking,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-fusion"
SCORES = OUT / "scores"
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
N_BOOTSTRAP = 2000
SEED = 42


def row_zscore(scores: np.ndarray) -> np.ndarray:
    """Per-row z-score (zero-variance rows become zeros)."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return ((scores - mean) / std).astype(np.float32)


def per_document_recall(scores: np.ndarray, gold: list[set[int]], k: int) -> np.ndarray:
    """Per-document recall@k."""
    ranked = ranked_labels(scores)
    return np.array(
        [
            len(set(ranked[row, : min(k, ranked.shape[1])].tolist()) & labels) / len(labels)
            for row, labels in enumerate(gold)
        ],
        dtype=np.float64,
    )


def per_document_r_precision(scores: np.ndarray, gold: list[set[int]]) -> np.ndarray:
    """Per-document R-precision."""
    ranked = ranked_labels(scores)
    return np.array(
        [
            len(set(ranked[row, : min(len(labels), ranked.shape[1])].tolist()) & labels)
            / len(labels)
            for row, labels in enumerate(gold)
        ],
        dtype=np.float64,
    )


def per_document_ndcg(scores: np.ndarray, gold: list[set[int]], k: int) -> np.ndarray:
    """Per-document nDCG@k with binary relevance."""
    ranked = ranked_labels(scores)
    values = []
    for row, labels in enumerate(gold):
        top = ranked[row, : min(k, ranked.shape[1])].tolist()
        dcg = sum(
            1.0 / np.log2(position + 2) for position, label in enumerate(top) if label in labels
        )
        idcg = sum(1.0 / np.log2(position + 2) for position in range(min(k, len(labels))))
        values.append(dcg / idcg if idcg else 0.0)
    return np.array(values, dtype=np.float64)


def bootstrap_difference(
    arm: np.ndarray, baseline: np.ndarray, *, n_resamples: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, float]:
    """Paired bootstrap of the mean difference (arm - baseline)."""
    difference = arm - baseline
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, difference.size, difference.size)
        means[index] = difference[sample].mean()
    return {
        "delta": round(float(difference.mean()), 6),
        "ci_low": round(float(np.quantile(means, 0.025)), 6),
        "ci_high": round(float(np.quantile(means, 0.975)), 6),
        "n_resamples": n_resamples,
    }


def select_weight(
    z_knn: np.ndarray, z_dictionary: np.ndarray, gold: list[set[int]], indexes: np.ndarray
) -> tuple[float, dict[str, float]]:
    """The weight that maximises R@50 on ``indexes`` (ties go to the smaller weight)."""
    scores = {}
    for weight in WEIGHTS:
        fused = z_knn[indexes] + weight * z_dictionary[indexes]
        scores[weight] = float(
            per_document_recall(fused, [gold[index] for index in indexes], 50).mean()
        )
    best = max(WEIGHTS, key=lambda weight: (scores[weight], -weight))
    return best, {str(weight): round(value, 6) for weight, value in scores.items()}


def evaluate_slice(
    scores: np.ndarray, gold: list[set[int]], bucket_of_label: dict[int, str], mask: np.ndarray
) -> dict[str, Any]:
    """Evaluate one score matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranking(scores[indexes], [gold[index] for index in indexes], bucket_of_label)


def main() -> None:
    """Run the fusion grid, the group cross-validation and the comparison."""
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

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]
    family_keys = [title_skeleton(record["title"]) for record in dev_records]
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array([key not in train_families for key in family_keys])
    clean = near_duplicate_cosines(train_records, dev_records) < NEAR_DUP_THRESHOLD
    family_new_clean = family_new & clean

    print("building the train-only corpus dictionary ...", flush=True)
    dictionary_started = time.perf_counter()
    corpus_map = build_corpus_map(train_texts, train_labels, labels)
    print(
        f"map in {time.perf_counter() - dictionary_started:.1f}s: {corpus_map.n_pairs} pairs",
        flush=True,
    )

    dictionary_scores: dict[str, np.ndarray] = {}
    for arm in DICTIONARY_ARMS:
        arm_started = time.perf_counter()
        dictionary_scores[arm] = score_cells(dev_texts, corpus_map, scoring=arm).astype(np.float32)
        print(f"  {arm}: {time.perf_counter() - arm_started:.1f}s", flush=True)

    print("loading the cached MiniLM vectors and computing the k-NN ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION)
    train_vectors = embedder.encode(train_texts, "document")
    dev_vectors = embedder.encode(dev_texts, "document")
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
    z_knn = row_zscore(knn_scores)

    groups = np.asarray(family_keys)
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(dev_records)), groups=groups))
    results: dict[str, Any] = {}
    for arm, scores in dictionary_scores.items():
        print(f"fusion for {arm} ...", flush=True)
        z_dictionary = row_zscore(scores)
        grid = {}
        for weight in WEIGHTS:
            fused = z_knn + weight * z_dictionary
            grid[str(weight)] = evaluate_ranking(fused, gold, bucket_of_label)

        cv_scores = np.empty_like(z_knn)
        selected: list[dict[str, Any]] = []
        for fold_index, (tune_index, validation_index) in enumerate(folds):
            best, tune_scores = select_weight(z_knn, z_dictionary, gold, tune_index)
            cv_scores[validation_index] = (
                z_knn[validation_index] + best * z_dictionary[validation_index]
            )
            selected.append(
                {
                    "fold": fold_index,
                    "documents": int(validation_index.size),
                    "weight": best,
                    "tune_r50": tune_scores,
                }
            )
        cv_metrics = evaluate_ranking(cv_scores, gold, bucket_of_label)
        cv_family_new = evaluate_slice(cv_scores, gold, bucket_of_label, family_new)
        cv_family_new_clean = evaluate_slice(cv_scores, gold, bucket_of_label, family_new_clean)
        knn_metrics = evaluate_ranking(knn_scores, gold, bucket_of_label)
        bootstrap = {
            "recall@50": bootstrap_difference(
                per_document_recall(cv_scores, gold, 50), per_document_recall(knn_scores, gold, 50)
            ),
            "r_precision": bootstrap_difference(
                per_document_r_precision(cv_scores, gold),
                per_document_r_precision(knn_scores, gold),
            ),
            "ndcg@10": bootstrap_difference(
                per_document_ndcg(cv_scores, gold, 10), per_document_ndcg(knn_scores, gold, 10)
            ),
        }
        best_grid_weight = max(
            WEIGHTS, key=lambda weight: (grid[str(weight)]["primary"]["recall@50"], -weight)
        )
        results[arm] = {
            "grid": grid,
            "best_grid_weight": best_grid_weight,
            "cv": cv_metrics,
            "cv_family_new": cv_family_new,
            "cv_family_new_clean": cv_family_new_clean,
            "selected_weights": selected,
            "bootstrap_vs_knn": bootstrap,
            "primary_arm": arm == "sum_zscore_positive",
        }
        print(
            f"  {arm}: CV R@50 {cv_metrics['primary']['recall@50']:.4f} "
            f"(k-NN {knn_metrics['primary']['recall@50']:.4f}), "
            f"delta {bootstrap['recall@50']['delta']:+.4f} "
            f"[{bootstrap['recall@50']['ci_low']:+.4f}, {bootstrap['recall@50']['ci_high']:+.4f}]",
            flush=True,
        )

    knn_metrics = evaluate_ranking(knn_scores, gold, bucket_of_label)
    comparison = {
        "knn_r50": knn_metrics["primary"]["recall@50"],
        "knn_family_new_r50": evaluate_slice(knn_scores, gold, bucket_of_label, family_new)[
            "primary"
        ]["recall@50"],
        "knn_family_new_clean_r50": evaluate_slice(
            knn_scores, gold, bucket_of_label, family_new_clean
        )["primary"]["recall@50"],
        "primary_arm": "sum_zscore_positive",
        "primary_cv_r50": results["sum_zscore_positive"]["cv"]["primary"]["recall@50"],
        "primary_cv_delta": results["sum_zscore_positive"]["bootstrap_vs_knn"]["recall@50"],
        "primary_cv_family_new_r50": results["sum_zscore_positive"]["cv_family_new"]["primary"][
            "recall@50"
        ],
        "primary_selected_weights": [
            entry["weight"] for entry in results["sum_zscore_positive"]["selected_weights"]
        ],
        "best_grid_weight": results["sum_zscore_positive"]["best_grid_weight"],
        "best_grid_r50": results["sum_zscore_positive"]["grid"][
            str(results["sum_zscore_positive"]["best_grid_weight"])
        ]["primary"]["recall@50"],
    }

    artifact = {
        "run_id": "eurlex-fusion",
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "weights": list(WEIGHTS),
            "dictionary_arms": list(DICTIONARY_ARMS),
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "folds": 5,
            "fold_grouping": "title skeleton (whole families)",
            "bootstrap_resamples": N_BOOTSTRAP,
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
            "family_new_documents": int(family_new.sum()),
            "family_new_clean_documents": int(family_new_clean.sum()),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "dictionary_pairs": corpus_map.n_pairs,
            "embedding_cache_hits": embedder.cache_hits,
            "embedding_cache_misses": embedder.cache_misses,
            "llm_calls": 0,
        },
        "comparison": comparison,
        "results": results,
        "notes": {
            "metrics": (
                "the grid is the full-dev number at each weight (tuned on dev); the CV "
                "number selects the weight on the other four family-grouped folds and "
                "evaluates the held-out fold, so it is not tuned on the evaluated documents."
            ),
            "fusion": (
                "per-document z-score of the k-NN + w * per-document z-score of the dictionary."
            ),
            "bootstrap": "paired bootstrap over documents (2,000 resamples), arm - k-NN.",
            "stage2": (
                "the candidate-level combiner is licensed only if the primary CV R@50 gate passes."
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(SCORES / "knn.npy", knn_scores)
    for arm, scores in dictionary_scores.items():
        np.save(SCORES / f"dictionary_{arm}.npy", scores)
    (SCORES / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (SCORES / "gold.json").write_text(json.dumps([sorted(g) for g in gold]), encoding="utf-8")
    np.save(SCORES / "family_new.npy", family_new)
    np.save(SCORES / "clean.npy", clean)
    print(json.dumps(comparison, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
