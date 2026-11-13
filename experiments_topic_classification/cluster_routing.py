"""TC-15 = menu item 9: cluster routing and label pruning on EURLEX-4K.

Cluster the 45,000 train documents with MiniBatchKMeans (K=500) on the cached
MiniLM vectors, give each cluster the labels its documents carry, route a dev
document to its C nearest clusters by centroid cosine, and rank only the
routed labels first (C=3 primary, C=1 secondary).  Compare with the full
lexical k-NN ranking on all_dev / family_new / family_new_clean.

Pre-registered in ``docs/topic_classification.md`` section 21 before this script
runs.  Artifact: ``results/eurlex-cluster-routing/metrics.json``.  Splits:
train = cluster (45,000), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import title_skeleton  # noqa: E402
from lexical_rrf import TFIDF_CONFIG, lexical_scores  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    _embedder_config,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranked,
    paired_bootstrap_ci,
    per_document_ndcg_at_k,
    per_document_r_precision,
    per_document_recall_at_k,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-cluster-routing"
RUN_ID = "eurlex-cluster-routing"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
N_CLUSTERS = 500
ROUTE_COUNTS = (1, 3)
PRIMARY_ROUTE = 3
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SHORTLIST_BAR = 0.01
LOSS_TOLERANCE = 0.01
EFFICIENCY_LOSS = 0.005
EFFICIENCY_SPACE = 0.25
WIN_SPACE = 0.50
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")


def slice_view(
    ranked: np.ndarray, gold: list[set[int]], mask: np.ndarray, bucket_of_label: dict[int, str]
) -> dict[str, Any]:
    """Evaluate a ranked matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranked(ranked[indexes], [gold[index] for index in indexes], bucket_of_label)


def routed_ranking(scores: np.ndarray, routed: np.ndarray) -> np.ndarray:
    """Rank the routed labels first (by score), then the rest (by score)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.empty_like(order)
    for row in range(scores.shape[0]):
        in_mask = routed[row][order[row]]
        ranked[row] = np.concatenate([order[row][in_mask], order[row][~in_mask]])
    return ranked


def routing_recall(routed: np.ndarray, gold: list[set[int]]) -> dict[str, float]:
    """The share of gold labels inside the routed set (macro and micro)."""
    macro = np.empty(len(gold), dtype=np.float64)
    hits = 0
    total = 0
    for row, labels in enumerate(gold):
        inside = sum(1 for label in labels if routed[row, label])
        macro[row] = inside / len(labels)
        hits += inside
        total += len(labels)
    return {
        "macro": round(float(macro.mean()), 6),
        "micro": round(hits / total, 6) if total else 0.0,
        "gold_assignments": total,
    }


def bootstrap_ranked(
    base_ranked: np.ndarray,
    treatment_ranked: np.ndarray,
    gold: list[set[int]],
    mask: np.ndarray,
    *,
    metric: str,
    k: int = 10,
) -> dict[str, Any]:
    """Paired per-document difference of one metric between two ranked matrices."""
    indexes = np.flatnonzero(mask)
    slice_gold = [gold[index] for index in indexes]

    def _values(ranked: np.ndarray) -> np.ndarray:
        if metric == "ndcg":
            return per_document_ndcg_at_k(ranked[indexes], slice_gold, k)
        if metric == "r_precision":
            return per_document_r_precision(ranked[indexes], slice_gold)
        return per_document_recall_at_k(ranked[indexes], slice_gold, 50)

    base_values = _values(base_ranked)
    treatment_values = _values(treatment_ranked)
    return {
        "baseline": round(float(base_values.mean()), 6),
        "treatment": round(float(treatment_values.mean()), 6),
        "delta": round(float(treatment_values.mean() - base_values.mean()), 6),
        "bootstrap": paired_bootstrap_ci(
            treatment_values - base_values, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
    }


def main() -> None:
    """Cluster, route, prune and compare."""
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
    dev_gold = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]

    print("computing the family split and the clean slice ...", flush=True)
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    slices = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_new_clean": family_new & clean,
    }
    print(
        "slices: " + ", ".join(f"{name} {int(mask.sum())}" for name, mask in slices.items()),
        flush=True,
    )

    print("loading the cached document vectors ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION)
    vectors = embedder.encode(train_texts + dev_texts, "document")
    train_vectors = vectors[: len(train_records)]
    dev_vectors = vectors[len(train_records) :]
    print(f"  cache hits {embedder.cache_hits}", flush=True)

    print(f"clustering the train documents (K={N_CLUSTERS}) ...", flush=True)
    cluster_started = time.perf_counter()
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        random_state=SEED,
    )
    train_clusters = kmeans.fit_predict(train_vectors)
    cluster_seconds = time.perf_counter() - cluster_started
    print(
        f"  inertia {kmeans.inertia_:.1f}, {cluster_seconds:.1f}s, "
        f"cluster sizes {int(np.bincount(train_clusters).min())}–"
        f"{int(np.bincount(train_clusters).max())}",
        flush=True,
    )

    print("building the cluster label sets ...", flush=True)
    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True
    cluster_labels = np.zeros((N_CLUSTERS, len(labels)), dtype=bool)
    for cluster in range(N_CLUSTERS):
        cluster_labels[cluster] = train_label_matrix[train_clusters == cluster].any(axis=0)
    cluster_label_sizes = cluster_labels.sum(axis=1)
    print(
        f"  label sets {int(cluster_label_sizes.min())}–{int(cluster_label_sizes.max())} "
        f"(mean {cluster_label_sizes.mean():.0f})",
        flush=True,
    )

    print("routing the dev documents ...", flush=True)
    routing_started = time.perf_counter()
    centroid_norms = np.linalg.norm(kmeans.cluster_centers_, axis=1, keepdims=True)
    normalized_centroids = kmeans.cluster_centers_ / np.maximum(centroid_norms, 1e-12)
    routing_scores = dev_vectors @ normalized_centroids.T
    route_order = np.argsort(-routing_scores, axis=1, kind="stable")
    routed_masks: dict[int, np.ndarray] = {}
    for route_count in ROUTE_COUNTS:
        masks = np.zeros((len(dev_records), len(labels)), dtype=bool)
        for row in range(len(dev_records)):
            clusters = route_order[row, :route_count]
            masks[row] = cluster_labels[clusters].any(axis=0)
        routed_masks[route_count] = masks
    routing_seconds = time.perf_counter() - routing_started
    print(f"  routed in {routing_seconds:.1f}s", flush=True)

    print("building the lexical base scores ...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=TFIDF_CONFIG["lowercase"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    dev_tfidf = vectorizer.transform(dev_texts)
    lexical = lexical_scores(dev_tfidf, train_tfidf, train_label_matrix)
    full_ranked = ranked_labels(lexical)

    methods: dict[str, Any] = {
        "full_lexical_knn": {
            "kind": "knn",
            "arm": "lexical TF-IDF k-NN over all 4,197 labels",
            "slices": {
                name: slice_view(full_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        }
    }
    comparisons: dict[str, Any] = {}
    routed_stats: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}

    for route_count in ROUTE_COUNTS:
        print(f"=== C={route_count}: pruning and evaluating ===", flush=True)
        routed = routed_masks[route_count]
        pruned_ranked = routed_ranking(lexical, routed)
        methods[f"routed_c{route_count}"] = {
            "kind": "cluster_routed",
            "arm": f"route to the {route_count} nearest cluster(s), rank their labels first",
            "slices": {
                name: slice_view(pruned_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        }
        space = routed.sum(axis=1)
        routed_stats[str(route_count)] = {
            "mean_labels": round(float(space.mean()), 1),
            "median_labels": int(np.median(space)),
            "share_of_label_space": round(float(space.mean() / len(labels)), 4),
            "routing_recall": routing_recall(routed, dev_gold),
            "per_slice_routing_recall": {
                name: routing_recall(
                    routed[np.flatnonzero(mask)], [dev_gold[i] for i in np.flatnonzero(mask)]
                )
                for name, mask in slices.items()
            },
        }
        comparisons[f"routed_c{route_count}"] = {
            name: {
                "r50": bootstrap_ranked(full_ranked, pruned_ranked, dev_gold, mask, metric="r50"),
                "r_precision": bootstrap_ranked(
                    full_ranked, pruned_ranked, dev_gold, mask, metric="r_precision"
                ),
                "ndcg": bootstrap_ranked(full_ranked, pruned_ranked, dev_gold, mask, metric="ndcg"),
            }
            for name, mask in slices.items()
        }

        family_new_comparison = comparisons[f"routed_c{route_count}"]["family_new"]
        share = routed_stats[str(route_count)]["share_of_label_space"]
        conditions = {
            "r50_gain_at_least_0.01": family_new_comparison["r50"]["delta"] >= SHORTLIST_BAR,
            "r50_ci_excludes_zero": family_new_comparison["r50"]["bootstrap"]["ci_low"] > 0.0,
            "r_precision_loss_at_most_0.01": family_new_comparison["r_precision"]["delta"]
            >= -LOSS_TOLERANCE,
            "ndcg_loss_at_most_0.01": family_new_comparison["ndcg"]["delta"] >= -LOSS_TOLERANCE,
            "routed_space_at_most_50pct": share <= WIN_SPACE,
        }
        efficiency = (
            family_new_comparison["r50"]["delta"] >= -EFFICIENCY_LOSS and share <= EFFICIENCY_SPACE
        )
        verdicts[f"c{route_count}"] = {
            "bar_met": all(conditions.values()),
            "conditions": conditions,
            "efficiency_only": bool(efficiency),
            "family_new_r50": family_new_comparison["r50"]["treatment"],
            "family_new_r50_delta": family_new_comparison["r50"]["delta"],
            "family_new_r50_95ci": [
                family_new_comparison["r50"]["bootstrap"]["ci_low"],
                family_new_comparison["r50"]["bootstrap"]["ci_high"],
            ],
            "family_new_r_precision_delta": family_new_comparison["r_precision"]["delta"],
            "family_new_ndcg@10_delta": family_new_comparison["ndcg"]["delta"],
            "routed_space_share": share,
            "routing_recall_micro": routed_stats[str(route_count)]["routing_recall"]["micro"],
            "outcome": (
                "win" if all(conditions.values()) else ("efficiency_only" if efficiency else "null")
            ),
        }
        print(
            f"  space {space.mean():.0f}/{len(labels)} ({share:.1%}), routing recall "
            f"{routed_stats[str(route_count)]['routing_recall']['micro']:.4f}, "
            f"family_new R@50 delta {family_new_comparison['r50']['delta']:+.4f}",
            flush=True,
        )

    primary = verdicts[f"c{PRIMARY_ROUTE}"]
    secondary = verdicts["c1"]
    if primary["bar_met"]:
        outcome = "win"
    elif secondary["bar_met"]:
        outcome = "secondary_win_only"
    elif primary["efficiency_only"]:
        outcome = "efficiency_only"
    else:
        outcome = "null"

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-15 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 9)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "clustering": "MiniBatchKMeans on the cached MiniLM document vectors",
            "n_clusters": N_CLUSTERS,
            "kmeans": {
                "batch_size": 4096,
                "n_init": 3,
                "max_iter": 100,
                "random_state": SEED,
            },
            "routing": "top-C clusters by centroid cosine; C=3 primary, C=1 secondary",
            "cluster_labels": "labels carried by at least one train document of the cluster",
            "ranking": "routed labels first by lexical score, then the rest by lexical score",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "clusters": {
            "n_clusters": N_CLUSTERS,
            "inertia": round(float(kmeans.inertia_), 1),
            "seconds": round(cluster_seconds, 1),
            "label_set_mean": round(float(cluster_label_sizes.mean()), 1),
            "label_set_min": int(cluster_label_sizes.min()),
            "label_set_max": int(cluster_label_sizes.max()),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "cluster_seconds": round(cluster_seconds, 1),
            "routing_seconds": round(routing_seconds, 1),
            "document_cache_hits": embedder.cache_hits,
            "llm_calls": 0,
        },
        "routed": routed_stats,
        "methods": methods,
        "comparison": comparisons,
        "verdict": {
            "primary_route": PRIMARY_ROUTE,
            "primary": primary,
            "secondary_route": 1,
            "secondary": secondary,
            "outcome": outcome,
        },
        "notes": {
            "bar": (
                "a win requires family_new R@50 +0.01 with the interval excluding 0, no "
                "R-precision/nDCG loss beyond 0.01, and a routed space at most 50% of "
                "4,197; efficiency_only allows a 0.005 R@50 loss at a space of at most 25%"
            ),
            "base": (
                "the lexical k-NN (the strongest single retriever); the learned pack is "
                "the current best overall but routing is measured against the full "
                "ranking of the same base"
            ),
            "ranking": (
                "the pruned labels are ranked first; the rest follow by lexical score, "
                "so R@100/F1-style tail measures remain comparable"
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "routed": {
                    key: {
                        "mean_labels": value["mean_labels"],
                        "share": value["share_of_label_space"],
                        "routing_recall": value["routing_recall"]["micro"],
                    }
                    for key, value in routed_stats.items()
                },
                "comparison": {
                    key: {
                        name: {
                            metric: values[metric]["delta"]
                            for metric in ("r50", "r_precision", "ndcg")
                        }
                        for name, values in by_slice.items()
                    }
                    for key, by_slice in comparisons.items()
                },
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
