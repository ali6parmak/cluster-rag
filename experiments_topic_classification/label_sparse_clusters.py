"""TC-21 sub-task 3: Arm B — the soft word-cluster bridge.

The train text (45,000 documents, **no labels**) is clustered with TF-IDF word
1-2 grams + MiniBatchKMeans.  A document's soft cluster membership is the
positive cosine to the centroids, L1-normalised; a label's **text** is attached
to the clusters the same way.  The cluster score is their overlap, so every
label — seen or unseen — gets a score.  The fixed hybrid keeps the retrieval
z-score for the seen labels and the cluster z-score for the unseen labels.

Pre-registered in ``docs/topic_classification.md`` section 27.4 before this
script runs.  Artifact: ``results/eurlex-label-sparse-clusters/metrics.json``.
The cluster fit reads train **text** only (never a label); the other 44,760
train documents' labels are never read; test is never read.
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

from label_sparse_common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    LABEL_TEXT_VARIANTS,
    SEEDS,
    SLICE_NAMES,
    EmbeddingBank,
    build_slices,
    compare_arms,
    embedding_knn,
    evaluate_scores,
    finalize_comparisons,
    flat_across_slices,
    label_classes,
    label_texts,
    lexical_knn,
    load_dataset,
    merge_comparisons,
    pooled_metrics,
    sample_documents,
    support_bucket_map,
    write_artifact,
)

from cluster_rag.topic_classification.clusters import (  # noqa: E402
    cluster_affinity_scores,
    cluster_hybrid,
    home_clusters,
    positive_row_normalize,
    top_cluster_hits,
    top_clusters,
)
from cluster_rag.topic_classification.transfer import (  # noqa: E402
    disjoint_hybrid,
    neighbour_transfer,
    row_zscore_columns,
    top_k_positions,
    transfer_reading,
)

OUT = ROOT / "results" / "eurlex-label-sparse-clusters"
RUN_ID = "eurlex-label-sparse-clusters"
CLUSTER_TFIDF = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200_000,
    "sublinear_tf": True,
}
KMEANS = {
    "batch_size": 4096,
    "n_init": 3,
    "max_iter": 100,
    "random_state": 42,
}
PRIMARY_K = 200
K_GRID = (200, 100, 300)
TOP_CLUSTERS = 3
TRANSFER_K = 10
BOOTSTRAP_METRICS = ("r50", "r_precision", "ndcg10", "unseen_r50")
COMPARISON_SLICES = ("all_dev", "family_new")
TRANSFER_ARM = "transfer_hybrid_title_parents_k10_lexical"
CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "cluster_hybrid_k200_title_parents",
        "k": 200,
        "variant": "title_parents",
        "primary": True,
    },
    {
        "name": "cluster_hybrid_k100_title_parents",
        "k": 100,
        "variant": "title_parents",
        "primary": False,
    },
    {
        "name": "cluster_hybrid_k300_title_parents",
        "k": 300,
        "variant": "title_parents",
        "primary": False,
    },
    {
        "name": "cluster_hybrid_k200_title",
        "k": 200,
        "variant": "title",
        "primary": False,
    },
    {
        "name": "cluster_hybrid_k200_title_parents_definition",
        "k": 200,
        "variant": "title_parents_definition",
        "primary": False,
    },
)


def label_tfidf(texts: list[str]) -> Any:
    """The label-text TF-IDF used for the transfer comparator (Arm A's space)."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return vectorizer.fit_transform(texts)


def normalize_centroids(centers: np.ndarray) -> np.ndarray:
    """L2-normalise the k-means centroids (cosine to a unit document row)."""
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    return (centers / np.maximum(norms, 1e-12)).astype(np.float32)


def cluster_diagnostic(
    gold: list[set[int]],
    seen: set[int],
    unseen: set[int],
    home: np.ndarray,
    top: np.ndarray,
    slices: dict[str, Any],
) -> dict[str, Any]:
    """Share of gold assignments whose label home cluster is in the doc's top-3."""
    result: dict[str, Any] = {}
    for name in SLICE_NAMES:
        indexes = np.flatnonzero(slices["masks"][name])
        slice_gold = [gold[index] for index in indexes]
        slice_top = top[indexes]
        for label_class, restricted in (
            ("all", slice_gold),
            ("seen", [labels & seen for labels in slice_gold]),
            ("unseen", [labels & unseen for labels in slice_gold]),
        ):
            hits, total = top_cluster_hits(restricted, home, slice_top)
            result[f"{name}/{label_class}"] = {
                "hits": hits,
                "assignments": total,
                "share": round(hits / total, 6) if total else 0.0,
            }
    return result


def pool_diagnostics(rows: list[dict[str, Any]], chance: float) -> dict[str, Any]:
    """Mean +- SD of each diagnostic share across seeds."""
    keys = [key for key, value in rows[0].items() if isinstance(value, dict) and "share" in value]
    pooled: dict[str, Any] = {"chance_share": round(chance, 6)}
    for key in sorted(keys):
        values = np.array([row[key]["share"] for row in rows], dtype=np.float64)
        totals = [row[key]["assignments"] for row in rows]
        pooled[key] = {
            "mean": round(float(values.mean()), 6),
            "sd": round(float(values.std(ddof=1)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
            "assignments_mean": round(float(np.mean(totals)), 1),
        }
    return pooled


def main() -> None:
    """Run Arm B on the five seeded label-sparse corpora."""
    started = time.perf_counter()
    data = load_dataset()
    labels = data["labels"]
    label_index = data["label_index"]
    train_labels = data["train_labels"]
    gold = data["gold"]
    n_labels = len(labels)

    slices = build_slices(data["train_records"], data["dev_records"])
    print(
        "slices: " + ", ".join(f"{name} {slices['documents'][name]}" for name in SLICE_NAMES),
        flush=True,
    )

    print("loading the cached MiniLM document vectors ...", flush=True)
    bank = EmbeddingBank()
    all_texts = data["train_texts"] + data["dev_texts"]
    document_vectors = bank.documents.encode(all_texts, "document")
    train_vectors = document_vectors[: len(data["train_records"])]
    dev_vectors = document_vectors[len(data["train_records"]) :]

    print("building the label-text spaces and embeddings ...", flush=True)
    texts_by_variant = {
        variant: label_texts(labels, data["vocabulary"], data["catalog"], variant)
        for variant in LABEL_TEXT_VARIANTS
    }
    label_vectors = {
        variant: bank.label_vectors(variant, texts_by_variant[variant])
        for variant in LABEL_TEXT_VARIANTS
    }
    transfer_matrix = label_tfidf(texts_by_variant["title_parents"])

    per_seed_all: dict[str, list[dict[str, float]]] = {}
    per_seed: dict[str, list[dict[str, Any]]] = {}
    comparison_store: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    states: dict[int, dict[str, Any]] = {}
    baselines: dict[int, dict[str, Any]] = {}

    def record(arm: str, seed: int, evaluation: dict[str, Any]) -> None:
        flat = flat_across_slices(evaluation["metrics"])
        per_seed_all.setdefault(arm, []).append(flat)
        per_seed.setdefault(arm, []).append({"seed": seed, "metrics": flat})

    print("phase A: the per-seed comparators ...", flush=True)
    for seed in SEEDS:
        seed_started = time.perf_counter()
        rows = sample_documents(seed)
        classes = label_classes(rows, train_labels, labels)
        support_map, support_counts = support_bucket_map(rows, train_labels, labels)
        seen = classes["seen"]
        unseen = classes["unseen"]
        seen_columns = np.array(sorted(seen), dtype=np.int64)
        unseen_columns = np.array(sorted(unseen), dtype=np.int64)
        samples.append(
            {
                "seed": seed,
                "seen_labels": classes["seen_labels"],
                "unseen_labels": classes["unseen_labels"],
                "unseen_assignment_share": round(
                    sum(len(document_gold & unseen) for document_gold in gold)
                    / sum(len(document_gold) for document_gold in gold),
                    6,
                ),
                "support_buckets": support_counts,
            }
        )

        index_label_matrix = np.zeros((len(rows), n_labels), dtype=bool)
        for position, row in enumerate(rows):
            for label in train_labels[int(row)]:
                index_label_matrix[position, label_index[label]] = True
        index_texts = [data["train_texts"][int(row)] for row in rows]
        lexical_scores, _ = lexical_knn(index_texts, index_label_matrix, data["dev_texts"])
        embedding_scores = embedding_knn(train_vectors[rows], index_label_matrix, dev_vectors)
        evaluations = {
            "lexical_knn": evaluate_scores(lexical_scores, gold, seen, unseen, support_map, slices),
            "embedding_knn": evaluate_scores(
                embedding_scores, gold, seen, unseen, support_map, slices
            ),
        }
        for variant in LABEL_TEXT_VARIANTS:
            scores = (dev_vectors @ label_vectors[variant].T).astype(np.float32)
            evaluations[f"label_text_{variant}"] = evaluate_scores(
                scores, gold, seen, unseen, support_map, slices
            )
            del scores

        # The sub-task-2 transfer hybrid, recomputed with the fixed settings.
        block = (transfer_matrix[unseen_columns] @ transfer_matrix[seen_columns].T).toarray()
        neighbours = seen_columns[top_k_positions(np.asarray(block, dtype=np.float32), TRANSFER_K)]
        transfer_scores = neighbour_transfer(lexical_scores, neighbours)
        evaluations[TRANSFER_ARM] = evaluate_scores(
            disjoint_hybrid(lexical_scores, transfer_scores, unseen_columns, n_labels),
            gold,
            seen,
            unseen,
            support_map,
            slices,
        )
        for arm, evaluation in evaluations.items():
            record(arm, seed, evaluation)
        baselines[seed] = evaluations
        states[seed] = {
            "seen": seen,
            "unseen": unseen,
            "seen_columns": seen_columns,
            "unseen_columns": unseen_columns,
            "support_map": support_map,
            "z_seen": row_zscore_columns(lexical_scores, seen_columns),
        }
        del lexical_scores, embedding_scores, transfer_scores, block
        print(f"  seed {seed} phase A {time.perf_counter() - seed_started:.1f}s", flush=True)

    print("fitting the train-text TF-IDF and the clusterings ...", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=CLUSTER_TFIDF["lowercase"],
        ngram_range=CLUSTER_TFIDF["ngram_range"],
        min_df=CLUSTER_TFIDF["min_df"],
        max_features=CLUSTER_TFIDF["max_features"],
        sublinear_tf=CLUSTER_TFIDF["sublinear_tf"],
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(data["train_texts"])
    dev_matrix = vectorizer.transform(data["dev_texts"])
    label_matrices = {
        variant: vectorizer.transform(texts_by_variant[variant]) for variant in LABEL_TEXT_VARIANTS
    }
    print(
        f"  vocabulary {len(vectorizer.vocabulary_)}, train nnz {train_matrix.nnz}, "
        f"dev nnz {dev_matrix.nnz}",
        flush=True,
    )

    diagnostics: dict[int, dict[str, Any]] = {}
    cluster_info: list[dict[str, Any]] = []
    for k in K_GRID:
        kmeans = MiniBatchKMeans(n_clusters=k, **KMEANS)
        cluster_started = time.perf_counter()
        train_clusters = kmeans.fit_predict(train_matrix)
        centroids = normalize_centroids(kmeans.cluster_centers_)
        dev_membership = positive_row_normalize(np.asarray(dev_matrix @ centroids.T))
        sizes = np.bincount(train_clusters, minlength=k)
        cluster_info.append(
            {
                "k": k,
                "inertia": round(float(kmeans.inertia_), 3),
                "seconds": round(time.perf_counter() - cluster_started, 1),
                "cluster_size_min": int(sizes.min()),
                "cluster_size_max": int(sizes.max()),
            }
        )
        print(
            f"  K={k}: {time.perf_counter() - cluster_started:.1f}s, "
            f"sizes {int(sizes.min())}-{int(sizes.max())}",
            flush=True,
        )
        variants = LABEL_TEXT_VARIANTS if k == PRIMARY_K else ("title_parents",)
        for variant in variants:
            affinity = positive_row_normalize(np.asarray(label_matrices[variant] @ centroids.T))
            cluster_scores = cluster_affinity_scores(dev_membership, affinity)
            config = next(item for item in CONFIGS if item["k"] == k and item["variant"] == variant)
            config_started = time.perf_counter()
            for seed in SEEDS:
                state = states[seed]
                record(
                    f"cluster_{k}_{variant}",
                    seed,
                    evaluate_scores(
                        cluster_scores,
                        gold,
                        state["seen"],
                        state["unseen"],
                        state["support_map"],
                        slices,
                    ),
                )
                hybrid = cluster_hybrid(
                    state["z_seen"],
                    cluster_scores,
                    state["seen_columns"],
                    state["unseen_columns"],
                    n_labels,
                )
                hybrid_evaluation = evaluate_scores(
                    hybrid,
                    gold,
                    state["seen"],
                    state["unseen"],
                    state["support_map"],
                    slices,
                )
                record(config["name"], seed, hybrid_evaluation)
                merge_comparisons(
                    comparison_store.setdefault(f"{config['name']}__vs__lexical_knn", {}),
                    compare_arms(
                        baselines[seed]["lexical_knn"],
                        hybrid_evaluation,
                        seed,
                        slices=COMPARISON_SLICES,
                        metrics=BOOTSTRAP_METRICS,
                    ),
                )
                merge_comparisons(
                    comparison_store.setdefault(f"{config['name']}__vs__{TRANSFER_ARM}", {}),
                    compare_arms(
                        baselines[seed][TRANSFER_ARM],
                        hybrid_evaluation,
                        seed,
                        slices=COMPARISON_SLICES,
                        metrics=BOOTSTRAP_METRICS,
                    ),
                )
                for variant_name in LABEL_TEXT_VARIANTS:
                    merge_comparisons(
                        comparison_store.setdefault(
                            f"{config['name']}__vs__label_text_{variant_name}", {}
                        ),
                        compare_arms(
                            baselines[seed][f"label_text_{variant_name}"],
                            hybrid_evaluation,
                            seed,
                            slices=COMPARISON_SLICES,
                            metrics=BOOTSTRAP_METRICS,
                        ),
                    )
                if config["primary"]:
                    diagnostics[seed] = cluster_diagnostic(
                        gold,
                        state["seen"],
                        state["unseen"],
                        home_clusters(affinity),
                        top_clusters(dev_membership, TOP_CLUSTERS),
                        slices,
                    )
                del hybrid
            del cluster_scores, affinity
            print(
                f"  {config['name']} {time.perf_counter() - config_started:.1f}s",
                flush=True,
            )
        del dev_membership, centroids

    pooled = {arm: pooled_metrics(rows) for arm, rows in per_seed_all.items()}
    comparisons = {name: finalize_comparisons(store) for name, store in comparison_store.items()}
    unseen_reference = {
        variant: pooled[f"label_text_{variant}"]["family_new/label_recall@50/unseen"]["mean"]
        for variant in LABEL_TEXT_VARIANTS
    }
    best_variant = max(unseen_reference, key=unseen_reference.get)
    readings = {
        config["name"]: transfer_reading(
            hybrid_pooled=pooled[config["name"]],
            title_pooled=pooled[f"label_text_{best_variant}"],
            lexical_pooled=pooled["lexical_knn"],
            comparison=comparisons[f"{config['name']}__vs__label_text_{best_variant}"],
        )
        for config in CONFIGS
    }
    primary = next(config for config in CONFIGS if config["primary"])
    primary_diagnostics = [
        row for seed, row in sorted(diagnostics.items(), key=lambda item: item[0])
    ]
    verdict = {
        "unseen_reference": {
            "variant": best_variant,
            "family_new_unseen_micro_recall@50": unseen_reference[best_variant],
            "by_variant": unseen_reference,
            "bar_unseen_micro_recall@50": round(unseen_reference[best_variant] + 0.05, 6),
        },
        "primary_config": primary["name"],
        "primary": readings[primary["name"]],
        "sensitivity": {config["name"]: readings[config["name"]] for config in CONFIGS},
        "outcome": "licensed" if readings[primary["name"]]["licensed"] else "null",
    }

    artifact: dict[str, Any] = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-21 sub-task 3 / Arm B (docs/topic_classification.md section 27.4)",
        "protocol": {
            "seeds": list(SEEDS),
            "labelled_documents_per_seed": 240,
            "label_count": n_labels,
            "label_space": "the sorted union of the train and dev label IDs; test never read",
            "seen_definition": "the label occurs in at least one of the 240 sampled documents",
            "samples": samples,
            "slices": {name: slices["documents"][name] for name in SLICE_NAMES},
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "document, paired within a seed; seed = the sampling seed",
            "pooling": "mean +- sample SD (ddof = 1) across the five seeds",
        },
        "config": {
            "cluster_tfidf": CLUSTER_TFIDF,
            "kmeans": {**KMEANS, "k_primary": PRIMARY_K, "k_grid": list(K_GRID)},
            "cluster_text": "the train split's 45,000 full_texts, no labels of any kind",
            "configs": list(CONFIGS),
            "soft_assignment": (
                "membership(d,c) = relu(cos(d, centroid_c)) / sum_c' relu(cos(d, centroid_c')); "
                "affinity(l,c) the same over the label text; score = membership @ affinity.T"
            ),
            "hybrid": (
                "seen labels take the lexical retrieval z-score; unseen labels take the "
                "cluster score z-scored within the unseen columns (Arm A's fixed rule)"
            ),
            "diagnostic": (
                "a gold label is inside the document's top-3 clusters when its home cluster "
                "(argmax affinity) is one of the document's three largest memberships"
            ),
        },
        "clusters": cluster_info,
        "arms": {
            arm: {"per_seed": per_seed[arm], "pooled": pooled[arm]} for arm in sorted(per_seed_all)
        },
        "comparisons": comparisons,
        "diagnostics": {
            "primary_config": primary["name"],
            "per_seed": primary_diagnostics,
            "pooled": pool_diagnostics(primary_diagnostics, TOP_CLUSTERS / PRIMARY_K)
            if primary_diagnostics
            else {},
        },
        "verdict": verdict,
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            **bank.stats(),
            "llm_calls": 0,
            "gpu": False,
        },
        "notes": {
            "leakage": (
                "the cluster fit reads the train split's text only — no labels of any of the "
                "45,000 documents; the 240 sampled documents are the entire labelled corpus. "
                "Dev text is not used to fit anything, and test is never read."
            ),
            "bar": (
                "licensed if unseen micro R@50 beats the best label-text variant by >= +0.05, "
                "the paired family_new unseen-macro interval excludes zero in >= 4/5 seeds, "
                "and family_new R@50 is no more than 0.01 below the lexical k-NN."
            ),
            "diagnostic": "the top-3 cluster share is descriptive, not a win condition.",
        },
    }
    write_artifact(OUT, artifact)
    print(json.dumps(verdict, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
