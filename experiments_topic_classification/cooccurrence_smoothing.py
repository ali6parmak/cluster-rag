"""TC-10 = menu item 4: label co-occurrence smoothing of the k-NN scores.

For each base score matrix (the embedding k-NN as the pre-registered primary,
the TF-IDF k-NN and the TC-9 RRF fusion as exploratory bases), the document's
top-10 labels vote for their train-only co-occurrence neighbours, and
``z(base) + alpha * z(boost)`` is scored with the same metrics.  Alpha is
selected by 5-fold family-grouped cross-validation on dev; the pooled CV
matrix is the headline.

Pre-registered in ``docs/topic_classification.md`` section 16 before this script
runs.  Artifact: ``results/eurlex-cooccurrence/metrics.json``.  Splits: train =
index (45,000), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import title_skeleton  # noqa: E402
from dictionary_fusion import per_document_recall, row_zscore  # noqa: E402
from embed_chunkmean import evaluate_slice, knn_scores  # noqa: E402
from lexical_rrf import TFIDF_CONFIG, lexical_scores  # noqa: E402
from multiview import read_bar, slice_bootstrap  # noqa: E402
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
from cluster_rag.topic_classification.cooccurrence import (  # noqa: E402
    cooccurrence_boost,
    cooccurrence_matrix,
)
from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion  # noqa: E402

OUT = ROOT / "results" / "eurlex-cooccurrence"
RUN_ID = "eurlex-cooccurrence"
EMBEDDING_CACHE_VERSION = "eurlex-4k-v1"
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
TOP_M = 10
RRF_TOP_K = 50
RRF_OFFSET = 60
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
BASE_NAMES = ("embedding_knn", "lexical_knn", "rrf_fusion")


def select_alpha(
    z_base: np.ndarray,
    z_boost: np.ndarray,
    gold: list[set[int]],
    indexes: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """The alpha that maximises R@50 on ``indexes`` (ties go to the smaller alpha)."""
    grid: dict[str, float] = {}
    for alpha in ALPHAS:
        smoothed = z_base[indexes] + alpha * z_boost[indexes]
        grid[str(alpha)] = round(
            float(per_document_recall(smoothed, [gold[index] for index in indexes], 50).mean()),
            6,
        )
    best = max(ALPHAS, key=lambda alpha: (grid[str(alpha)], -alpha))
    return best, grid


def grid_over_dev(
    z_base: np.ndarray,
    z_boost: np.ndarray,
    gold: list[set[int]],
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Full-dev R@50 per alpha on each slice (tuned, reported beside the CV)."""
    grid: dict[str, dict[str, float]] = {}
    for alpha in ALPHAS:
        smoothed = z_base + alpha * z_boost
        grid[str(alpha)] = {
            name: round(
                float(
                    per_document_recall(
                        smoothed[indexes], [gold[index] for index in indexes], 50
                    ).mean()
                ),
                6,
            )
            for name, mask in masks.items()
            for indexes in [np.flatnonzero(mask)]
        }
    return grid


def main() -> None:
    """Run the co-occurrence smoothing grid, CV and comparison."""
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
    groups = np.array([title_skeleton(record["title"]) for record in dev_records])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(dev_records)), groups=groups))
    print(
        "slices: "
        + ", ".join(f"{name} {int(mask.sum())}" for name, mask in slices.items())
        + f"; folds {len(folds)}",
        flush=True,
    )

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True

    print("building the train-only co-occurrence matrix ...", flush=True)
    matrix_started = time.perf_counter()
    cooccurrence = cooccurrence_matrix(train_label_matrix)
    n_pairs = int((cooccurrence > 0).sum())
    n_rows = int((cooccurrence.sum(axis=1) > 0).sum())
    print(
        f"  {n_pairs} directed pairs, {n_rows} labels with a row, "
        f"{time.perf_counter() - matrix_started:.1f}s",
        flush=True,
    )

    print("building the three base score matrices ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=EMBEDDING_CACHE_VERSION
    )
    embedding_scores = knn_scores(
        embedder.encode(train_texts, "document"),
        train_label_matrix,
        embedder.encode(dev_texts, "document"),
    )
    print(f"  embedding k-NN done (cache hits {embedder.cache_hits})", flush=True)

    vectorizer = TfidfVectorizer(
        lowercase=TFIDF_CONFIG["lowercase"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    dev_matrix = vectorizer.transform(dev_texts)
    lexical = lexical_scores(dev_matrix, train_matrix, train_label_matrix)
    print("  lexical k-NN done", flush=True)
    fused = reciprocal_rank_fusion([embedding_scores, lexical], top_k=RRF_TOP_K, offset=RRF_OFFSET)
    print("  RRF fusion done", flush=True)

    bases = {
        "embedding_knn": embedding_scores,
        "lexical_knn": lexical,
        "rrf_fusion": fused,
    }

    methods: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    grids: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}

    for name, scores in bases.items():
        print(f"=== {name}: smoothing ===", flush=True)
        base_method = {
            "kind": "knn",
            "arm": name,
            "slices": {
                slice_name: evaluate_slice(scores, gold, bucket_of_label, mask)
                for slice_name, mask in slices.items()
            },
        }
        z_base = row_zscore(scores)
        boost_started = time.perf_counter()
        boost = cooccurrence_boost(scores, cooccurrence, top_m=TOP_M)
        z_boost = row_zscore(boost)
        print(
            f"  boost {time.perf_counter() - boost_started:.1f}s, "
            f"nonzero rows {int((boost.sum(axis=1) > 0).sum())}",
            flush=True,
        )

        pooled = np.empty_like(z_base)
        selected: list[dict[str, Any]] = []
        for train_indexes, held_indexes in folds:
            best_alpha, fold_grid = select_alpha(z_base, z_boost, gold, train_indexes)
            pooled[held_indexes] = z_base[held_indexes] + best_alpha * z_boost[held_indexes]
            selected.append(
                {
                    "alpha": best_alpha,
                    "held_out_documents": int(len(held_indexes)),
                    "r50_on_training_folds": fold_grid[str(best_alpha)],
                }
            )
        smoothed_method = {
            "kind": "cooccurrence_smoothed",
            "arm": f"{name} + z(cooccurrence boost), top_m={TOP_M}",
            "slices": {
                slice_name: evaluate_slice(pooled, gold, bucket_of_label, mask)
                for slice_name, mask in slices.items()
            },
        }
        methods[name] = {"base": base_method, "smoothed_cv": smoothed_method}
        selections[name] = selected
        grids[name] = grid_over_dev(z_base, z_boost, gold, slices)
        comparisons[name] = {
            slice_name: slice_bootstrap(scores, pooled, gold, mask)
            for slice_name, mask in slices.items()
        }
        verdicts[name] = read_bar(comparisons[name], base_method, smoothed_method)
        smoothed_r50 = smoothed_method["slices"]["family_new"]["primary"]["recall@50"]
        base_r50 = base_method["slices"]["family_new"]["primary"]["recall@50"]
        print(
            f"  CV alphas {[entry['alpha'] for entry in selected]} -> "
            f"family_new R@50 {smoothed_r50:.4f} (base {base_r50:.4f})",
            flush=True,
        )
        del z_base, z_boost, boost, pooled

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-10 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 4)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "embedding_cache_version": EMBEDDING_CACHE_VERSION,
            "tfidf": TFIDF_CONFIG,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "cooccurrence": (
                "train-only C[a,b] = docs carrying both labels (a!=b, zero diagonal); "
                "P[b|a] = C / row-sum"
            ),
            "top_m": TOP_M,
            "smoothing": "per-row z-score of base and boost; z(base) + alpha * z(boost)",
            "alphas": list(ALPHAS),
            "alpha_selection": f"{N_FOLDS}-fold GroupKFold on dev, grouped by title skeleton",
            "rrf": f"reciprocal-rank fusion, top_k={RRF_TOP_K}, offset={RRF_OFFSET}",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "cooccurrence": {
            "directed_pairs": n_pairs,
            "labels_with_row": n_rows,
            "labels": len(labels),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "embedding_cache_hits": embedder.cache_hits,
            "embedding_cache_misses": embedder.cache_misses,
            "tfidf_vocabulary": int(len(vectorizer.vocabulary_)),
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparisons,
        "grid": grids,
        "selected_alphas": selections,
        "verdict": {
            "primary_base": "embedding_knn",
            "primary": verdicts["embedding_knn"],
            "exploratory": {
                "lexical_knn": verdicts["lexical_knn"],
                "rrf_fusion": verdicts["rrf_fusion"],
            },
            "outcome": verdicts["embedding_knn"]["outcome"],
        },
        "notes": {
            "bar": (
                "a win requires the primary (embedding) base's CV-smoothed scores to "
                "improve family_new R@50 by at least 0.01 with the paired 95% interval "
                "excluding 0, and R-precision / nDCG@10 losing at most 0.01"
            ),
            "bases": (
                "embedding_knn is the pre-registered base; lexical_knn and rrf_fusion "
                "are exploratory secondaries from TC-9 and do not decide the item"
            ),
            "leakage": (
                "the co-occurrence matrix is built from train labels only; alpha is "
                "selected on dev by family-grouped CV"
            ),
            "bucket_rule": "a collapse on any bucket is reported even if the pooled number rises",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_alphas": {
                    name: [entry["alpha"] for entry in entries]
                    for name, entries in selections.items()
                },
                "comparison": comparisons,
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
