"""TC-14 = menu item 8: the learned candidate pack on EURLEX-4K.

A per-candidate HistGB model over 11 features (embedding / lexical / RRF scores,
label rate, label-name cosine, document length, bucket, retriever ranks), trained
on **out-of-fold train candidates** (5 folds grouped by title skeleton, so a
train document never retrieves from its own family) and applied to the dev
candidate union.  The pack's ranking is compared with the lexical k-NN (the
strongest single retriever) and the TC-9 RRF fusion.

Pre-registered in ``docs/topic_classification.md`` section 20 before this script
runs.  Artifact: ``results/eurlex-candidate-pack/metrics.json``.  Splits:
train = fit (45,000, cross-fitted), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from embed_chunkmean import knn_scores  # noqa: E402
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
from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranked,
    paired_bootstrap_ci,
    per_document_ndcg_at_k,
    per_document_r_precision,
    per_document_recall_at_k,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-candidate-pack"
RUN_ID = "eurlex-candidate-pack"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
CANDIDATE_K = 50
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SHORTLIST_BAR = 0.01
ORDERING_BAR = 0.01
LOSS_TOLERANCE = 0.01
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
MODEL_PARAMS = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 100,
    "early_stopping": False,
    "random_state": SEED,
}
FEATURES = (
    "embedding_score",
    "lexical_score",
    "rrf_score",
    "label_rate",
    "label_cosine",
    "doc_words",
    "bucket_head",
    "bucket_torso",
    "bucket_tail",
    "rank_embedding",
    "rank_lexical",
)


def slice_view(
    ranked: np.ndarray, gold: list[set[int]], mask: np.ndarray, bucket_of_label: dict[int, str]
) -> dict[str, Any]:
    """Evaluate a ranked matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranked(ranked[indexes], [gold[index] for index in indexes], bucket_of_label)


def build_candidate_rows(
    *,
    index_texts: list[str],
    index_vectors: np.ndarray,
    index_label_matrix: np.ndarray,
    query_texts: list[str],
    query_vectors: np.ndarray,
    query_gold: list[set[int]],
    query_words: np.ndarray,
    label_vectors: np.ndarray,
    bucket_flags: dict[str, np.ndarray],
    candidate_k: int = CANDIDATE_K,
) -> dict[str, Any]:
    """Candidates, features and targets for one (index, query) split.

    The index is always label-disjoint from the query documents' own labels in
    the training call (the fold construction guarantees it): the features here
    only ever read the index's labels and the query's text.
    """
    vectorizer = TfidfVectorizer(
        lowercase=TFIDF_CONFIG["lowercase"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )
    index_tfidf = vectorizer.fit_transform(index_texts)
    query_tfidf = vectorizer.transform(query_texts)
    lexical = lexical_scores(query_tfidf, index_tfidf, index_label_matrix)
    embedding = knn_scores(index_vectors, index_label_matrix, query_vectors)
    rrf = reciprocal_rank_fusion([embedding, lexical], top_k=CANDIDATE_K, offset=60)
    support = index_label_matrix.sum(axis=0).astype(np.float64) / len(index_texts)
    frequency_order = np.argsort(-support, kind="stable")
    embedding_order = np.argsort(-embedding, axis=1, kind="stable")
    lexical_order = np.argsort(-lexical, axis=1, kind="stable")
    rrf_order = np.argsort(-rrf, axis=1, kind="stable")

    n_query, n_labels = embedding.shape
    rank_embedding = np.empty((n_query, n_labels), dtype=np.int16)
    rank_lexical = np.empty((n_query, n_labels), dtype=np.int16)
    positions = np.arange(n_labels, dtype=np.int16)
    for row in range(n_query):
        rank_embedding[row, embedding_order[row]] = positions
        rank_lexical[row, lexical_order[row]] = positions

    label_cosine = np.empty((n_query, n_labels), dtype=np.float32)
    for start in range(0, n_query, 1024):
        stop = min(start + 1024, n_query)
        label_cosine[start:stop] = query_vectors[start:stop] @ label_vectors.T

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    candidate_blocks: list[np.ndarray] = []
    for row in range(n_query):
        candidates = np.unique(
            np.concatenate(
                [
                    embedding_order[row, :candidate_k],
                    lexical_order[row, :candidate_k],
                    frequency_order[:candidate_k],
                ]
            )
        )
        features = np.empty((len(candidates), len(FEATURES)), dtype=np.float32)
        features[:, 0] = embedding[row, candidates]
        features[:, 1] = lexical[row, candidates]
        features[:, 2] = rrf[row, candidates]
        features[:, 3] = support[candidates]
        features[:, 4] = label_cosine[row, candidates]
        features[:, 5] = float(query_words[row])
        features[:, 6] = bucket_flags["head"][candidates]
        features[:, 7] = bucket_flags["torso"][candidates]
        features[:, 8] = bucket_flags["tail"][candidates]
        features[:, 9] = np.minimum(rank_embedding[row, candidates], candidate_k)
        features[:, 10] = np.minimum(rank_lexical[row, candidates], candidate_k)
        gold = query_gold[row]
        targets = np.fromiter(
            (1 if int(label) in gold else 0 for label in candidates),
            dtype=np.int8,
            count=len(candidates),
        )
        feature_blocks.append(features)
        target_blocks.append(targets)
        candidate_blocks.append(candidates)

    return {
        "features": np.concatenate(feature_blocks),
        "targets": np.concatenate(target_blocks),
        "candidates": candidate_blocks,
        "rrf_order": rrf_order,
        "embedding_scores": embedding,
        "lexical_scores": lexical,
        "rrf_scores": rrf,
        "vocabulary": int(len(vectorizer.vocabulary_)),
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
    """Fit the out-of-fold pack and evaluate it on dev."""
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
    train_gold = [{label_index[label] for label in record_labels} for record_labels in train_labels]
    dev_gold = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)
    bucket_names = np.array([bucket_of_label[index] for index in range(len(labels))])
    bucket_flags = {name: (bucket_names == name) for name in ("head", "torso", "tail", "unseen")}

    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    missing = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} labels have no title: {missing[:5]}")
    label_titles = [vocabulary[label] for label in labels]

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]
    all_texts = train_texts + dev_texts
    all_words = np.array([len(text.split()) for text in all_texts], dtype=np.float32)

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

    print("loading the cached embeddings ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    document_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION
    )
    document_vectors = document_embedder.encode(all_texts, "document")
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_VERSION
    )
    label_vectors = label_embedder.encode(label_titles, "label")
    train_vectors = document_vectors[: len(train_records)]
    dev_vectors = document_vectors[len(train_records) :]
    print(
        f"  document cache hits {document_embedder.cache_hits}, "
        f"label hits {label_embedder.cache_hits}",
        flush=True,
    )

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True

    print("building the out-of-fold train candidates ...", flush=True)
    groups = np.array([title_skeleton(record["title"]) for record in train_records])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(train_records)), groups=groups))
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (index_rows, held_rows) in enumerate(folds):
        fold_started = time.perf_counter()
        index_rows = np.asarray(index_rows)
        held_rows = np.asarray(held_rows)
        data = build_candidate_rows(
            index_texts=[train_texts[row] for row in index_rows],
            index_vectors=train_vectors[index_rows],
            index_label_matrix=train_label_matrix[index_rows],
            query_texts=[train_texts[row] for row in held_rows],
            query_vectors=train_vectors[held_rows],
            query_gold=[train_gold[row] for row in held_rows],
            query_words=all_words[held_rows],
            label_vectors=label_vectors,
            bucket_flags=bucket_flags,
        )
        feature_blocks.append(data["features"])
        target_blocks.append(data["targets"])
        fold_rows.append(
            {
                "fold": fold_number,
                "index_documents": int(len(index_rows)),
                "held_out_documents": int(len(held_rows)),
                "rows": int(len(data["targets"])),
                "positives": int(data["targets"].sum()),
                "mean_candidates": round(float(data["features"].shape[0] / len(held_rows)), 2),
                "vocabulary": data["vocabulary"],
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
        print(
            f"  fold {fold_number}: {len(index_rows)} index / {len(held_rows)} held, "
            f"{len(data['targets'])} rows, {int(data['targets'].sum())} positives "
            f"({time.perf_counter() - fold_started:.0f}s)",
            flush=True,
        )
        del data

    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    del feature_blocks, target_blocks
    positives = int(targets.sum())
    training_rows = int(len(targets))
    positive_rate = round(positives / training_rows, 6)
    print(
        f"training rows {features.shape}, positive rate {positives / len(targets):.4f}",
        flush=True,
    )

    print("fitting the HistGB pack ...", flush=True)
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    del features, targets
    print(f"  fitted in {time.perf_counter() - started:.0f}s total", flush=True)

    print("building the dev candidates against the full train index ...", flush=True)
    dev_data = build_candidate_rows(
        index_texts=train_texts,
        index_vectors=train_vectors,
        index_label_matrix=train_label_matrix,
        query_texts=dev_texts,
        query_vectors=dev_vectors,
        query_gold=dev_gold,
        query_words=all_words[len(train_records) :],
        label_vectors=label_vectors,
        bucket_flags=bucket_flags,
    )
    print(
        f"  dev rows {len(dev_data['targets'])}, "
        f"mean candidates {len(dev_data['targets']) / len(dev_records):.2f}",
        flush=True,
    )
    pack_probabilities = model.predict_proba(dev_data["features"])[:, 1]
    del dev_data["features"]

    print("building the pack ranking ...", flush=True)
    pack_ranked = np.empty((len(dev_records), len(labels)), dtype=np.int32)
    offset = 0
    rrf_order = dev_data["rrf_order"]
    for row in range(len(dev_records)):
        candidates = dev_data["candidates"][row]
        scores = pack_probabilities[offset : offset + len(candidates)]
        offset += len(candidates)
        order = np.lexsort((rrf_order[row, candidates], -scores))
        ordered = candidates[order]
        used = np.zeros(len(labels), dtype=bool)
        used[candidates] = True
        rest = rrf_order[row][~used[rrf_order[row]]]
        pack_ranked[row] = np.concatenate([ordered, rest])

    print("building the comparator rankings ...", flush=True)
    embedding_ranked = ranked_labels(dev_data["embedding_scores"])
    lexical_ranked = ranked_labels(dev_data["lexical_scores"])
    rrf_ranked = ranked_labels(dev_data["rrf_scores"])

    methods = {
        "embedding_knn": {
            "kind": "knn",
            "arm": "embedding k-NN (reference)",
            "slices": {
                name: slice_view(embedding_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
        "lexical_knn": {
            "kind": "knn",
            "arm": "lexical TF-IDF k-NN (primary comparator)",
            "slices": {
                name: slice_view(lexical_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
        "rrf_fusion": {
            "kind": "rank_fusion",
            "arm": "RRF(embedding, lexical) (secondary comparator)",
            "slices": {
                name: slice_view(rrf_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
        "candidate_pack": {
            "kind": "learned_candidate_pack",
            "arm": "HistGB over the out-of-fold candidate features",
            "slices": {
                name: slice_view(pack_ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
    }

    comparisons: dict[str, Any] = {}
    for comparator, comparator_ranked in (
        ("lexical_knn", lexical_ranked),
        ("rrf_fusion", rrf_ranked),
    ):
        comparisons[comparator] = {
            name: {
                "r50": bootstrap_ranked(
                    comparator_ranked, pack_ranked, dev_gold, mask, metric="r50"
                ),
                "r_precision": bootstrap_ranked(
                    comparator_ranked, pack_ranked, dev_gold, mask, metric="r_precision"
                ),
                "ndcg": bootstrap_ranked(
                    comparator_ranked, pack_ranked, dev_gold, mask, metric="ndcg"
                ),
            }
            for name, mask in slices.items()
        }

    primary_comparison = comparisons["lexical_knn"]["family_new"]
    shortlist_conditions = {
        "r50_gain_at_least_0.01": primary_comparison["r50"]["delta"] >= SHORTLIST_BAR,
        "r50_ci_excludes_zero": primary_comparison["r50"]["bootstrap"]["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": primary_comparison["r_precision"]["delta"]
        >= -LOSS_TOLERANCE,
        "ndcg_loss_at_most_0.01": primary_comparison["ndcg"]["delta"] >= -LOSS_TOLERANCE,
    }
    ordering_conditions = {
        "ndcg_gain_at_least_0.01": primary_comparison["ndcg"]["delta"] >= ORDERING_BAR,
        "ndcg_ci_excludes_zero": primary_comparison["ndcg"]["bootstrap"]["ci_low"] > 0.0,
        "r50_loss_at_most_0.01": primary_comparison["r50"]["delta"] >= -LOSS_TOLERANCE,
    }
    shortlist_win = all(shortlist_conditions.values())
    ordering_win = all(ordering_conditions.values())
    verdict = {
        "primary_comparator": "lexical_knn",
        "shortlist_conditions": shortlist_conditions,
        "ordering_conditions": ordering_conditions,
        "family_new_r50": primary_comparison["r50"]["treatment"],
        "family_new_r50_comparator": primary_comparison["r50"]["baseline"],
        "family_new_r50_delta": primary_comparison["r50"]["delta"],
        "family_new_r50_95ci": [
            primary_comparison["r50"]["bootstrap"]["ci_low"],
            primary_comparison["r50"]["bootstrap"]["ci_high"],
        ],
        "family_new_ndcg@10": primary_comparison["ndcg"]["treatment"],
        "family_new_ndcg@10_comparator": primary_comparison["ndcg"]["baseline"],
        "family_new_ndcg@10_delta": primary_comparison["ndcg"]["delta"],
        "family_new_ndcg@10_95ci": [
            primary_comparison["ndcg"]["bootstrap"]["ci_low"],
            primary_comparison["ndcg"]["bootstrap"]["ci_high"],
        ],
        "family_new_r_precision_delta": primary_comparison["r_precision"]["delta"],
        "outcome": ("win" if shortlist_win else ("ordering_win_only" if ordering_win else "null")),
    }

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-14 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 8)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "label_cache_version": LABEL_CACHE_VERSION,
            "candidate_k": CANDIDATE_K,
            "candidate_union": "embedding top-50 + lexical top-50 + frequency top-50",
            "features": list(FEATURES),
            "model": "HistGradientBoostingClassifier",
            "model_params": MODEL_PARAMS,
            "folds": f"{N_FOLDS}-fold GroupKFold on train, grouped by title skeleton",
            "comparators": ["lexical_knn", "rrf_fusion"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "training": {
            "folds": fold_rows,
            "rows": training_rows,
            "positive_rate": positive_rate,
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "label_cache_hits": label_embedder.cache_hits,
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparisons,
        "verdict": verdict,
        "notes": {
            "leakage": (
                "each train document's candidates/features come from the other four folds "
                "only; the model never sees a document's own labels or its family"
            ),
            "dictionary_omitted": (
                "the z-scored dictionary arms are not features: a leak-free fold-specific "
                "dictionary for 45k train candidates is unaffordable; the frequency prior "
                "is the cheap count arm"
            ),
            "bar": (
                "shortlist win: family_new R@50 +0.01 over the lexical k-NN with the "
                "interval excluding 0 and no R-precision/nDCG loss beyond 0.01; ordering "
                "win: nDCG@10 +0.01 with no R@50 loss beyond 0.01"
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
                "comparison": {k: v["family_new"] for k, v in comparisons.items()},
                "verdict": verdict,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
