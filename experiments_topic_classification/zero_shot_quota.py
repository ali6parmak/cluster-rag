"""TC-11 = menu item 5: a zero-shot quota on the EURLEX-4K shortlist.

The seen-label k-NN cannot score a label with no train example. This experiment
reserves the bottom N slots of the top-50 for the N zero-shot labels with the
highest title cosine, selects N by 5-fold family-grouped CV on dev (maximising
the unseen-bucket R@50 subject to an all-dev R@50 loss <= 0.01), and reads the
result on all_dev / family_new / family_new_clean.

Pre-registered in ``docs/topic_classification.md`` section 17 before this script
runs.  Artifact: ``results/eurlex-zero-shot-quota/metrics.json``.  Splits:
train = index (45,000), dev = evaluation (6,000); test is never read.
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
    per_document_recall_at_k,
    ranked_labels,
)
from cluster_rag.topic_classification.quota import quota_insertion  # noqa: E402

OUT = ROOT / "results" / "eurlex-zero-shot-quota"
RUN_ID = "eurlex-zero-shot-quota"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
QUOTAS = (0, 1, 2, 3, 5, 8, 10, 15, 20)
DEPTH = 50
RRF_TOP_K = 50
RRF_OFFSET = 60
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
LOSS_LIMIT = 0.01
UNSEEN_BAR = 0.10
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")


def slice_view(
    ranked: np.ndarray, gold: list[set[int]], mask: np.ndarray, bucket_of_label: dict[int, str]
) -> dict[str, Any]:
    """Evaluate a ranked matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranked(ranked[indexes], [gold[index] for index in indexes], bucket_of_label)


def unseen_recall(
    ranked: np.ndarray, gold: list[set[int]], unseen_set: set[int]
) -> tuple[int, int]:
    """Unseen-bucket hits and assignments (the micro recall's numerator/denominator)."""
    hits = 0
    total = 0
    for row, labels in enumerate(gold):
        top = set(ranked[row, : min(DEPTH, ranked.shape[1])].tolist())
        for label in labels:
            if label in unseen_set:
                total += 1
                hits += int(label in top)
    return hits, total


def bootstrap_ranked(
    base_ranked: np.ndarray,
    treatment_ranked: np.ndarray,
    gold: list[set[int]],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Paired per-document R@50 difference between two ranked matrices."""
    indexes = np.flatnonzero(mask)
    slice_gold = [gold[index] for index in indexes]
    base_recall = per_document_recall_at_k(base_ranked[indexes], slice_gold, DEPTH)
    treatment_recall = per_document_recall_at_k(treatment_ranked[indexes], slice_gold, DEPTH)
    return {
        "baseline_recall@50": round(float(base_recall.mean()), 6),
        "treatment_recall@50": round(float(treatment_recall.mean()), 6),
        "delta_recall@50": round(float(treatment_recall.mean() - base_recall.mean()), 6),
        "bootstrap": paired_bootstrap_ci(
            treatment_recall - base_recall, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
    }


def select_quota(
    base_ranked: np.ndarray,
    candidate_order: np.ndarray,
    gold: list[set[int]],
    unseen_set: set[int],
    indexes: np.ndarray,
) -> tuple[int, dict[str, Any]]:
    """The quota that maximises unseen R@50 subject to the all-dev loss limit."""
    slice_gold = [gold[index] for index in indexes]
    base_r50 = float(per_document_recall_at_k(base_ranked[indexes], slice_gold, DEPTH).mean())
    chosen = 0
    best_unseen = -1.0
    diagnostics: dict[str, Any] = {}
    for quota in QUOTAS:
        order = quota_insertion(
            base_ranked[indexes], candidate_order[indexes], quota=quota, depth=DEPTH
        )
        recall = per_document_recall_at_k(order, slice_gold, DEPTH)
        loss = base_r50 - float(recall.mean())
        hits, assignments = unseen_recall(order, slice_gold, unseen_set)
        unseen = hits / assignments if assignments else 0.0
        diagnostics[str(quota)] = {
            "r50": round(float(recall.mean()), 6),
            "loss": round(loss, 6),
            "unseen_r50": round(unseen, 6),
            "unseen_hits": hits,
        }
        qualifies = loss <= LOSS_LIMIT
        better = unseen > best_unseen or (unseen == best_unseen and quota < chosen)
        if qualifies and better:
            chosen = quota
            best_unseen = unseen
    return chosen, diagnostics


def main() -> None:
    """Run the zero-shot quota grid, CV and comparison."""
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
    unseen_assignments = sum(1 for document in gold for label in document if label in unseen_set)
    unseen_documents = sum(1 for document in gold if document & unseen_set)
    print(
        f"labels {len(labels)}, zero-shot labels {len(unseen_columns)}, "
        f"assignments {unseen_assignments} over {unseen_documents} documents",
        flush=True,
    )

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

    print("building the three base rankings ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION)
    train_vectors = embedder.encode(train_texts, "document")
    dev_vectors = embedder.encode(dev_texts, "document")
    embedding_scores = knn_scores(train_vectors, train_label_matrix, dev_vectors)
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
    fused = reciprocal_rank_fusion([embedding_scores, lexical], top_k=RRF_TOP_K, offset=RRF_OFFSET)
    print("  bases ready", flush=True)

    print("embedding the label titles and scoring the candidates ...", flush=True)
    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    missing = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} labels have no title: {missing[:5]}")
    label_titles = [vocabulary[label] for label in labels]
    label_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=LABEL_CACHE_VERSION
    )
    label_vectors = label_embedder.encode(label_titles, "label")
    candidate_cosines = (dev_vectors @ label_vectors[unseen_columns].T).astype(np.float32)
    candidate_local_order = np.argsort(-candidate_cosines, axis=1, kind="stable")
    candidate_order = np.asarray(unseen_columns, dtype=np.int64)[candidate_local_order]
    print(f"  candidate order {candidate_order.shape}", flush=True)

    bases = {
        "embedding_knn": ranked_labels(embedding_scores),
        "lexical_knn": ranked_labels(lexical),
        "rrf_fusion": ranked_labels(fused),
    }

    methods: dict[str, Any] = {}
    grids: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}

    for name, base_ranked in bases.items():
        print(f"=== {name}: quota grid and CV ===", flush=True)
        base_method = {
            "kind": "knn",
            "arm": name,
            "slices": {
                slice_name: slice_view(base_ranked, gold, mask, bucket_of_label)
                for slice_name, mask in slices.items()
            },
        }
        grid: dict[str, Any] = {}
        for quota in QUOTAS:
            order = quota_insertion(base_ranked, candidate_order, quota=quota, depth=DEPTH)
            hits, assignments = unseen_recall(order, gold, unseen_set)
            grid[str(quota)] = {
                "all_dev_r50": slice_view(order, gold, slices["all_dev"], bucket_of_label)[
                    "primary"
                ]["recall@50"],
                "unseen_r50": round(hits / assignments, 6),
                "unseen_hits": hits,
                "family_new_r50": slice_view(order, gold, slices["family_new"], bucket_of_label)[
                    "primary"
                ]["recall@50"],
            }
        grids[name] = grid

        pooled = np.empty_like(base_ranked)
        selected: list[dict[str, Any]] = []
        for train_indexes, held_indexes in folds:
            quota, diagnostics = select_quota(
                base_ranked, candidate_order, gold, unseen_set, train_indexes
            )
            pooled[held_indexes] = quota_insertion(
                base_ranked[held_indexes], candidate_order[held_indexes], quota=quota, depth=DEPTH
            )
            selected.append(
                {
                    "quota": quota,
                    "held_out_documents": int(len(held_indexes)),
                    "fold_grid": diagnostics,
                }
            )
        quota_method = {
            "kind": "zero_shot_quota",
            "arm": f"{name} + bottom-of-top-50 quota",
            "slices": {
                slice_name: slice_view(pooled, gold, mask, bucket_of_label)
                for slice_name, mask in slices.items()
            },
        }
        methods[name] = {"base": base_method, "quota_cv": quota_method}
        selections[name] = selected
        comparisons[name] = {
            slice_name: bootstrap_ranked(base_ranked, pooled, gold, mask)
            for slice_name, mask in slices.items()
        }
        unseen = quota_method["slices"]["all_dev"]["diagnostics"]["label_recall@50"].get(
            "unseen", {}
        )
        unseen_r50 = float(unseen.get("recall", 0.0))
        unseen_hits = int(unseen.get("hits", 0))
        all_dev_loss = round(
            base_method["slices"]["all_dev"]["primary"]["recall@50"]
            - quota_method["slices"]["all_dev"]["primary"]["recall@50"],
            6,
        )
        family_new_loss = round(
            base_method["slices"]["family_new"]["primary"]["recall@50"]
            - quota_method["slices"]["family_new"]["primary"]["recall@50"],
            6,
        )
        conditions = {
            "unseen_r50_at_least_0.10": unseen_r50 >= UNSEEN_BAR,
            "all_dev_r50_loss_at_most_0.01": all_dev_loss <= LOSS_LIMIT,
            "family_new_r50_loss_at_most_0.01": family_new_loss <= LOSS_LIMIT,
        }
        verdicts[name] = {
            "bar_met": all(conditions.values()),
            "conditions": conditions,
            "unseen_r50": unseen_r50,
            "unseen_hits": unseen_hits,
            "unseen_assignments": int(unseen.get("labels", 0)),
            "all_dev_r50_loss": all_dev_loss,
            "family_new_r50_loss": family_new_loss,
            "outcome": "win" if all(conditions.values()) else "null",
        }
        print(
            f"  quotas {[entry['quota'] for entry in selected]} -> unseen "
            f"{unseen_hits}/{unseen.get('labels', 0)} ({unseen_r50:.4f}), "
            f"all_dev loss {all_dev_loss:.4f}, family_new loss {family_new_loss:.4f}",
            flush=True,
        )

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-11 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 5)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "label_cache_version": LABEL_CACHE_VERSION,
            "quotas": list(QUOTAS),
            "shortlist_depth": DEPTH,
            "insertion": "bottom N slots of the top-50; displaced entries move to 51..",
            "candidate_ranking": "cosine to the 89 zero-shot label titles",
            "selection": (
                f"{N_FOLDS}-fold GroupKFold on dev (title skeleton); maximise unseen R@50 "
                f"subject to an all-dev R@50 loss <= {LOSS_LIMIT}; ties to smaller N"
            ),
            "rrf": f"reciprocal-rank fusion, top_k={RRF_TOP_K}, offset={RRF_OFFSET}",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "zero_shot": {
            "labels": len(unseen_columns),
            "assignments": int(unseen_assignments),
            "documents": int(unseen_documents),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": embedder.cache_hits,
            "document_cache_misses": embedder.cache_misses,
            "label_cache_hits": label_embedder.cache_hits,
            "label_cache_misses": label_embedder.cache_misses,
            "llm_calls": 0,
        },
        "methods": methods,
        "grid": grids,
        "selected_quotas": selections,
        "comparison": comparisons,
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
                "a win requires unseen R@50 >= 0.10 with all-dev R@50 loss <= 0.01 and "
                "family_new R@50 loss <= 0.01 on the pooled CV matrix"
            ),
            "insertion": (
                "reserved labels sit at the bottom of the top-50, so R-precision and "
                "nDCG@10 are unchanged by construction; only R@50/coverage/unseen move"
            ),
            "bases": (
                "embedding_knn is the pre-registered base; lexical_knn and rrf_fusion "
                "are exploratory and do not decide the item"
            ),
            "leakage": "label titles only; the quota is selected on dev by family-grouped CV",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_quotas": {
                    name: [entry["quota"] for entry in entries]
                    for name, entries in selections.items()
                },
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
