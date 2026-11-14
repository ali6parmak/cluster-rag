"""TC-19 = the off-menu EUROVOC parent channel on EURLEX-4K.

Adds four label-side hierarchy features to the TC-14 pack — parent support,
parent subtree rate, parent-title cosine and parent consistency — through the
committed ``catalogs/eurlex_labels.jsonl`` (built from the pinned official
EUROVOC SKOS).  Two arms, same folds and same dev rows: A = TC-14's 11-feature
pack (control), B = the same pack plus the four parent features.  A secondary
leg ranks the TC-11 zero-shot quota by ``max(own-title cosine, parent cosine)``.

Pre-registered in ``docs/topic_classification.md`` section 23 before this script
runs.  Artifact: ``results/eurlex-parent-channel/metrics.json``.  Splits:
train = fit (45,000, cross-fitted); dev = evaluation (6,000); test is never read.
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

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from candidate_pack import (  # noqa: E402
    MODEL_PARAMS,
    bootstrap_ranked,
    build_candidate_rows,
    slice_view,
)
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    _embedder_config,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)
from zero_shot_quota import unseen_recall  # noqa: E402

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    per_document_recall_at_k,
    ranked_labels,
)
from cluster_rag.topic_classification.quota import quota_insertion  # noqa: E402

OUT = ROOT / "results" / "eurlex-parent-channel"
RUN_ID = "eurlex-parent-channel"
CATALOG = ROOT / "catalogs" / "eurlex_labels.jsonl"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
N_FOLDS = 5
SEED = 42
BOOTSTRAP_RESAMPLES = 2_000
SHORTLIST_BAR = 0.01
ORDERING_BAR = 0.01
LOSS_TOLERANCE = 0.01
DEPTH = 50
TOP_VOTERS = 10
QUOTA = 5
QUOTA_UNSEEN_BAR = 0.10
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
PARENT_FEATURES = (
    "parent_support",
    "parent_rate",
    "parent_cosine",
    "parent_consistency",
)


def load_parent_structures(labels: list[str], label_index: dict[str, int]) -> dict[str, Any]:
    """The catalogue's parent side, re-indexed onto the 4,197-label pack space."""
    rows = {
        str(record["id"]): record
        for record in (json.loads(line) for line in CATALOG.open(encoding="utf-8"))
    }
    missing = [label for label in labels if label not in rows]
    if missing:
        raise ValueError(f"{len(missing)} labels missing from the catalogue: {missing[:5]}")
    parent_ids = sorted({parent for label in labels for parent in rows[label]["parents"]})
    parent_column = {parent: index for index, parent in enumerate(parent_ids)}
    label_parents = [
        np.asarray([parent_column[parent] for parent in rows[label]["parents"]], dtype=np.int32)
        for label in labels
    ]
    label_ancestor_columns = [
        np.asarray(
            [
                label_index[ancestor]
                for ancestor in rows[label]["ancestors"]
                if ancestor in label_index
            ],
            dtype=np.int32,
        )
        for label in labels
    ]
    parent_matrix = np.zeros((len(labels), len(parent_ids)), dtype=bool)
    for label_index_, columns in enumerate(label_parents):
        parent_matrix[label_index_, columns] = True
    parent_label_columns = np.asarray(
        [label_index.get(parent, -1) for parent in parent_ids], dtype=np.int32
    )
    parent_title: dict[str, str] = {}
    for label in labels:
        for parent, name in zip(rows[label]["parents"], rows[label]["parent_names"], strict=True):
            parent_title.setdefault(parent, name)
    parent_titles = [parent_title[parent] for parent in parent_ids]
    if not all(parent_titles):
        raise ValueError("a parent has no English title in the catalogue")
    return {
        "parent_ids": parent_ids,
        "parent_label_columns": parent_label_columns,
        "label_parents": label_parents,
        "label_ancestor_columns": label_ancestor_columns,
        "parent_matrix": parent_matrix,
        "parent_titles": parent_titles,
    }


def subtree_rate_vector(index_label_matrix: np.ndarray, structures: dict[str, Any]) -> np.ndarray:
    """For each parent: share of index documents carrying it or one of its children."""
    n_documents, n_labels = index_label_matrix.shape
    n_parents = len(structures["parent_ids"])
    counts = np.zeros((n_documents, n_parents), dtype=np.int32)
    # One pass per parent: the union of its children's documents (few children each).
    for column in range(n_parents):
        children = np.flatnonzero(structures["parent_matrix"][:, column])
        if len(children):
            counts[:, column] = index_label_matrix[:, children].any(axis=1).astype(np.int32)
        own = structures["parent_label_columns"][column]
        if own >= 0:
            counts[:, column] |= index_label_matrix[:, own].astype(np.int32)
    return counts.mean(axis=0)


def parent_feature_block(
    *,
    candidates: list[np.ndarray],
    rrf_scores: np.ndarray,
    document_vectors: np.ndarray,
    parent_vectors: np.ndarray,
    index_label_matrix: np.ndarray,
    structures: dict[str, Any],
    top_voters: int = TOP_VOTERS,
) -> np.ndarray:
    """The four parent features for one (index, query) split's candidate rows."""
    n_parents = len(structures["parent_ids"])
    label_parents = structures["label_parents"]
    label_ancestor_columns = structures["label_ancestor_columns"]
    parent_matrix = structures["parent_matrix"]
    parent_label_columns = structures["parent_label_columns"]
    rates = subtree_rate_vector(index_label_matrix, structures)
    rate_of_label = np.zeros(len(label_parents), dtype=np.float64)
    for label, columns in enumerate(label_parents):
        if len(columns):
            rate_of_label[label] = float(rates[columns].mean())
    cosines = (document_vectors @ parent_vectors.T).astype(np.float32)
    blocks: list[np.ndarray] = []
    for row in range(len(candidates)):
        rrf = rrf_scores[row]
        top = np.argpartition(-rrf, top_voters)[:top_voters]
        voters = {int(column) for column in top}
        voter_counts = (
            parent_matrix[list(voters)].sum(axis=0).astype(np.int32)
            if voters
            else np.zeros(n_parents, dtype=np.int32)
        )
        denominators = len(voters)
        candidate_labels = candidates[row]
        block = np.zeros((len(candidate_labels), 4), dtype=np.float32)
        for position, candidate in enumerate(candidate_labels):
            columns = label_parents[candidate]
            if not len(columns):
                continue
            block[position, 1] = rate_of_label[candidate]
            support = 0.0
            for parent in columns:
                label_column = parent_label_columns[parent]
                if label_column >= 0 and label_column in voters:
                    support = max(support, float(rrf[label_column]))
            block[position, 0] = support
            block[position, 2] = float(cosines[row, columns].max())
            excluded = [candidate] if candidate in voters else []
            for ancestor in label_ancestor_columns[candidate]:
                if int(ancestor) in voters:
                    excluded.append(int(ancestor))
            denominator = denominators - len(excluded)
            if denominator > 0:
                counts = voter_counts[columns].copy()
                for voter in excluded:
                    counts -= parent_matrix[voter, columns]
                block[position, 3] = float(counts.max()) / denominator
        blocks.append(block)
    return np.concatenate(blocks)


def fit_pack(features: np.ndarray, targets: np.ndarray) -> Any:
    """Fit the TC-14 HistGB pack."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    return model


def rank_with_pack(
    dev_data: dict[str, Any], probabilities: np.ndarray, n_documents: int, n_labels: int
) -> np.ndarray:
    """Sort each document's candidates by the pack score, ties by RRF rank (TC-14)."""
    ranked = np.empty((n_documents, n_labels), dtype=np.int32)
    offset = 0
    rrf_order = dev_data["rrf_order"]
    for row in range(n_documents):
        candidates = dev_data["candidates"][row]
        scores = probabilities[offset : offset + len(candidates)]
        offset += len(candidates)
        order = np.lexsort((rrf_order[row, candidates], -scores))
        ordered = candidates[order]
        used = np.zeros(n_labels, dtype=bool)
        used[candidates] = True
        rest = rrf_order[row][~used[rrf_order[row]]]
        ranked[row] = np.concatenate([ordered, rest])
    return ranked


def compare(
    baseline: np.ndarray,
    treatment: np.ndarray,
    gold: list[set[int]],
    slices: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Paired bootstrap of treatment minus baseline on every slice and metric."""
    return {
        name: {
            "r50": bootstrap_ranked(baseline, treatment, gold, mask, metric="r50"),
            "r_precision": bootstrap_ranked(baseline, treatment, gold, mask, metric="r_precision"),
            "ndcg": bootstrap_ranked(baseline, treatment, gold, mask, metric="ndcg"),
        }
        for name, mask in slices.items()
    }


def main() -> None:
    """Train the two packs, run the quota leg and write the artifact."""
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
    label_titles = [vocabulary[label] for label in labels]

    train_texts = [full_text(record) for record in train_records]
    dev_texts = [full_text(record) for record in dev_records]
    all_texts = train_texts + dev_texts
    all_words = np.array([len(text.split()) for text in all_texts], dtype=np.float32)

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
        f"labels {len(labels)}, slices "
        + ", ".join(f"{name} {int(mask.sum())}" for name, mask in slices.items()),
        flush=True,
    )

    structures = load_parent_structures(labels, label_index)
    n_parents = len(structures["parent_ids"])
    print(
        f"catalogue: {n_parents} parents, "
        f"{int(structures['parent_matrix'].any(axis=1).sum())} labels with a parent",
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
    parent_vectors = label_embedder.encode(structures["parent_titles"], "label")
    train_vectors = document_vectors[: len(train_records)]
    dev_vectors = document_vectors[len(train_records) :]
    print(
        f"  document cache hits {document_embedder.cache_hits}, "
        f"label hits {label_embedder.cache_hits}",
        flush=True,
    )

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, gold in enumerate(train_gold):
        train_label_matrix[row, list(gold)] = True

    groups = np.array([title_skeleton(record["title"]) for record in train_records])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(train_records)), groups=groups))

    print("building the out-of-fold candidate rows and parent features ...", flush=True)
    fold_rows: list[dict[str, Any]] = []
    fold_meta: list[dict[str, Any]] = []
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
        parent_block = parent_feature_block(
            candidates=data["candidates"],
            rrf_scores=data["rrf_scores"],
            document_vectors=train_vectors[held_rows],
            parent_vectors=parent_vectors,
            index_label_matrix=train_label_matrix[index_rows],
            structures=structures,
        )
        fold_rows.append(
            {
                "features": data["features"],
                "targets": data["targets"],
                "candidates": data["candidates"],
                "rrf_order": data["rrf_order"],
                "parent": parent_block,
            }
        )
        fold_meta.append(
            {
                "fold": fold_number,
                "index_documents": int(len(index_rows)),
                "held_out_documents": int(len(held_rows)),
                "rows": int(len(data["targets"])),
                "positives": int(data["targets"].sum()),
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
        del data
        print(
            f"  fold {fold_number}: {len(fold_rows[-1]['features'])} rows "
            f"({time.perf_counter() - fold_started:.0f}s)",
            flush=True,
        )

    print("building the dev candidate rows and parent features ...", flush=True)
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
    dev_parent = parent_feature_block(
        candidates=dev_data["candidates"],
        rrf_scores=dev_data["rrf_scores"],
        document_vectors=dev_vectors,
        parent_vectors=parent_vectors,
        index_label_matrix=train_label_matrix,
        structures=structures,
    )
    print(f"  dev rows {len(dev_data['targets'])}", flush=True)

    features = np.concatenate([row["features"] for row in fold_rows])
    parent_features = np.concatenate([row["parent"] for row in fold_rows])
    targets = np.concatenate([row["targets"] for row in fold_rows])
    dev_features = np.hstack([dev_data["features"], dev_parent])
    print(
        f"training rows {features.shape}, parents {parent_features.shape}; "
        f"dev {dev_features.shape}",
        flush=True,
    )

    print("fitting arm A (the 11-feature control) ...", flush=True)
    model_a = fit_pack(features, targets)
    probabilities_a = model_a.predict_proba(dev_data["features"])[:, 1]
    ranked_a = rank_with_pack(dev_data, probabilities_a, len(dev_records), len(labels))
    print("fitting arm B (the 15-feature pack) ...", flush=True)
    model_b = fit_pack(np.hstack([features, parent_features]), targets)
    probabilities_b = model_b.predict_proba(dev_features)[:, 1]
    ranked_b = rank_with_pack(dev_data, probabilities_b, len(dev_records), len(labels))

    methods = {
        "control_pack": {
            "arm": "TC-14 11-feature pack (control)",
            "slices": {
                name: slice_view(ranked_a, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
        "parent_pack": {
            "arm": "TC-14 pack + the four parent features (15)",
            "slices": {
                name: slice_view(ranked_b, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        },
    }
    comparisons = {"parent_pack_vs_control_pack": compare(ranked_a, ranked_b, dev_gold, slices)}
    primary = comparisons["parent_pack_vs_control_pack"]["family_new"]
    shortlist_conditions = {
        "r50_gain_at_least_0.01": primary["r50"]["delta"] >= SHORTLIST_BAR,
        "r50_ci_excludes_zero": primary["r50"]["bootstrap"]["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": primary["r_precision"]["delta"] >= -LOSS_TOLERANCE,
        "ndcg_loss_at_most_0.01": primary["ndcg"]["delta"] >= -LOSS_TOLERANCE,
    }
    ordering_conditions = {
        "ndcg_gain_at_least_0.01": primary["ndcg"]["delta"] >= ORDERING_BAR,
        "ndcg_ci_excludes_zero": primary["ndcg"]["bootstrap"]["ci_low"] > 0.0,
        "r50_loss_at_most_0.01": primary["r50"]["delta"] >= -LOSS_TOLERANCE,
    }

    print("running the zero-shot quota leg ...", flush=True)
    train_support = np.zeros(len(labels), dtype=np.int32)
    for gold in train_gold:
        train_support[list(gold)] += 1
    unseen_columns = np.flatnonzero(train_support == 0)
    unseen_set = {int(column) for column in unseen_columns}
    unseen_assignments = sum(
        1 for document in dev_gold for label in document if label in unseen_set
    )
    base_ranked = ranked_labels(dev_data["embedding_scores"])
    own_cosines = (dev_vectors @ label_vectors[unseen_columns].T).astype(np.float32)
    parent_cosines = (dev_vectors @ parent_vectors.T).astype(np.float32)
    parent_best = np.full((len(dev_records), len(unseen_columns)), -np.inf, dtype=np.float32)
    for local, column in enumerate(unseen_columns):
        parents = structures["label_parents"][int(column)]
        if len(parents):
            parent_best[:, local] = parent_cosines[:, parents].max(axis=1)
    parent_aware = np.maximum(own_cosines, parent_best)
    own_order = np.asarray(unseen_columns, dtype=np.int64)[
        np.argsort(-own_cosines, axis=1, kind="stable")
    ]
    parent_order = np.asarray(unseen_columns, dtype=np.int64)[
        np.argsort(-parent_aware, axis=1, kind="stable")
    ]

    def quota_view(order: np.ndarray) -> dict[str, Any]:
        base = float(per_document_recall_at_k(base_ranked, dev_gold, DEPTH).mean())
        inserted = quota_insertion(base_ranked, order, quota=QUOTA, depth=DEPTH)
        hits, assignments = unseen_recall(inserted, dev_gold, unseen_set)
        return {
            "unseen_hits": int(hits),
            "unseen_assignments": int(assignments),
            "unseen_r50": round(hits / assignments, 6) if assignments else 0.0,
            "all_dev_r50": slice_view(inserted, dev_gold, slices["all_dev"], bucket_of_label)[
                "primary"
            ]["recall@50"],
            "family_new_r50": slice_view(inserted, dev_gold, slices["family_new"], bucket_of_label)[
                "primary"
            ]["recall@50"],
            "all_dev_r50_loss": round(
                base
                - slice_view(inserted, dev_gold, slices["all_dev"], bucket_of_label)["primary"][
                    "recall@50"
                ],
                6,
            ),
        }

    quota_leg = {
        "base": "embedding k-NN (TC-11's primary base)",
        "base_all_dev_r50": slice_view(base_ranked, dev_gold, slices["all_dev"], bucket_of_label)[
            "primary"
        ]["recall@50"],
        "base_family_new_r50": slice_view(
            base_ranked, dev_gold, slices["family_new"], bucket_of_label
        )["primary"]["recall@50"],
        "own_title_cosine": quota_view(own_order),
        "parent_aware": quota_view(parent_order),
        "zero_shot_labels": int(len(unseen_columns)),
        "zero_shot_assignments": int(unseen_assignments),
    }
    quota_gain = (
        quota_leg["parent_aware"]["unseen_hits"] > 34
        and quota_leg["parent_aware"]["all_dev_r50_loss"] <= LOSS_TOLERANCE
    )
    quota_leg["bar"] = {
        "baseline_unseen_hits": 34,
        "gain_requires_more_than_34": True,
        "all_dev_r50_loss_at_most_0.01": quota_leg["parent_aware"]["all_dev_r50_loss"]
        <= LOSS_TOLERANCE,
        "met": bool(quota_gain),
        "outcome": "gain" if quota_gain else "null",
    }

    feature_stats = {
        name: {
            "train_mean": round(float(parent_features[:, index].mean()), 6),
            "train_nonzero_share": round(float((parent_features[:, index] != 0).mean()), 6),
            "dev_mean": round(float(dev_parent[:, index].mean()), 6),
            "dev_nonzero_share": round(float((dev_parent[:, index] != 0).mean()), 6),
        }
        for index, name in enumerate(PARENT_FEATURES)
    }
    shortlist_win = all(shortlist_conditions.values())
    ordering_win = all(ordering_conditions.values())
    verdict = {
        "primary_comparator": "control_pack (TC-14 11 features, base embeddings)",
        "shortlist_conditions": shortlist_conditions,
        "ordering_conditions": ordering_conditions,
        "family_new_r50": primary["r50"]["treatment"],
        "family_new_r50_comparator": primary["r50"]["baseline"],
        "family_new_r50_delta": primary["r50"]["delta"],
        "family_new_r50_95ci": [
            primary["r50"]["bootstrap"]["ci_low"],
            primary["r50"]["bootstrap"]["ci_high"],
        ],
        "family_new_ndcg@10_delta": primary["ndcg"]["delta"],
        "family_new_ndcg@10_95ci": [
            primary["ndcg"]["bootstrap"]["ci_low"],
            primary["ndcg"]["bootstrap"]["ci_high"],
        ],
        "family_new_r_precision_delta": primary["r_precision"]["delta"],
        "quota_leg": quota_leg["bar"],
        "outcome": ("win" if shortlist_win else ("ordering_win_only" if ordering_win else "null")),
    }

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-19 (TOPIC_CLASSIFICATION_PLAN.md section 6, off-menu)",
        "catalog": {
            "path": str(CATALOG.relative_to(ROOT)),
            "labels": len(labels),
            "parents": int(n_parents),
            "labels_with_parent": int(structures["parent_matrix"].any(axis=1).sum()),
            "parent_names": "the English skos:prefLabel of each parent",
            "source": "docs/topic_classification.md section 23 (pinned SKOS)",
        },
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "label_cache_version": LABEL_CACHE_VERSION,
            "parent_features": list(PARENT_FEATURES),
            "parent_aggregation": {
                "parent_support": "max over parents among the top-10 RRF voters",
                "parent_rate": "mean over parents of the index subtree rate",
                "parent_cosine": "max over parents of the document-parent-title cosine",
                "parent_consistency": "max over parents of the excluded-voter share",
            },
            "parentless": "0 for every parent feature",
            "top_voters": TOP_VOTERS,
            "model": "HistGradientBoostingClassifier",
            "model_params": MODEL_PARAMS,
            "folds": f"{N_FOLDS}-fold GroupKFold on train, grouped by title skeleton",
            "quota": {
                "base": "embedding k-NN",
                "insertion": "bottom N slots of the top-50",
                "n": QUOTA,
                "candidate_order": "max(own-title cosine, parent cosine)",
            },
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {"documents": {name: int(mask.sum()) for name, mask in slices.items()}},
        "training": {
            "folds": fold_meta,
            "rows": int(len(targets)),
            "positive_rate": round(float(targets.mean()), 6),
            "parent_feature_stats": feature_stats,
        },
        "methods": methods,
        "comparison": comparisons,
        "quota_leg": quota_leg,
        "verdict": verdict,
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "label_cache_hits": label_embedder.cache_hits,
            "llm_calls": 0,
        },
        "notes": {
            "leakage": (
                "each train document's candidates and features come from the other four folds; "
                "the parent features read only the index's labels, the catalogue and the train "
                "subtree rate, never the query document's own labels"
            ),
            "bar": (
                "B must beat A on family_new R@50 by at least 0.01 with the interval excluding 0 "
                "and lose R-precision/nDCG@10 by no more than 0.01; else null. The quota leg "
                "needs more than 34/96 unseen hits with an all-dev R@50 loss <= 0.01."
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
                "methods": {
                    name: methods[name]["slices"]["family_new"]["primary"]
                    for name in ("control_pack", "parent_pack")
                },
                "primary": primary,
                "quota": quota_leg,
                "verdict": verdict,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
