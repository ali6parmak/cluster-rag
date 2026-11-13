"""TC-16 = menu item 10: the hard-negative discipline ablation.

Train the TC-14 pack twice with the same features, model and dev build:
(A) family-grouped folds (the family-disjoint training split, TC-14's
construction) and (B) random-document folds (near-duplicate siblings usually
in the retrieval index).  Compare the two on all_dev / family_new /
family_seen / family_new_clean, and report each arm's template exposure.

Pre-registered in ``docs/topic_classification.md`` section 22 before this script
runs.  Artifact: ``results/eurlex-hard-negative-discipline/metrics.json``.
Splits: train = fit (45,000), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from candidate_pack import (  # noqa: E402
    FEATURES,
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

from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import ranked_labels  # noqa: E402

OUT = ROOT / "results" / "eurlex-hard-negative-discipline"
RUN_ID = "eurlex-hard-negative-discipline"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_VERSION = "eurlex-4k-label-v1"
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
BAR = 0.01
LOSS_TOLERANCE = 0.01
SLICES = ("all_dev", "family_new", "family_seen", "family_new_clean")


def pack_ranking(
    probabilities: np.ndarray, dev_data: dict[str, Any], n_labels: int, n_documents: int
) -> np.ndarray:
    """The pack's ranking: candidates by model score (ties by RRF rank), then the rest."""
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


def train_arm(
    *,
    fold_assignment: list[tuple[np.ndarray, np.ndarray]],
    train_texts: list[str],
    train_vectors: np.ndarray,
    train_label_matrix: np.ndarray,
    train_gold: list[set[int]],
    train_words: np.ndarray,
    label_vectors: np.ndarray,
    bucket_flags: dict[str, np.ndarray],
) -> tuple[HistGradientBoostingClassifier, list[dict[str, Any]]]:
    """Fit the pack on one fold assignment and return the per-fold stats."""
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (index_rows, held_rows) in enumerate(fold_assignment):
        data = build_candidate_rows(
            index_texts=[train_texts[row] for row in index_rows],
            index_vectors=train_vectors[index_rows],
            index_label_matrix=train_label_matrix[index_rows],
            query_texts=[train_texts[row] for row in held_rows],
            query_vectors=train_vectors[held_rows],
            query_gold=[train_gold[row] for row in held_rows],
            query_words=train_words[held_rows],
            label_vectors=label_vectors,
            bucket_flags=bucket_flags,
        )
        feature_blocks.append(data["features"])
        target_blocks.append(data["targets"])
        fold_rows.append(
            {
                "fold": fold_number,
                "rows": int(len(data["targets"])),
                "positives": int(data["targets"].sum()),
            }
        )
        del data
    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    del feature_blocks, target_blocks
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    return model, fold_rows


def family_exposure(
    doc_families: np.ndarray, fold_assignment: list[tuple[np.ndarray, np.ndarray]], n_docs: int
) -> dict[str, float]:
    """Share of documents with a same-family sibling in their own retrieval index."""
    exposure = np.zeros(n_docs, dtype=bool)
    for index_rows, held_rows in fold_assignment:
        index_families = set(doc_families[index_rows].tolist())
        exposure[held_rows] = [int(doc_families[row]) in index_families for row in held_rows]
    return {
        "mean": round(float(exposure.mean()), 4),
        "documents": int(exposure.sum()),
    }


def main() -> None:
    """Fit both arms, compare them, and write the artifact."""
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

    print("computing the family split and the clean slice ...", flush=True)
    train_family_ids = {
        family: index
        for index, family in enumerate(
            sorted({title_skeleton(record["title"]) for record in train_records})
        )
    }
    doc_families = np.array(
        [train_family_ids[title_skeleton(record["title"])] for record in train_records]
    )
    train_families = {title_skeleton(record["title"]) for record in train_records}
    family_new = np.array(
        [title_skeleton(record["title"]) not in train_families for record in dev_records]
    )
    family_seen = ~family_new
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean = max_cosines < NEAR_DUP_THRESHOLD
    slices = {
        "all_dev": np.ones(len(dev_records), dtype=bool),
        "family_new": family_new,
        "family_seen": family_seen,
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

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True

    families = [title_skeleton(record["title"]) for record in train_records]
    grouped_folds = [
        (np.asarray(index), np.asarray(held))
        for index, held in GroupKFold(n_splits=N_FOLDS).split(
            np.arange(len(train_records)), groups=np.array(families)
        )
    ]
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(train_records))
    random_folds = []
    for chunk in np.array_split(shuffled, N_FOLDS):
        held = np.sort(chunk)
        random_folds.append((np.setdiff1d(np.arange(len(train_records)), held), held))
    fold_assignments = {"family_grouped": grouped_folds, "random": random_folds}

    print("building the shared dev candidates ...", flush=True)
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
    print(f"  dev rows {len(dev_data['targets'])}", flush=True)

    from lexical_rrf import TFIDF_CONFIG, lexical_scores  # noqa: PLC0415 (local import)
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

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
    lexical_ranked = ranked_labels(lexical)
    del lexical, train_tfidf, dev_tfidf

    arms: dict[str, Any] = {}
    exposures: dict[str, Any] = {}
    for arm_name, folds in fold_assignments.items():
        print(f"=== arm {arm_name}: {len(folds)} folds ===", flush=True)
        arm_started = time.perf_counter()
        model, fold_rows = train_arm(
            fold_assignment=folds,
            train_texts=train_texts,
            train_vectors=train_vectors,
            train_label_matrix=train_label_matrix,
            train_gold=train_gold,
            train_words=all_words[: len(train_records)],
            label_vectors=label_vectors,
            bucket_flags=bucket_flags,
        )
        probabilities = model.predict_proba(dev_data["features"])[:, 1]
        ranked = pack_ranking(probabilities, dev_data, len(labels), len(dev_records))
        exposures[arm_name] = family_exposure(doc_families, folds, len(train_records))
        arms[arm_name] = {
            "arm": arm_name,
            "slices": {
                name: slice_view(ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
            "fold_rows": fold_rows,
            "template_exposure": exposures[arm_name],
            "seconds": round(time.perf_counter() - arm_started, 1),
            "ranked": ranked,
        }
        arm_r50 = arms[arm_name]["slices"]["family_new"]["primary"]["recall@50"]
        print(
            f"  template exposure {exposures[arm_name]['mean']:.3f}, "
            f"family_new R@50 {arm_r50:.4f}, {arms[arm_name]['seconds']:.0f}s",
            flush=True,
        )

    print("comparing the arms ...", flush=True)
    comparison = {
        name: {
            "r50": bootstrap_ranked(
                arms["family_grouped"]["ranked"],
                arms["random"]["ranked"],
                dev_gold,
                mask,
                metric="r50",
            ),
            "r_precision": bootstrap_ranked(
                arms["family_grouped"]["ranked"],
                arms["random"]["ranked"],
                dev_gold,
                mask,
                metric="r_precision",
            ),
            "ndcg": bootstrap_ranked(
                arms["family_grouped"]["ranked"],
                arms["random"]["ranked"],
                dev_gold,
                mask,
                metric="ndcg",
            ),
            "lexical_r50": bootstrap_ranked(
                lexical_ranked, arms["family_grouped"]["ranked"], dev_gold, mask, metric="r50"
            ),
            "lexical_random_r50": bootstrap_ranked(
                lexical_ranked, arms["random"]["ranked"], dev_gold, mask, metric="r50"
            ),
        }
        for name, mask in slices.items()
    }

    family_new = comparison["family_new"]
    conditions = {
        "r50_gain_at_least_0.01": family_new["r50"]["delta"] >= BAR,
        "r50_ci_excludes_zero": family_new["r50"]["bootstrap"]["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": family_new["r_precision"]["delta"] >= -LOSS_TOLERANCE,
        "ndcg_loss_at_most_0.01": family_new["ndcg"]["delta"] >= -LOSS_TOLERANCE,
    }
    hurts = family_new["r50"]["delta"] <= -BAR and family_new["r50"]["bootstrap"]["ci_high"] < 0.0
    verdict = {
        "discipline_conditions": conditions,
        "family_new_r50_disciplined": family_new["r50"]["baseline"],
        "family_new_r50_random": family_new["r50"]["treatment"],
        "family_new_r50_delta": family_new["r50"]["delta"],
        "family_new_r50_95ci": [
            family_new["r50"]["bootstrap"]["ci_low"],
            family_new["r50"]["bootstrap"]["ci_high"],
        ],
        "family_new_ndcg_delta": family_new["ndcg"]["delta"],
        "template_exposure": exposures,
        "outcome": (
            "discipline_matters"
            if all(conditions.values())
            else ("discipline_hurts" if hurts else "null")
        ),
    }

    for arm in arms.values():
        del arm["ranked"]
    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-16 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 10)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "features": list(FEATURES),
            "model_params": MODEL_PARAMS,
            "arms": {
                "family_grouped": "5 folds grouped by title skeleton (the TC-14 construction)",
                "random": "5 random document folds (families cross folds)",
            },
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "family_seen_documents": int(family_seen.sum()),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "llm_calls": 0,
        },
        "arms": arms,
        "comparison": comparison,
        "verdict": verdict,
        "notes": {
            "question": (
                "does the family-disjoint training split (the menu item 10 "
                "construction) matter for the pack?"
            ),
            "manipulation_check": (
                "template exposure = share of train documents with a same-family sibling "
                "in their own retrieval index; expected 0 for the grouped arm"
            ),
            "bar": (
                "discipline_matters if the grouped arm beats the random arm on family_new "
                "R@50 by 0.01 with the interval excluding 0 and no R-precision/nDCG loss "
                "beyond 0.01"
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
                "exposure": exposures,
                "family_new": {
                    "disciplined_r50": family_new["r50"]["baseline"],
                    "random_r50": family_new["r50"]["treatment"],
                    "delta": family_new["r50"]["delta"],
                    "ci": [
                        family_new["r50"]["bootstrap"]["ci_low"],
                        family_new["r50"]["bootstrap"]["ci_high"],
                    ],
                },
                "verdict": verdict,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
