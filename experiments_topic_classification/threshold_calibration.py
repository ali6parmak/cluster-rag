"""TC-12 = menu item 6: cardinality-calibrated thresholds on EURLEX-4K.

Turn the k-NN score matrix into a label set two ways — fixed top-5 (the
cardinality-matched control) and a global score threshold calibrated to the
train split's mean cardinality (5.074 labels/doc) — select the threshold by
5-fold family-grouped CV on dev, and compare the set metrics on all_dev /
family_new / family_new_clean.  The shortlist R@50 is reported unchanged.

Pre-registered in ``docs/topic_classification.md`` section 18 before this script
runs.  Artifact: ``results/eurlex-threshold-calibration/metrics.json``.  Splits:
train = index and cardinality target (45,000), dev = evaluation (6,000); test is
never read.
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
from cluster_rag.eval.multilabel import (  # noqa: E402
    apply_decision_rule,
    decision_f1_metrics,
)
from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion  # noqa: E402
from cluster_rag.topic_classification.metrics import evaluate_ranking  # noqa: E402
from cluster_rag.topic_classification.thresholding import target_cardinality_threshold  # noqa: E402

OUT = ROOT / "results" / "eurlex-threshold-calibration"
RUN_ID = "eurlex-threshold-calibration"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
CONTROL_K = 5
GRID_TARGETS = (2, 3, 4, 5, 6, 7, 8, 10)
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
CARDINALITY_TOLERANCE = 0.25
F1_BAR = 0.01
RECALL_TOLERANCE = 0.01
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
BUCKETS = ("head", "torso", "tail", "unseen")


def cardinality_stats(predicted: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    """Predicted vs gold set sizes."""
    predicted_counts = predicted.sum(axis=1).astype(np.float64)
    gold_counts = gold.sum(axis=1).astype(np.float64)
    return {
        "predicted_mean": round(float(predicted_counts.mean()), 3),
        "predicted_median": round(float(np.median(predicted_counts)), 3),
        "gold_mean": round(float(gold_counts.mean()), 3),
        "mae": round(float(np.abs(predicted_counts - gold_counts).mean()), 3),
    }


def evaluate_decision(
    predicted: np.ndarray,
    gold: np.ndarray,
    bucket_masks: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Set metrics on one document mask, plus the per-bucket views."""
    rows = np.flatnonzero(mask)
    view_predicted = predicted[rows]
    view_gold = gold[rows]
    metrics = decision_f1_metrics(view_predicted, view_gold)
    return {
        "micro_precision": round(metrics["micro_precision"], 6),
        "micro_recall": round(metrics["micro_recall"], 6),
        "micro_f1": round(metrics["micro_f1"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "sample_f1": round(metrics["sample_f1"], 6),
        "predicted_cells": metrics["n_predicted_cells"],
        "gold_cells": metrics["n_gold_cells"],
        "cardinality": cardinality_stats(view_predicted, view_gold),
        "buckets": {
            bucket: {
                "micro_precision": round(
                    decision_f1_metrics(view_predicted, view_gold, column_mask=column_mask)[
                        "micro_precision"
                    ],
                    6,
                ),
                "micro_recall": round(
                    decision_f1_metrics(view_predicted, view_gold, column_mask=column_mask)[
                        "micro_recall"
                    ],
                    6,
                ),
                "micro_f1": round(
                    decision_f1_metrics(view_predicted, view_gold, column_mask=column_mask)[
                        "micro_f1"
                    ],
                    6,
                ),
            }
            for bucket, column_mask in bucket_masks.items()
        },
    }


def paired_decision_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    gold: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Paired document bootstrap of micro-F1 and micro-recall differences."""
    rows = np.flatnonzero(mask)
    left = left[rows]
    right = right[rows]
    truth = gold[rows]

    def _counts(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (decision & truth).sum(axis=1),
            (decision & ~truth).sum(axis=1),
            (~decision & truth).sum(axis=1),
        )

    left_counts = _counts(left)
    right_counts = _counts(right)
    generator = np.random.default_rng(SEED)
    samples = generator.integers(0, len(rows), size=(BOOTSTRAP_RESAMPLES, len(rows)))

    def _bootstrap(
        left_values: tuple[np.ndarray, ...], right_values: tuple[np.ndarray, ...]
    ) -> dict[str, float]:
        differences_f1 = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
        for index, sample in enumerate(samples):
            ltp, lfp, lfn = (values[sample].sum() for values in left_values)
            rtp, rfp, rfn = (values[sample].sum() for values in right_values)
            differences_f1[index] = (2 * ltp) / (2 * ltp + lfp + lfn) - (2 * rtp) / (
                2 * rtp + rfp + rfn
            )
        ltp, lfp, lfn = (float(values.sum()) for values in left_values)
        rtp, rfp, rfn = (float(values.sum()) for values in right_values)
        return {
            "left": round((2 * ltp) / (2 * ltp + lfp + lfn), 6),
            "right": round((2 * rtp) / (2 * rtp + rfp + rfn), 6),
            "delta": round(
                (2 * ltp) / (2 * ltp + lfp + lfn) - (2 * rtp) / (2 * rtp + rfp + rfn), 6
            ),
            "ci_low": round(float(np.percentile(differences_f1, 2.5)), 6),
            "ci_high": round(float(np.percentile(differences_f1, 97.5)), 6),
        }

    def _pooled_recall(values: tuple[np.ndarray, ...]) -> float:
        tp, _, fn = values
        return float(tp.sum() / (tp.sum() + fn.sum()))

    recall_differences = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    left_tp, _, left_fn = left_counts
    right_tp, _, right_fn = right_counts
    for index, sample in enumerate(samples):
        left_recall = left_tp[sample].sum() / (left_tp[sample].sum() + left_fn[sample].sum())
        right_recall = right_tp[sample].sum() / (right_tp[sample].sum() + right_fn[sample].sum())
        recall_differences[index] = left_recall - right_recall
    return {
        "micro_f1": _bootstrap(left_counts, right_counts),
        "micro_recall": {
            "left": round(_pooled_recall(left_counts), 6),
            "right": round(_pooled_recall(right_counts), 6),
            "delta": round(_pooled_recall(left_counts) - _pooled_recall(right_counts), 6),
            "ci_low": round(float(np.percentile(recall_differences, 2.5)), 6),
            "ci_high": round(float(np.percentile(recall_differences, 97.5)), 6),
        },
    }


def main() -> None:
    """Run the threshold calibration, the control and the comparison."""
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
    gold_sets = [{label_index[label] for label in record_labels} for record_labels in dev_labels]
    bucket_of_label, bucket_counts = label_buckets(labels, train_labels)
    target_cardinality = float(np.mean([len(record_labels) for record_labels in train_labels]))
    print(f"train mean cardinality (target): {target_cardinality:.4f}", flush=True)

    gold_matrix = np.zeros((len(dev_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(dev_labels):
        for label in record_labels:
            gold_matrix[row, label_index[label]] = True
    bucket_masks = {
        bucket: np.array([bucket_of_label[index] == bucket for index in range(len(labels))])
        for bucket in BUCKETS
    }

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

    print("building the three base score matrices ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=DOCUMENT_CACHE_VERSION)
    embedding_scores = knn_scores(
        embedder.encode(train_texts, "document"),
        train_label_matrix,
        embedder.encode(dev_texts, "document"),
    )
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
    fused = reciprocal_rank_fusion([embedding_scores, lexical], top_k=50, offset=60)
    bases = {
        "embedding_knn": embedding_scores,
        "lexical_knn": lexical,
        "rrf_fusion": fused,
    }
    print("  bases ready", flush=True)

    methods: dict[str, Any] = {}
    grids: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}

    for name, scores in bases.items():
        print(f"=== {name}: decisions ===", flush=True)
        top_k_decision = apply_decision_rule(scores, rule="top_k", k=CONTROL_K)

        pooled_decision = np.zeros_like(top_k_decision)
        selected: list[dict[str, Any]] = []
        for train_indexes, held_indexes in folds:
            tau = target_cardinality_threshold(scores[train_indexes], target_cardinality)
            pooled_decision[held_indexes] = scores[held_indexes] >= tau
            selected.append(
                {
                    "threshold": round(float(tau), 6),
                    "held_out_documents": int(len(held_indexes)),
                    "train_realised_cardinality": round(
                        float((scores[train_indexes] >= tau).sum(axis=1).mean()), 3
                    ),
                }
            )

        methods[name] = {
            "control_top5": {
                "kind": "decision",
                "arm": f"top-{CONTROL_K}",
                "slices": {
                    slice_name: evaluate_decision(top_k_decision, gold_matrix, bucket_masks, mask)
                    for slice_name, mask in slices.items()
                },
            },
            "treatment_threshold": {
                "kind": "decision",
                "arm": f"score >= tau (train mean cardinality {target_cardinality:.3f})",
                "slices": {
                    slice_name: evaluate_decision(pooled_decision, gold_matrix, bucket_masks, mask)
                    for slice_name, mask in slices.items()
                },
            },
        }
        selections[name] = selected
        comparisons[name] = {
            slice_name: paired_decision_bootstrap(
                pooled_decision, top_k_decision, gold_matrix, mask
            )
            for slice_name, mask in slices.items()
        }

        grid: dict[str, Any] = {}
        for target in GRID_TARGETS:
            tau = target_cardinality_threshold(scores, float(target))
            decision = scores >= tau
            grid[str(target)] = {
                slice_name: evaluate_decision(decision, gold_matrix, bucket_masks, mask)
                for slice_name, mask in slices.items()
            }
        grids[name] = grid

        # The shortlist is unchanged by the decision; report it as the check.
        shortlist = {
            slice_name: evaluate_ranking(
                scores[np.flatnonzero(mask)],
                [gold_sets[index] for index in np.flatnonzero(mask)],
                bucket_of_label,
            )["primary"]["recall@50"]
            for slice_name, mask in slices.items()
        }
        methods[name]["shortlist_r50"] = shortlist

        family_new_metrics = methods[name]["treatment_threshold"]["slices"]["family_new"]
        all_dev_metrics = methods[name]["treatment_threshold"]["slices"]["all_dev"]
        comparison = comparisons[name]["family_new"]
        realised = all_dev_metrics["cardinality"]["predicted_mean"]
        conditions = {
            "micro_f1_gain_at_least_0.01": comparison["micro_f1"]["delta"] >= F1_BAR,
            "ci_excludes_zero": comparison["micro_f1"]["ci_low"] > 0.0,
            "micro_recall_loss_at_most_0.01": comparison["micro_recall"]["delta"]
            >= -RECALL_TOLERANCE,
            "cardinality_within_0.25": abs(realised - target_cardinality) <= CARDINALITY_TOLERANCE,
        }
        verdicts[name] = {
            "bar_met": all(conditions.values()),
            "conditions": conditions,
            "family_new_micro_f1": family_new_metrics["micro_f1"],
            "family_new_micro_f1_control": methods[name]["control_top5"]["slices"]["family_new"][
                "micro_f1"
            ],
            "micro_f1_delta": comparison["micro_f1"]["delta"],
            "micro_f1_95ci": [
                comparison["micro_f1"]["ci_low"],
                comparison["micro_f1"]["ci_high"],
            ],
            "micro_recall_delta": comparison["micro_recall"]["delta"],
            "realised_all_dev_cardinality": realised,
            "outcome": "win" if all(conditions.values()) else "null",
        }
        print(
            f"  top5 F1 {methods[name]['control_top5']['slices']['family_new']['micro_f1']:.4f} vs "
            f"threshold F1 {family_new_metrics['micro_f1']:.4f} "
            f"(delta {comparison['micro_f1']['delta']:+.4f}), "
            f"cardinality {realised:.2f}",
            flush=True,
        )

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-12 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 6)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "document_cache_version": DOCUMENT_CACHE_VERSION,
            "control": f"top-{CONTROL_K}",
            "treatment": "global score threshold at the train mean cardinality",
            "target_cardinality": round(target_cardinality, 6),
            "grid_targets": list(GRID_TARGETS),
            "selection": f"{N_FOLDS}-fold GroupKFold on dev (title skeleton)",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "embedding_cache_hits": embedder.cache_hits,
            "embedding_cache_misses": embedder.cache_misses,
            "llm_calls": 0,
        },
        "methods": methods,
        "grid": grids,
        "selected_thresholds": selections,
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
                "a win requires family_new micro-F1 +0.01 over top-5 with the paired 95% "
                "interval excluding 0, micro-recall losing at most 0.01, and the realised "
                "all_dev cardinality within 0.25 of the train mean"
            ),
            "calibration": (
                "the threshold targets the cardinality, not the metric; the realised "
                "cardinality is reported, not assumed"
            ),
            "control": "top-5 is the fixed-cardinality operating point at the train mean",
            "shortlist": "the decision does not change the ranking; R@50 is reported as a check",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "thresholds": {
                    name: [entry["threshold"] for entry in entries]
                    for name, entries in selections.items()
                },
                "comparison": {
                    name: {
                        slice_name: comparison["micro_f1"]
                        for slice_name, comparison in by_slice.items()
                    }
                    for name, by_slice in comparisons.items()
                },
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
