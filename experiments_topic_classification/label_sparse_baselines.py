"""TC-21 sub-task 1: the label-sparse harness and its four baselines.

240 labelled train documents (five seeded samples) over the full 4,197-label
space.  Four arms: the lexical TF-IDF k-NN, the MiniLM embedding k-NN, the
TC-14 11-feature pack trained on out-of-fold candidates from the 240 only, and
three label-text scorers (title / +parents / +definition) that score every
label zero-shot.  Each seed is evaluated on all_dev / family_new /
family_new_clean; the five seeds give per-seed numbers and the mean +- spread.

Pre-registered in ``docs/topic_classification.md`` section 27 before this script
runs.  Artifact: ``results/eurlex-label-sparse/metrics.json``.  The other 44,760
train documents' labels are never read; test is never read.
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

from audit_eurlex import title_skeleton  # noqa: E402
from candidate_pack import MODEL_PARAMS, build_candidate_rows  # noqa: E402
from label_sparse_common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    LABEL_TEXT_VARIANTS,
    PRIMARY_LABEL_VARIANT,
    SEEDS,
    SLICE_NAMES,
    EmbeddingBank,
    build_slices,
    compare_arms,
    embedding_knn,
    evaluate_ranked_output,
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

OUT = ROOT / "results" / "eurlex-label-sparse"
RUN_ID = "eurlex-label-sparse"
N_FOLDS = 5
BOOTSTRAP_METRICS = ("r50", "r_precision", "ndcg10", "unseen_r50")
COMPARISON_SLICES = ("all_dev", "family_new")


def small_pack_ranking(
    *,
    rows: np.ndarray,
    train_records: list[dict[str, Any]],
    train_texts: list[str],
    train_labels: list[list[str]],
    train_vectors: np.ndarray,
    label_vectors: np.ndarray,
    bucket_flags: dict[str, np.ndarray],
    dev_texts: list[str],
    dev_vectors: np.ndarray,
    dev_gold: list[set[int]],
    label_index: dict[str, int],
    n_labels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train the TC-14 pack on out-of-fold candidates from the 240 and rank dev."""
    sub_texts = [train_texts[int(row)] for row in rows]
    sub_titles = [train_records[int(row)]["title"] for row in rows]
    sub_vectors = train_vectors[rows]
    sub_labels = [train_labels[int(row)] for row in rows]
    sub_gold = [{label_index[label] for label in record_labels} for record_labels in sub_labels]
    sub_words = np.array([len(text.split()) for text in sub_texts], dtype=np.float32)
    dev_words = np.array([len(text.split()) for text in dev_texts], dtype=np.float32)

    sub_label_matrix = np.zeros((len(rows), n_labels), dtype=bool)
    for row, record_labels in enumerate(sub_labels):
        for label in record_labels:
            sub_label_matrix[row, label_index[label]] = True

    groups = np.array([title_skeleton(title) for title in sub_titles])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(rows)), groups=groups))
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, (index_rows, held_rows) in enumerate(folds):
        index_rows = np.asarray(index_rows)
        held_rows = np.asarray(held_rows)
        data = build_candidate_rows(
            index_texts=[sub_texts[row] for row in index_rows],
            index_vectors=sub_vectors[index_rows],
            index_label_matrix=sub_label_matrix[index_rows],
            query_texts=[sub_texts[row] for row in held_rows],
            query_vectors=sub_vectors[held_rows],
            query_gold=[sub_gold[row] for row in held_rows],
            query_words=sub_words[held_rows],
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
            }
        )
        del data

    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    training_rows = int(len(targets))
    del feature_blocks, target_blocks
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    del features, targets

    dev_data = build_candidate_rows(
        index_texts=sub_texts,
        index_vectors=sub_vectors,
        index_label_matrix=sub_label_matrix,
        query_texts=dev_texts,
        query_vectors=dev_vectors,
        query_gold=dev_gold,
        query_words=dev_words,
        label_vectors=label_vectors,
        bucket_flags=bucket_flags,
    )
    probabilities = model.predict_proba(dev_data["features"])[:, 1]
    del dev_data["features"]

    pack_ranked = np.empty((len(dev_texts), n_labels), dtype=np.int32)
    offset = 0
    rrf_order = dev_data["rrf_order"]
    for row in range(len(dev_texts)):
        candidates = dev_data["candidates"][row]
        scores = probabilities[offset : offset + len(candidates)]
        offset += len(candidates)
        order = np.lexsort((rrf_order[row, candidates], -scores))
        ordered = candidates[order]
        used = np.zeros(n_labels, dtype=bool)
        used[candidates] = True
        rest = rrf_order[row][~used[rrf_order[row]]]
        pack_ranked[row] = np.concatenate([ordered, rest])
    return pack_ranked, {
        "folds": fold_rows,
        "training_rows": training_rows,
        "dev_rows": int(offset),
    }


def main() -> None:
    """Run the four baselines on the five seeded label-sparse corpora."""
    started = time.perf_counter()
    data = load_dataset()
    train_records = data["train_records"]
    dev_records = data["dev_records"]
    labels = data["labels"]
    label_index = data["label_index"]
    train_labels = data["train_labels"]
    gold = data["gold"]
    n_labels = len(labels)

    print(
        f"train {len(train_records)}, dev {len(dev_records)}, labels {n_labels}",
        flush=True,
    )
    slices = build_slices(train_records, dev_records)
    print(
        "slices: " + ", ".join(f"{name} {slices['documents'][name]}" for name in SLICE_NAMES),
        flush=True,
    )

    print("loading the cached MiniLM document vectors ...", flush=True)
    bank = EmbeddingBank()
    all_texts = data["train_texts"] + data["dev_texts"]
    document_vectors = bank.documents.encode(all_texts, "document")
    train_vectors = document_vectors[: len(train_records)]
    dev_vectors = document_vectors[len(train_records) :]

    print("embedding the three label-text variants ...", flush=True)
    label_vectors: dict[str, np.ndarray] = {}
    label_text_by_variant: dict[str, list[str]] = {}
    for variant in LABEL_TEXT_VARIANTS:
        texts = label_texts(labels, data["vocabulary"], data["catalog"], variant)
        label_text_by_variant[variant] = texts
        label_vectors[variant] = bank.label_vectors(variant, texts)
        print(f"  {variant}: {len(texts)} texts", flush=True)
    definition_coverage = sum(1 for label in labels if data["catalog"][label]["definition"])
    parent_coverage = sum(1 for label in labels if data["catalog"][label]["parents"])

    arm_meta = {
        "lexical_knn": {
            "kind": "lexical_knn",
            "arm": "TF-IDF word 1-2 grams on the 240, cosine, k=100 clamped, power 1.0",
        },
        "embedding_knn": {
            "kind": "embedding_knn",
            "arm": "MiniLM whole document on the 240, cosine, k=100 clamped, power 1.0",
        },
        "small_pack": {
            "kind": "learned_candidate_pack",
            "arm": "TC-14 11 features, HistGB, 5 family-grouped folds on the 240",
        },
    }
    for variant in LABEL_TEXT_VARIANTS:
        arm_meta[f"label_text_{variant}"] = {
            "kind": "label_text_zero_shot",
            "arm": f"MiniLM document vs label text ({variant})",
        }

    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_meta}
    per_seed_all: dict[str, list[dict[str, float]]] = {name: [] for name in arm_meta}
    comparison_store: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    pack_diagnostics: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_started = time.perf_counter()
        rows = sample_documents(seed)
        classes = label_classes(rows, train_labels, labels)
        support_map, support_counts = support_bucket_map(rows, train_labels, labels)
        unseen = classes["unseen"]
        unseen_assignments = sum(len(document_gold & unseen) for document_gold in gold)
        documents_with_unseen = sum(1 for document_gold in gold if document_gold & unseen)
        samples.append(
            {
                "seed": seed,
                "seen_labels": classes["seen_labels"],
                "unseen_labels": classes["unseen_labels"],
                "unseen_assignments": unseen_assignments,
                "unseen_assignment_share": round(unseen_assignments / sum(len(g) for g in gold), 6),
                "documents_with_unseen": documents_with_unseen,
                "support_buckets": support_counts,
            }
        )
        print(
            f"seed {seed}: seen {classes['seen_labels']}, unseen {classes['unseen_labels']}, "
            f"unseen assignments {unseen_assignments}",
            flush=True,
        )

        index_label_matrix = np.zeros((len(rows), n_labels), dtype=bool)
        for position, row in enumerate(rows):
            for label in train_labels[int(row)]:
                index_label_matrix[position, label_index[label]] = True

        index_texts = [data["train_texts"][int(row)] for row in rows]
        lexical_scores, vocabulary = lexical_knn(index_texts, index_label_matrix, data["dev_texts"])
        embedding_scores = embedding_knn(train_vectors[rows], index_label_matrix, dev_vectors)
        knn_seconds = time.perf_counter() - seed_started
        print(
            f"  lexical vocabulary {vocabulary}; k-NNs {knn_seconds:.1f}s",
            flush=True,
        )

        bucket_names = np.array([support_map[index] for index in range(n_labels)])
        bucket_flags = {
            "head": bucket_names == "5+",
            "torso": bucket_names == "2-4",
            "tail": bucket_names == "1",
            "unseen": bucket_names == "unseen",
        }

        evaluations = {
            "lexical_knn": evaluate_scores(
                lexical_scores, gold, classes["seen"], unseen, support_map, slices
            ),
            "embedding_knn": evaluate_scores(
                embedding_scores, gold, classes["seen"], unseen, support_map, slices
            ),
        }
        del lexical_scores, embedding_scores

        for variant in LABEL_TEXT_VARIANTS:
            scores = (dev_vectors @ label_vectors[variant].T).astype(np.float32)
            evaluations[f"label_text_{variant}"] = evaluate_scores(
                scores, gold, classes["seen"], unseen, support_map, slices
            )
            del scores

        pack_ranked, pack_info = small_pack_ranking(
            rows=rows,
            train_records=train_records,
            train_texts=data["train_texts"],
            train_labels=train_labels,
            train_vectors=train_vectors,
            label_vectors=label_vectors["title"],
            bucket_flags=bucket_flags,
            dev_texts=data["dev_texts"],
            dev_vectors=dev_vectors,
            dev_gold=gold,
            label_index=label_index,
            n_labels=n_labels,
        )
        evaluations["small_pack"] = evaluate_ranked_output(
            pack_ranked, gold, classes["seen"], unseen, support_map, slices
        )
        del pack_ranked
        pack_diagnostics.append({"seed": seed, **pack_info})
        print(
            f"  arms evaluated ({time.perf_counter() - seed_started:.1f}s)",
            flush=True,
        )

        title_evaluation = evaluations["label_text_title"]
        for arm, evaluation in evaluations.items():
            flat = flat_across_slices(evaluation["metrics"])
            per_seed_all[arm].append(flat)
            per_seed[arm].append({"seed": seed, "metrics": flat})
            if arm != "lexical_knn":
                merge_comparisons(
                    comparison_store.setdefault(f"{arm}__vs__lexical_knn", {}),
                    compare_arms(
                        evaluations["lexical_knn"],
                        evaluation,
                        seed,
                        slices=COMPARISON_SLICES,
                        metrics=BOOTSTRAP_METRICS,
                    ),
                )
            if arm != "label_text_title":
                merge_comparisons(
                    comparison_store.setdefault(f"{arm}__vs__label_text_title", {}),
                    compare_arms(
                        title_evaluation,
                        evaluation,
                        seed,
                        slices=COMPARISON_SLICES,
                        metrics=BOOTSTRAP_METRICS,
                    ),
                )

    pooled = {arm: pooled_metrics(rows) for arm, rows in per_seed_all.items()}
    comparisons = {name: finalize_comparisons(store) for name, store in comparison_store.items()}

    unseen_reference = {
        variant: pooled[f"label_text_{variant}"]["family_new/label_recall@50/unseen"]["mean"]
        for variant in LABEL_TEXT_VARIANTS
    }
    best_variant = max(unseen_reference, key=unseen_reference.get)
    seen_only_unseen = {
        arm: pooled[arm]["family_new/label_recall@50/unseen"]["mean"]
        for arm in ("lexical_knn", "embedding_knn", "small_pack")
    }
    verdict = {
        "descriptive": True,
        "unseen_reference": {
            "variant": best_variant,
            "family_new_unseen_micro_recall@50": unseen_reference[best_variant],
            "by_variant": unseen_reference,
        },
        "b1_seen_only_arms_cannot_reach_unseen": {
            arm: {"family_new_unseen_micro_recall@50": value}
            for arm, value in seen_only_unseen.items()
        },
        "b1_passed": all(value == 0.0 for value in seen_only_unseen.values()),
    }

    artifact: dict[str, Any] = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-21 sub-task 1 (docs/topic_classification.md section 27.2)",
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
            "lexical": {
                "tfidf": {
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_features": 200_000,
                    "sublinear_tf": True,
                    "lowercase": True,
                },
                "k": 100,
                "power": 1.0,
                "index": "the 240 sampled documents only",
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
                "cache_version": "eurlex-4k-v1",
                "k": 100,
                "power": 1.0,
                "index_rows": "the 240 sampled documents only",
            },
            "pack": {
                "features": 11,
                "model": "HistGradientBoostingClassifier",
                "model_params": MODEL_PARAMS,
                "folds": "5 GroupKFold on the 240, grouped by title skeleton",
                "candidate_union": "embedding top-50 + lexical top-50 + frequency top-50",
                "buckets": "240 support: head >=5, torso 2-4, tail 1, unseen 0",
            },
            "label_text": {
                "variants": list(LABEL_TEXT_VARIANTS),
                "primary_variant": PRIMARY_LABEL_VARIANT,
                "cache_versions": {
                    "title": "eurlex-4k-label-v1",
                    "title_parents": "eurlex-4k-label-v2",
                    "title_parents_definition": "eurlex-4k-label-v2",
                },
                "parents_coverage": parent_coverage,
                "definition_coverage": definition_coverage,
                "score": "cosine(document vector, label-text vector)",
            },
        },
        "arms": {
            name: {
                **arm_meta[name],
                "per_seed": per_seed[name],
                "pooled": pooled[name],
            }
            for name in arm_meta
        },
        "comparisons": comparisons,
        "verdict": verdict,
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            **bank.stats(),
            "llm_calls": 0,
            "gpu": False,
        },
        "pack_diagnostics": pack_diagnostics,
        "notes": {
            "leakage": (
                "the 240 sampled documents are the entire labelled corpus; the other 44,760 "
                "train documents' labels are never read. The slices and the near-duplicate "
                "check read train text/titles only, and test is never read."
            ),
            "ties": (
                "every arm returns a full 4,197-label score vector; the seen-only arms score "
                "unseen labels 0 and ties break by label column order (the track convention)."
            ),
            "pack_expectation": (
                "expected weak at 240 documents; the TC-14 parameters are used unchanged, no "
                "tuning, and the result is reported as-is."
            ),
            "metrics": (
                "R@50 / R-precision / nDCG@10 are macro over documents; label_recall@50 is "
                "micro over gold assignments and is split by seen/unseen and by the 240-support "
                "bucket. The unseen/seen macro recall@50 averages only documents carrying a "
                "gold label of that class."
            ),
        },
    }
    write_artifact(OUT, artifact)
    print(json.dumps({"verdict": verdict}, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
