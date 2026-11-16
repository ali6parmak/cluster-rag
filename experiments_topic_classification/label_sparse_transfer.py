"""TC-21 sub-task 2: Arm A — the label-text k-NN transfer.

Every unseen label borrows the document evidence of its top-k seen labels by
label-text TF-IDF cosine: ``transfer(d, u) = mean_{j in top-k seen(u)}
retrieval(d, j)``.  The fixed hybrid keeps the retrieval z-score for the seen
labels and the transfer z-score for the unseen labels.  Primary settings:
title + immediate parents, k = 10, lexical k-NN evidence.  Fixed sensitivity:
k = 5/20, the title-only and title+definition variants, and the embedding k-NN
as secondary evidence.

Pre-registered in ``docs/topic_classification.md`` section 27.3 before this
script runs.  Artifact: ``results/eurlex-label-sparse-transfer/metrics.json``.
The other 44,760 train documents' labels are never read; test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from label_sparse_common import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    LABEL_TEXT_VARIANTS,
    PRIMARY_LABEL_VARIANT,
    READOUT_KS,
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

from cluster_rag.topic_classification.metrics import (  # noqa: E402
    label_recall_at_k,
    ndcg_at_k,
    r_precision,
    ranked_labels,
    recall_at_k,
)
from cluster_rag.topic_classification.transfer import (  # noqa: E402
    disjoint_hybrid,
    neighbour_transfer,
    top_k_positions,
    transfer_reading,
)

OUT = ROOT / "results" / "eurlex-label-sparse-transfer"
RUN_ID = "eurlex-label-sparse-transfer"
LABEL_TFIDF = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_features": 200_000,
    "sublinear_tf": True,
}
BOOTSTRAP_METRICS = ("r50", "r_precision", "ndcg10", "unseen_r50")
COMPARISON_SLICES = ("all_dev", "family_new")
CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "transfer_hybrid_title_parents_k10_lexical",
        "variant": "title_parents",
        "k": 10,
        "evidence": "lexical",
        "primary": True,
    },
    {
        "name": "transfer_hybrid_title_parents_k5_lexical",
        "variant": "title_parents",
        "k": 5,
        "evidence": "lexical",
        "primary": False,
    },
    {
        "name": "transfer_hybrid_title_parents_k20_lexical",
        "variant": "title_parents",
        "k": 20,
        "evidence": "lexical",
        "primary": False,
    },
    {
        "name": "transfer_hybrid_title_k10_lexical",
        "variant": "title",
        "k": 10,
        "evidence": "lexical",
        "primary": False,
    },
    {
        "name": "transfer_hybrid_title_parents_definition_k10_lexical",
        "variant": "title_parents_definition",
        "k": 10,
        "evidence": "lexical",
        "primary": False,
    },
    {
        "name": "transfer_hybrid_title_parents_k10_embedding",
        "variant": "title_parents",
        "k": 10,
        "evidence": "embedding",
        "primary": False,
    },
)


def label_text_matrix(texts: list[str]) -> Any:
    """Fit the label-text TF-IDF (the neighbour space) on the 4,197 labels."""
    vectorizer = TfidfVectorizer(
        lowercase=LABEL_TFIDF["lowercase"],
        ngram_range=LABEL_TFIDF["ngram_range"],
        min_df=LABEL_TFIDF["min_df"],
        max_features=LABEL_TFIDF["max_features"],
        sublinear_tf=LABEL_TFIDF["sublinear_tf"],
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, int(len(vectorizer.vocabulary_))


def similarity_block(
    matrix: Any, unseen_columns: np.ndarray, seen_columns: np.ndarray
) -> np.ndarray:
    """The unseen-by-seen label-text cosine block (the only part transfer needs)."""
    block = (matrix[unseen_columns] @ matrix[seen_columns].T).toarray()
    return np.asarray(block, dtype=np.float32)


def unseen_only_metrics(
    transfer: np.ndarray, gold: list[set[int]], unseen_columns: np.ndarray
) -> dict[str, Any]:
    """The transfer arm's own readout: rank the unseen labels alone."""
    position = {int(label): index for index, label in enumerate(unseen_columns)}
    full_gold = [{position[label] for label in labels if label in position} for labels in gold]
    documents = [row for row, labels in enumerate(full_gold) if labels]
    if not documents:
        return {"documents": 0, "assignments": 0}
    ranked = ranked_labels(transfer[documents])
    slice_gold = [full_gold[row] for row in documents]
    buckets = dict.fromkeys(range(len(unseen_columns)), "unseen")
    readout: dict[str, Any] = {
        "documents": len(documents),
        "assignments": int(sum(len(labels) for labels in slice_gold)),
        "micro_recall@50": label_recall_at_k(ranked, slice_gold, buckets, 50)["unseen"]["recall"],
        "macro_recall@50": round(recall_at_k(ranked, slice_gold, 50), 6),
        "r_precision": round(r_precision(ranked, slice_gold), 6),
        "ndcg@10": round(ndcg_at_k(ranked, slice_gold, 10), 6),
    }
    for k in READOUT_KS:
        if k == 50:
            continue
        readout[f"micro_recall@{k}"] = label_recall_at_k(ranked, slice_gold, buckets, k)["unseen"][
            "recall"
        ]
    return readout


def pool_readouts(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean +- sample SD across seeds for the unseen-only readout."""
    scalar_keys = [key for key in rows[0] if key not in ("seed", "documents", "assignments")]
    pooled: dict[str, dict[str, float]] = {
        "documents": {"mean": float(np.mean([row["documents"] for row in rows]))},
        "assignments": {"mean": float(np.mean([row["assignments"] for row in rows]))},
    }
    for key in scalar_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        pooled[key] = {
            "mean": round(float(values.mean()), 6),
            "sd": round(float(values.std(ddof=1)), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
        }
    return pooled


def main() -> None:
    """Run Arm A on the five seeded label-sparse corpora."""
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
    matrices = {
        variant: label_text_matrix(texts_by_variant[variant]) for variant in LABEL_TEXT_VARIANTS
    }
    for variant, (_, vocabulary) in matrices.items():
        print(f"  {variant}: label TF-IDF vocabulary {vocabulary}", flush=True)

    per_seed: dict[str, list[dict[str, Any]]] = {}
    per_seed_all: dict[str, list[dict[str, float]]] = {}
    readouts: dict[str, list[dict[str, Any]]] = {config["name"]: [] for config in CONFIGS}
    comparison_store: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    def record(arm: str, seed: int, evaluation: dict[str, Any]) -> None:
        flat = flat_across_slices(evaluation["metrics"])
        per_seed_all.setdefault(arm, []).append(flat)
        per_seed.setdefault(arm, []).append({"seed": seed, "metrics": flat})

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
        for arm, evaluation in evaluations.items():
            record(arm, seed, evaluation)

        blocks = {
            variant: similarity_block(matrices[variant][0], unseen_columns, seen_columns)
            for variant in LABEL_TEXT_VARIANTS
        }
        for config in CONFIGS:
            name = config["name"]
            evidence = lexical_scores if config["evidence"] == "lexical" else embedding_scores
            neighbours = seen_columns[top_k_positions(blocks[config["variant"]], int(config["k"]))]
            transfer = neighbour_transfer(evidence, neighbours)
            hybrid = disjoint_hybrid(evidence, transfer, unseen_columns, n_labels)
            evaluation = evaluate_scores(hybrid, gold, seen, unseen, support_map, slices)
            record(name, seed, evaluation)
            readout = unseen_only_metrics(transfer, gold, unseen_columns)
            readouts[name].append({"seed": seed, **readout})
            merge_comparisons(
                comparison_store.setdefault(f"{name}__vs__lexical_knn", {}),
                compare_arms(
                    evaluations["lexical_knn"],
                    evaluation,
                    seed,
                    slices=COMPARISON_SLICES,
                    metrics=BOOTSTRAP_METRICS,
                ),
            )
            for variant in LABEL_TEXT_VARIANTS:
                merge_comparisons(
                    comparison_store.setdefault(f"{name}__vs__label_text_{variant}", {}),
                    compare_arms(
                        evaluations[f"label_text_{variant}"],
                        evaluation,
                        seed,
                        slices=COMPARISON_SLICES,
                        metrics=BOOTSTRAP_METRICS,
                    ),
                )
            del transfer, hybrid
            flat = flat_across_slices(evaluation["metrics"])
            print(
                f"  seed {seed} {name}: family_new R@50 "
                f"{flat['family_new/recall@50']:.4f}, "
                f"unseen {flat['family_new/label_recall@50/unseen']:.4f}",
                flush=True,
            )
        del lexical_scores, embedding_scores, blocks
        print(f"seed {seed} done in {time.perf_counter() - seed_started:.1f}s", flush=True)

    arms = set(per_seed_all)
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
        "change_set": "TC-21 sub-task 2 / Arm A (docs/topic_classification.md section 27.3)",
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
            "label_text_tfidf": LABEL_TFIDF,
            "configs": list(CONFIGS),
            "primary_variant": PRIMARY_LABEL_VARIANT,
            "transfer": (
                "transfer(d,u) = mean retrieval score over u's top-k seen label-text neighbours"
            ),
            "hybrid": (
                "per document, z-score the retrieval ranking over the seen columns and the "
                "transfer ranking over the unseen columns separately; seen labels take the "
                "retrieval z-score, unseen labels the transfer z-score"
            ),
            "evidence": "lexical k-NN primary; embedding k-NN secondary",
        },
        "arms": {
            arm: {
                "per_seed": per_seed[arm],
                "pooled": pooled[arm],
                **({"unseen_only": pool_readouts(readouts[arm])} if arm in readouts else {}),
            }
            for arm in sorted(arms)
        },
        "comparisons": comparisons,
        "verdict": verdict,
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            **bank.stats(),
            "llm_calls": 0,
            "gpu": False,
        },
        "notes": {
            "leakage": (
                "the 240 sampled documents are the entire labelled corpus; the other 44,760 "
                "train documents' labels are never read. The label-text space is label "
                "metadata only (titles, parent names, official definitions)."
            ),
            "unseen_only": (
                "the transfer arm's own readout ranks the unseen labels alone; the hybrid is "
                "the full 4,197-label ranking the bar is read on."
            ),
            "bar": (
                "licensed if unseen micro R@50 beats the best label-text variant by >= +0.05, "
                "the paired family_new unseen-macro interval excludes zero in >= 4/5 seeds, "
                "and family_new R@50 is no more than 0.01 below the lexical k-NN."
            ),
            "ties": "a full ranking for every arm; ties break by label column order.",
        },
    }
    write_artifact(OUT, artifact)
    print(json.dumps(verdict, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
