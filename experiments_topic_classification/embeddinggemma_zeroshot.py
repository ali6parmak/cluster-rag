"""TC-20 (part 1) = the embeddinggemma-300m zero-shot arm on EURLEX-4K.

Encodes the 51,000 train+dev documents and the 4,197 label titles with
``google/embeddinggemma-300m`` at 1,024 tokens (document prompt / query prompt),
runs the same k-NN as every other arm, and rebuilds the TC-14 11-feature pack
with the embeddinggemma channel.  The fine-tuned arm is deferred (its measured
cost was outside the approved budget); this run is the pre-registered
diagnostic.

Pre-registered in ``docs/topic_classification.md`` section 26 (and its pre-run
amendment) before this script runs.  Artifact:
``results/eurlex-embeddinggemma-zeroshot/metrics.json``.  Splits: train = fit
(out-of-fold), dev = evaluation (6,000); test is never read.
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
from parent_channel import rank_with_pack  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.config import EmbedderConfig, PrefixesConfig  # noqa: E402
from cluster_rag.embed.cache import CachedEmbedder  # noqa: E402
from cluster_rag.embed.hf import HFEmbedder  # noqa: E402
from cluster_rag.topic_classification.metrics import ranked_labels  # noqa: E402

OUT = ROOT / "results" / "eurlex-embeddinggemma-zeroshot"
RUN_ID = "eurlex-embeddinggemma-zeroshot"
MODEL_ID = "google/embeddinggemma-300m"
MODEL_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
DIMENSION = 768
MAX_SEQ_LENGTH = 1_024
CACHE_VERSION = "eurlex-4k-embeddinggemma-v1"
PREFIXES = PrefixesConfig(
    document="title: none | text: {content}",
    query="task: search result | query: {content}",
    label="task: search result | query: {content}",
)
N_FOLDS = 5
SEED = 42
BOOTSTRAP_RESAMPLES = 2_000
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
# The static comparators from the track's artifacts (section 26's baselines).
LEXICAL_REFERENCE = {"recall@50": 0.913765, "r_precision": 0.54148, "ndcg@10": 0.690237}
MINILM_REFERENCE = {"recall@50": 0.894357, "r_precision": 0.486891, "ndcg@10": 0.636414}
TC14_PACK_REFERENCE = {"recall@50": 0.927834, "r_precision": 0.58184, "ndcg@10": 0.734284}


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
    """Run the zero-shot arm and write the artifact."""
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

    print("loading embeddinggemma-300m ...", flush=True)
    config = EmbedderConfig(
        backend="hf",
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dimension=DIMENSION,
        context_tokens=MAX_SEQ_LENGTH,
        prefixes=PREFIXES,
        batch_size=32,
        device="cuda",
    )
    backend = HFEmbedder(config)
    backend._model.max_seq_length = MAX_SEQ_LENGTH  # noqa: SLF001 (the run's fixed length)
    token_counts = backend.token_counts(all_texts)
    truncation = {
        "documents": int(len(token_counts)),
        "max_seq_length": MAX_SEQ_LENGTH,
        "mean_tokens": round(float(token_counts.mean()), 1),
        "median_tokens": int(np.median(token_counts)),
        "p90_tokens": int(np.percentile(token_counts, 90)),
        "p95_tokens": int(np.percentile(token_counts, 95)),
        "truncated": int((token_counts > MAX_SEQ_LENGTH).sum()),
        "truncated_share": round(float((token_counts > MAX_SEQ_LENGTH).mean()), 6),
    }
    print(f"  truncation {truncation}", flush=True)

    print("encoding the documents and titles ...", flush=True)
    document_embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=CACHE_VERSION)
    documents = document_embedder.encode(all_texts, "document")
    label_embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version=CACHE_VERSION)
    label_vectors = label_embedder.encode(label_titles, "label")
    train_vectors = documents[: len(train_records)]
    dev_vectors = documents[len(train_records) :]
    print(
        f"  documents {documents.shape}, titles {label_vectors.shape}; "
        f"document cache hits {document_embedder.cache_hits}, "
        f"label hits {label_embedder.cache_hits}",
        flush=True,
    )

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, gold in enumerate(train_gold):
        train_label_matrix[row, list(gold)] = True
    groups = np.array([title_skeleton(record["title"]) for record in train_records])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(np.arange(len(train_records)), groups=groups))

    print("building the out-of-fold candidate rows ...", flush=True)
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
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
        del data
        print(f"  fold {fold_number} ({fold_rows[-1]['seconds']}s)", flush=True)

    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    del feature_blocks, target_blocks
    print(f"training rows {features.shape}, positive rate {targets.mean():.4f}", flush=True)

    print("fitting the zero-shot pack ...", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(features, targets)
    del features, targets

    print("building the dev candidates ...", flush=True)
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
    probabilities = model.predict_proba(dev_data["features"])[:, 1]
    pack_ranked = rank_with_pack(dev_data, probabilities, len(dev_records), len(labels))

    arms = {
        "embeddinggemma_knn": ranked_labels(dev_data["embedding_scores"]),
        "lexical_knn": ranked_labels(dev_data["lexical_scores"]),
        "zeroshot_pack": pack_ranked,
    }
    method_labels = {
        "embeddinggemma_knn": "embeddinggemma-300m k-NN (zero-shot)",
        "lexical_knn": "lexical TF-IDF k-NN (control)",
        "zeroshot_pack": "TC-14 11-feature pack with the embeddinggemma channel",
    }
    methods = {
        key: {
            "arm": method_labels[key],
            "slices": {
                name: slice_view(ranked, dev_gold, mask, bucket_of_label)
                for name, mask in slices.items()
            },
        }
        for key, ranked in arms.items()
    }

    lexical_exact = {
        metric: {
            "artifact": LEXICAL_REFERENCE[metric],
            "measured": methods["lexical_knn"]["slices"]["family_new"]["primary"][metric],
            "delta": round(
                methods["lexical_knn"]["slices"]["family_new"]["primary"][metric]
                - LEXICAL_REFERENCE[metric],
                9,
            ),
        }
        for metric in ("recall@50", "r_precision", "ndcg@10")
    }
    if any(abs(entry["delta"]) > 1e-6 for entry in lexical_exact.values()):
        raise ValueError(f"the lexical control does not reproduce its artifact: {lexical_exact}")

    comparisons = {
        "pack_minus_knn": compare(
            arms["embeddinggemma_knn"], arms["zeroshot_pack"], dev_gold, slices
        ),
        "knn_minus_lexical": compare(
            arms["lexical_knn"], arms["embeddinggemma_knn"], dev_gold, slices
        ),
        "pack_minus_lexical": compare(arms["lexical_knn"], arms["zeroshot_pack"], dev_gold, slices),
    }
    primary = methods["zeroshot_pack"]["slices"]["family_new"]["primary"]
    knn_primary = methods["embeddinggemma_knn"]["slices"]["family_new"]["primary"]
    readings = {
        "embeddinggemma_knn_family_new": knn_primary,
        "lexical_reference_family_new": LEXICAL_REFERENCE,
        "minilm_reference_family_new": MINILM_REFERENCE,
        "zeroshot_pack_family_new": primary,
        "tc14_pack_reference_family_new": TC14_PACK_REFERENCE,
        "knn_vs_lexical_r50_delta": round(
            knn_primary["recall@50"] - LEXICAL_REFERENCE["recall@50"], 6
        ),
        "knn_vs_minilm_r50_delta": round(
            knn_primary["recall@50"] - MINILM_REFERENCE["recall@50"], 6
        ),
        "pack_vs_tc14_r50_delta": round(primary["recall@50"] - TC14_PACK_REFERENCE["recall@50"], 6),
        "closes_the_gap_to_lexical": bool(
            knn_primary["recall@50"] > LEXICAL_REFERENCE["recall@50"]
        ),
    }

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-20 (part 1: the zero-shot arm; the fine-tuned arm is deferred)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "embedding_dimension": DIMENSION,
            "max_seq_length": MAX_SEQ_LENGTH,
            "prompts": {
                "document": PREFIXES.document,
                "query": PREFIXES.query,
                "label": PREFIXES.label,
            },
            "cache_version": CACHE_VERSION,
            "knn": "k=100, power=1.0, cosine votes",
            "features": "TC-14 11 features, embedding channel = embeddinggemma",
            "model": "HistGradientBoostingClassifier",
            "model_params": MODEL_PARAMS,
            "folds": f"{N_FOLDS}-fold GroupKFold on train, grouped by title skeleton",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
            "fine_tuned_arm": "deferred (measured 14.1 h, outside the approved budget)",
        },
        "slices": {"documents": {name: int(mask.sum()) for name, mask in slices.items()}},
        "truncation": truncation,
        "training": {"folds": fold_rows},
        "methods": methods,
        "comparison": comparisons,
        "readings": readings,
        "notes": {
            "leakage": (
                "train documents' pack features come from the other four folds only; dev is "
                "evaluation only; the title embeddings are label metadata"
            ),
            "scope": (
                "the zero-shot arm only; no fine-tuned embeddinggemma model was trained, so the "
                "section's primary (fine-tuned) bar is not evaluated"
            ),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "document_cache_hits": document_embedder.cache_hits,
            "label_cache_hits": label_embedder.cache_hits,
            "llm_calls": 0,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "family_new": {
                    key: methods[key]["slices"]["family_new"]["primary"] for key in arms
                },
                "readings": readings,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
