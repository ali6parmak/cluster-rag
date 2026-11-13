"""TC-7 = menu item 1: chunk-mean document embeddings against the whole-document baseline.

The baseline is the current MiniLM-L6-v2 whole-document k-NN (k=100, power 1.0),
cached under ``eurlex-4k-v1``. The treatment embeds the same title + header +
recitals + main-body text as **non-overlapping 200-word chunks**, averages the
chunk vectors, L2-normalizes the mean and runs the identical k-NN, cached under
``eurlex-4k-chunk200-v1``.

Pre-registered in ``docs/topic_classification.md`` section 13 before this script
runs.  Artifact: ``results/eurlex-embed-chunkmean/metrics.json``.  Splits:
train = index (45,000), dev = evaluation (6,000); test is never read.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_topic_classification"))

from audit_eurlex import title_skeleton  # noqa: E402
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
from cluster_rag.retrieve.knn import knn_vote_scores  # noqa: E402
from cluster_rag.topic_classification.chunking import mean_pool, word_chunks  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranking,
    paired_bootstrap_ci,
    per_document_recall_at_k,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-embed-chunkmean"
RUN_ID = "eurlex-embed-chunkmean"
BASELINE_CACHE_VERSION = "eurlex-4k-v1"
TREATMENT_CACHE_VERSION = "eurlex-4k-chunk200-v1"
CHUNK_WORDS = 200
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")


def evaluate_slice(
    scores: np.ndarray,
    gold: list[set[int]],
    bucket_of_label: dict[int, str],
    mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one score matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranking(scores[indexes], [gold[index] for index in indexes], bucket_of_label)


def knn_scores(
    train_vectors: np.ndarray,
    train_label_matrix: np.ndarray,
    dev_vectors: np.ndarray,
) -> np.ndarray:
    """The shared k-NN (k=100, power 1.0), batched over the dev rows."""
    scores = np.zeros((len(dev_vectors), train_label_matrix.shape[1]), dtype=np.float32)
    for start in range(0, len(dev_vectors), 256):
        stop = min(start + 256, len(dev_vectors))
        scores[start:stop] = knn_vote_scores(
            train_vectors,
            train_label_matrix,
            dev_vectors[start:stop],
            k=KNN_NEIGHBORS,
            power=VOTE_POWER,
        )
    return scores


def document_vectors_from_chunks(
    chunk_lists: list[list[str]], chunk_vectors: np.ndarray
) -> np.ndarray:
    """Mean-pool each document's chunk vectors into one unit-norm vector."""
    offsets = np.cumsum([0] + [len(chunks) for chunks in chunk_lists])
    vectors = np.empty((len(chunk_lists), chunk_vectors.shape[1]), dtype=np.float32)
    for row, chunks in enumerate(chunk_lists):
        if not chunks:
            raise ValueError(f"document {row} has no chunks (the text is blank)")
        vectors[row] = mean_pool(chunk_vectors[offsets[row] : offsets[row + 1]])
    return vectors


def token_diagnostics(texts: list[str], *, chunk_words: int, backend: HFEmbedder) -> dict[str, Any]:
    """How much of each view survives the model's 256-token context (read-only).

    A diagnostic, not a metric: it counts the tokenizer's subword tokens on the
    same text view the k-NN embeds, so the truncation share is measured rather
    than inferred.  Both views are reported because the treatment's premise was
    that 200-word chunks fit the context.

    Args:
        texts: The documents to count (the dev split keeps this cheap).
        chunk_words: The fixed chunk size used by the treatment.
        backend: The HF embedder (it owns the tokenizer).

    Returns:
        JSON-friendly counts for the whole-document and chunk views.
    """
    chunks = [chunk for text in texts for chunk in word_chunks(text, chunk_words=chunk_words)]
    doc_tokens = backend.token_counts(texts)
    chunk_tokens = backend.token_counts(chunks)
    context = backend.info.context_tokens

    def _summary(tokens: np.ndarray) -> dict[str, float | int]:
        return {
            "texts": int(tokens.size),
            "mean": round(float(tokens.mean()), 2),
            "median": round(float(np.median(tokens)), 2),
            "p90": round(float(np.percentile(tokens, 90)), 2),
            "max": int(tokens.max()),
            "share_over_context": round(float((tokens > context).mean()), 4),
        }

    return {
        "documents": _summary(doc_tokens),
        "chunks": _summary(chunk_tokens),
        "context_tokens": int(context),
        "split": "dev",
        "note": (
            "tokenizer token counts on the same text view; a read-only truncation "
            "diagnostic, not an evaluated metric"
        ),
    }


def main() -> None:
    """Run the chunk-mean experiment and write the artifact."""
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
    all_texts = train_texts + dev_texts

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

    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True

    backend = HFEmbedder(_embedder_config())

    print("baseline: loading the cached whole-document MiniLM vectors ...", flush=True)
    baseline_started = time.perf_counter()
    baseline_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=BASELINE_CACHE_VERSION
    )
    train_base = baseline_embedder.encode(train_texts, "document")
    dev_base = baseline_embedder.encode(dev_texts, "document")
    print(
        f"  {time.perf_counter() - baseline_started:.1f}s "
        f"(hits {baseline_embedder.cache_hits}, misses {baseline_embedder.cache_misses})",
        flush=True,
    )

    print("treatment: splitting into 200-word chunks ...", flush=True)
    chunk_lists = [word_chunks(text, chunk_words=CHUNK_WORDS) for text in all_texts]
    chunk_counts = np.array([len(chunks) for chunks in chunk_lists])
    flat_chunks = [chunk for chunks in chunk_lists for chunk in chunks]
    print(
        f"  {len(flat_chunks)} chunks for {len(all_texts)} documents "
        f"(mean {chunk_counts.mean():.2f}, max {chunk_counts.max()})",
        flush=True,
    )

    print("treatment: embedding the chunks ...", flush=True)
    treatment_started = time.perf_counter()
    treatment_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=TREATMENT_CACHE_VERSION
    )
    chunk_vectors = treatment_embedder.encode(flat_chunks, "document")
    print(
        f"  embedded in {time.perf_counter() - treatment_started:.1f}s "
        f"(hits {treatment_embedder.cache_hits}, misses {treatment_embedder.cache_misses})",
        flush=True,
    )

    print("treatment: mean-pooling and normalizing ...", flush=True)
    pooled = document_vectors_from_chunks(chunk_lists, chunk_vectors)
    train_treat = pooled[: len(train_records)]
    dev_treat = pooled[len(train_records) :]
    del chunk_vectors, pooled

    print("computing the two k-NN score matrices ...", flush=True)
    knn_started = time.perf_counter()
    scores_base = knn_scores(train_base, train_label_matrix, dev_base)
    print(f"  baseline k-NN {time.perf_counter() - knn_started:.1f}s", flush=True)
    knn_started = time.perf_counter()
    scores_treat = knn_scores(train_treat, train_label_matrix, dev_treat)
    print(f"  treatment k-NN {time.perf_counter() - knn_started:.1f}s", flush=True)

    print("evaluating both methods on every slice ...", flush=True)
    methods: dict[str, Any] = {
        "baseline_whole_document": {
            "kind": "embedding_knn",
            "arm": f"MiniLM whole document, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
            "cache_version": BASELINE_CACHE_VERSION,
            "cache_hits": baseline_embedder.cache_hits,
            "cache_misses": baseline_embedder.cache_misses,
        },
        "treatment_chunk_mean": {
            "kind": "embedding_knn",
            "arm": (f"MiniLM {CHUNK_WORDS}-word chunk mean, k={KNN_NEIGHBORS}, power={VOTE_POWER}"),
            "cache_version": TREATMENT_CACHE_VERSION,
            "cache_hits": treatment_embedder.cache_hits,
            "cache_misses": treatment_embedder.cache_misses,
            "chunks": int(len(flat_chunks)),
            "chunks_per_document_mean": round(float(chunk_counts.mean()), 3),
            "chunks_per_document_max": int(chunk_counts.max()),
        },
    }
    for method_name, scores in (
        ("baseline_whole_document", scores_base),
        ("treatment_chunk_mean", scores_treat),
    ):
        methods[method_name]["slices"] = {
            name: evaluate_slice(scores, gold, bucket_of_label, mask)
            for name, mask in slices.items()
        }

    print("paired bootstrap over documents (2,000 resamples) ...", flush=True)
    comparison: dict[str, Any] = {}
    for name, mask in slices.items():
        indexes = np.flatnonzero(mask)
        base_ranked = ranked_labels(scores_base[indexes])
        treat_ranked = ranked_labels(scores_treat[indexes])
        slice_gold = [gold[index] for index in indexes]
        base_recall = per_document_recall_at_k(base_ranked, slice_gold, 50)
        treat_recall = per_document_recall_at_k(treat_ranked, slice_gold, 50)
        comparison[name] = {
            "baseline_recall@50": round(float(base_recall.mean()), 6),
            "treatment_recall@50": round(float(treat_recall.mean()), 6),
            "delta_recall@50": round(float(treat_recall.mean() - base_recall.mean()), 6),
            "bootstrap": paired_bootstrap_ci(
                treat_recall - base_recall,
                n_resamples=BOOTSTRAP_RESAMPLES,
                seed=SEED,
            ),
        }

    treatment_family_new = methods["treatment_chunk_mean"]["slices"]["family_new"]
    baseline_family_new = methods["baseline_whole_document"]["slices"]["family_new"]
    r50_delta = comparison["family_new"]["delta_recall@50"]
    ci = comparison["family_new"]["bootstrap"]
    r_precision_delta = round(
        treatment_family_new["primary"]["r_precision"]
        - baseline_family_new["primary"]["r_precision"],
        6,
    )
    ndcg_delta = round(
        treatment_family_new["primary"]["ndcg@10"] - baseline_family_new["primary"]["ndcg@10"], 6
    )
    conditions = {
        "r50_gain_at_least_0.01": r50_delta >= 0.01,
        "r50_ci_excludes_zero": ci["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": r_precision_delta >= -0.01,
        "ndcg10_loss_at_most_0.01": ndcg_delta >= -0.01,
    }
    verdict = {
        "bar_met": all(conditions.values()),
        "conditions": conditions,
        "family_new_delta_recall@50": r50_delta,
        "family_new_bootstrap_95ci": [ci["ci_low"], ci["ci_high"]],
        "family_new_delta_r_precision": r_precision_delta,
        "family_new_delta_ndcg@10": ndcg_delta,
        "outcome": "win" if all(conditions.values()) else "null",
    }

    print("measuring the truncation diagnostic (read-only) ...", flush=True)
    truncation = token_diagnostics(dev_texts, chunk_words=CHUNK_WORDS, backend=backend)

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-7 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 1)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "baseline_cache_version": BASELINE_CACHE_VERSION,
            "treatment_cache_version": TREATMENT_CACHE_VERSION,
            "chunk_words": CHUNK_WORDS,
            "chunking": "non-overlapping consecutive word windows, whitespace-split",
            "pooling": "mean of the chunk vectors, then L2-normalize",
            "text_view": "title + header + recitals + main_body, newline-joined",
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
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
            "chunks_embedded": int(len(flat_chunks)),
            "baseline_cache_hits": baseline_embedder.cache_hits,
            "baseline_cache_misses": baseline_embedder.cache_misses,
            "treatment_cache_hits": treatment_embedder.cache_hits,
            "treatment_cache_misses": treatment_embedder.cache_misses,
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparison,
        "verdict": verdict,
        "token_diagnostics": truncation,
        "notes": {
            "bar": (
                "A win requires family_new R@50 +0.01 with the paired 95% bootstrap "
                "interval excluding 0, and R-precision and nDCG@10 losing at most 0.01."
            ),
            "chunk_size": (
                "fixed at 200 words in the pre-registration; it is not tuned on dev in "
                "this change set."
            ),
            "baseline": (
                "the same whole-document MiniLM k-NN as sections 8-9, served from the "
                "warm eurlex-4k-v1 cache; it reproduces the section-9 numbers."
            ),
            "bootstrap": (
                "percentile interval of the mean per-document R@50 difference, resampling "
                "documents within the slice with the same row indices for both methods."
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"comparison": comparison, "verdict": verdict}, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
