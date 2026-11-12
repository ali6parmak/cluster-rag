"""First EURLEX-4K experiment: dictionary vs embedding k-NN shortlists.

The product question: with 4,271 labels, can the train-only corpus dictionary
produce a useful shortlist, and how does it compare with a cheap embedding
k-NN?  The experiment is pre-registered in ``docs/topic_classification.md``
(section "First experiment") before this script runs.

Artifacts (gitignored, like every run):
    results/eurlex-shortlist/metrics.json
    results/eurlex-shortlist/config.json

The script reads ``data/eurlex_4k/`` read-only, uses train as the index/fit
split and dev as the evaluation split, and never touches test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as normalize_rows

from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.embed.hf import HFEmbedder
from cluster_rag.retrieve.knn import knn_vote_scores
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.lexical.corpus_map_score import score_cells
from cluster_rag.topic_classification.metrics import evaluate_ranking

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eurlex_4k"
OUT = ROOT / "results" / "eurlex-shortlist"
EMBEDDER_NAME = "minilm"
EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEAR_DUP_THRESHOLD = 0.90
KNN_NEIGHBORS = 100
VOTE_POWER = 1.0
DICTIONARY_ARMS = ("sum_zscore_positive", "sum_logodds", "support_count")
SEED = 42


def load_split(split: str) -> list[dict[str, Any]]:
    """Read one split's JSONL records."""
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def full_text(record: dict[str, Any]) -> str:
    """The document as the dictionary and the embedding see it."""
    return "\n".join(
        [record["title"], record["header"], record["recitals"], "\n".join(record["main_body"])]
    )


def near_dup_text(record: dict[str, Any]) -> str:
    """The audit's near-duplicate view: title, header and 2,500 chars of each body."""
    return " ".join(
        [
            record["title"],
            record["header"],
            record["recitals"][:2500],
            " ".join(record["main_body"])[:2500],
        ]
    )


def near_duplicate_cosines(
    train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]]
) -> np.ndarray:
    """Each dev document's best TF-IDF cosine to a train document."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_texts = [near_dup_text(record) for record in train_records]
    dev_texts = [near_dup_text(record) for record in dev_records]
    vectorizer.fit(train_texts)
    train_matrix = normalize_rows(vectorizer.transform(train_texts))
    dev_matrix = normalize_rows(vectorizer.transform(dev_texts))
    maxima = np.empty(len(dev_texts), dtype=np.float32)
    for start in range(0, len(dev_texts), 500):
        similarity = dev_matrix[start : start + 500] @ train_matrix.T
        maxima[start : start + 500] = np.asarray(similarity.max(axis=1).toarray()).ravel()
    return maxima


def label_buckets(
    labels: list[str], train_labels: list[list[str]]
) -> tuple[dict[int, str], dict[str, int]]:
    """Map each label column to head/torso/tail/unseen and count the buckets."""
    support = dict.fromkeys(labels, 0)
    for record_labels in train_labels:
        for label in set(record_labels):
            support[label] += 1
    bucket_of_label: dict[int, str] = {}
    counts = {"head": 0, "torso": 0, "tail": 0, "unseen": 0}
    for index, label in enumerate(labels):
        value = support[label]
        if value >= 100:
            bucket = "head"
        elif value >= 10:
            bucket = "torso"
        elif value >= 1:
            bucket = "tail"
        else:
            bucket = "unseen"
        bucket_of_label[index] = bucket
        counts[bucket] += 1
    return bucket_of_label, counts


def evaluate(
    scores: np.ndarray,
    gold: list[set[int]],
    bucket_of_label: dict[int, str],
    clean_mask: np.ndarray,
) -> dict[str, Any]:
    """Overall and clean-slice metrics for one score matrix."""
    overall = evaluate_ranking(scores, gold, bucket_of_label)
    clean_gold = [labels for labels, keep in zip(gold, clean_mask, strict=True) if keep]
    clean = (
        evaluate_ranking(scores[clean_mask], clean_gold, bucket_of_label) if clean_gold else None
    )
    return {"overall": overall, "clean": clean}


def main() -> None:
    """Run the experiment and write the artifact."""
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

    print(f"train {len(train_records)}, dev {len(dev_records)}, labels {len(labels)}", flush=True)
    print("computing the near-duplicate slice ...", flush=True)
    max_cosines = near_duplicate_cosines(train_records, dev_records)
    clean_mask = max_cosines < NEAR_DUP_THRESHOLD
    print(
        f"clean slice {int(clean_mask.sum())}/{len(clean_mask)} "
        f"({clean_mask.mean():.1%}); leaky {int((~clean_mask).sum())}",
        flush=True,
    )

    print("building the train-only corpus dictionary ...", flush=True)
    dictionary_started = time.perf_counter()
    corpus_map = build_corpus_map(train_texts, train_labels, labels)
    print(
        f"map built in {time.perf_counter() - dictionary_started:.1f}s: "
        f"{corpus_map.n_pairs} pairs, {corpus_map.n_words} words",
        flush=True,
    )

    methods: dict[str, Any] = {}
    for arm in DICTIONARY_ARMS:
        arm_started = time.perf_counter()
        scores = score_cells(dev_texts, corpus_map, scoring=arm)
        methods[f"dictionary_{arm}"] = {
            "kind": "dictionary",
            "arm": arm,
            "seconds": round(time.perf_counter() - arm_started, 1),
            **evaluate(scores, gold, bucket_of_label, clean_mask),
        }
        del scores
        print(
            f"  {arm}: {time.perf_counter() - arm_started:.1f}s, "
            f"R@50 {methods[f'dictionary_{arm}']['overall']['primary']['recall@50']:.4f}",
            flush=True,
        )

    print("embedding the train and dev documents with MiniLM ...", flush=True)
    embedding_started = time.perf_counter()
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(backend, ROOT / "cache", preprocessing_version="eurlex-4k-v1")
    train_vectors = embedder.encode(train_texts, "document")
    dev_vectors = embedder.encode(dev_texts, "document")
    print(
        f"embedded in {time.perf_counter() - embedding_started:.1f}s "
        f"(cache hits {embedder.cache_hits}, misses {embedder.cache_misses})",
        flush=True,
    )

    print("computing the k-NN shortlist ...", flush=True)
    knn_started = time.perf_counter()
    train_label_matrix = np.zeros((len(train_records), len(labels)), dtype=bool)
    for row, record_labels in enumerate(train_labels):
        for label in record_labels:
            train_label_matrix[row, label_index[label]] = True
    knn_scores = np.zeros((len(dev_records), len(labels)), dtype=np.float32)
    for start in range(0, len(dev_records), 256):
        stop = min(start + 256, len(dev_records))
        knn_scores[start:stop] = knn_vote_scores(
            train_vectors,
            train_label_matrix,
            dev_vectors[start:stop],
            k=KNN_NEIGHBORS,
            power=VOTE_POWER,
        )
    methods["embedding_knn"] = {
        "kind": "embedding_knn",
        "arm": f"MiniLM, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
        "seconds": round(time.perf_counter() - knn_started, 1),
        **evaluate(knn_scores, gold, bucket_of_label, clean_mask),
    }
    print(
        f"  k-NN: {time.perf_counter() - knn_started:.1f}s, "
        f"R@50 {methods['embedding_knn']['overall']['primary']['recall@50']:.4f}",
        flush=True,
    )
    del knn_scores

    print("computing the train-frequency prior ...", flush=True)
    prior = np.zeros(len(labels), dtype=np.float32)
    for label in labels:
        prior[label_index[label]] = sum(
            1 for record_labels in train_labels if label in set(record_labels)
        )
    prior_scores = np.tile(prior, (len(dev_records), 1))
    methods["frequency_prior"] = {
        "kind": "frequency_prior",
        "arm": "train document frequency",
        "seconds": 0.0,
        **evaluate(prior_scores, gold, bucket_of_label, clean_mask),
    }

    artifact = {
        "run_id": "eurlex-shortlist",
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedder": EMBEDDER_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "knn_neighbors": KNN_NEIGHBORS,
            "vote_power": VOTE_POWER,
            "dictionary_arms": list(DICTIONARY_ARMS),
            "near_duplicate_threshold": NEAR_DUP_THRESHOLD,
            "clean_documents": int(clean_mask.sum()),
            "leaky_documents": int((~clean_mask).sum()),
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "dictionary_pairs": corpus_map.n_pairs,
            "dictionary_words": corpus_map.n_words,
            "embedding_cache_hits": embedder.cache_hits,
            "embedding_cache_misses": embedder.cache_misses,
            "llm_calls": 0,
        },
        "dev_max_train_cosine": [round(float(value), 6) for value in max_cosines],
        "methods": methods,
        "notes": {
            "metrics": (
                "recall@k and R-precision are macro over documents; nDCG uses binary "
                "relevance; label_recall@50 is micro over gold label assignments."
            ),
            "ties": "descending stable sort; ties break by label column order.",
            "dictionary": (
                "train-only corpus map (word↔label), labels with no train example carry "
                "the row floor; sum_zscore_positive is the primary arm."
            ),
            "embedding": (
                "MiniLM-L6-v2 sees the whole document text but the model context is 256 "
                "tokens, so it truncates; the dictionary reads the whole text."
            ),
            "clean_slice": (
                "dev documents whose best train TF-IDF cosine is below the near-duplicate "
                "threshold (0.90); the audit measured 26.1% of dev/test docs above it."
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "config.json").write_text(
        json.dumps(artifact["config"] | {"notes": artifact["notes"]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


def _embedder_config() -> Any:
    """The pinned MiniLM embedder config (the shared pool's entry)."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from embedder_pool import MODELS  # noqa: PLC0415  (the sweep loader's own seam)

    return MODELS[EMBEDDER_NAME]


if __name__ == "__main__":
    main()
