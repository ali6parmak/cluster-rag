"""TC-9 = menu item 3: lexical k-NN fused with the embedding k-NN by RRF.

The embedding k-NN is the shortlist generator; a TF-IDF k-NN over the same text
has a different error profile.  This experiment scores the lexical arm with the
same voting rule as the embedding arm, fuses the two rankings by reciprocal-rank
fusion (top-50 per view, offset 60, no weight tuned), and reads the result on
all_dev / family_new / family_new_clean against the embedding k-NN.

Pre-registered in ``docs/topic_classification.md`` section 15 before this script
runs.  Artifact: ``results/eurlex-lexical-rrf/metrics.json``.  Splits: train =
index (45,000), dev = evaluation (6,000); test is never read.
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

from audit_eurlex import title_skeleton  # noqa: E402
from embed_chunkmean import evaluate_slice, knn_scores  # noqa: E402
from multiview import read_bar, slice_bootstrap  # noqa: E402
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
from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion  # noqa: E402
from cluster_rag.topic_classification.retrieval import vote_scores_from_similarity  # noqa: E402

OUT = ROOT / "results" / "eurlex-lexical-rrf"
RUN_ID = "eurlex-lexical-rrf"
EMBEDDING_CACHE_VERSION = "eurlex-4k-v1"
TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200_000,
    "sublinear_tf": True,
    "dtype": "float32",
}
RRF_TOP_K = 50
RRF_OFFSET = 60
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")
BATCH = 256


def lexical_scores(
    dev_matrix: Any,
    train_matrix: Any,
    train_label_matrix: np.ndarray,
) -> np.ndarray:
    """Cosine k-NN votes of the TF-IDF rows, batched over the dev documents."""
    n_documents = dev_matrix.shape[0]
    scores = np.zeros((n_documents, train_label_matrix.shape[1]), dtype=np.float32)
    for start in range(0, n_documents, BATCH):
        stop = min(start + BATCH, n_documents)
        similarity = np.asarray((dev_matrix[start:stop] @ train_matrix.T).toarray())
        scores[start:stop] = vote_scores_from_similarity(
            similarity,
            train_label_matrix,
            k=KNN_NEIGHBORS,
            power=VOTE_POWER,
        )
    return scores


def main() -> None:
    """Run the lexical + RRF experiment and write the artifact."""
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

    print("baseline: loading the cached whole-document vectors ...", flush=True)
    backend = HFEmbedder(_embedder_config())
    embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=EMBEDDING_CACHE_VERSION
    )
    train_vectors = embedder.encode(train_texts, "document")
    dev_vectors = embedder.encode(dev_texts, "document")
    print(
        f"  hits {embedder.cache_hits}, misses {embedder.cache_misses}",
        flush=True,
    )

    print("lexical: fitting TF-IDF (word 1-2 grams) on train ...", flush=True)
    tfidf_started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        lowercase=TFIDF_CONFIG["lowercase"],
        ngram_range=TFIDF_CONFIG["ngram_range"],
        min_df=TFIDF_CONFIG["min_df"],
        max_features=TFIDF_CONFIG["max_features"],
        sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    dev_matrix = vectorizer.transform(dev_texts)
    print(
        f"  vocabulary {len(vectorizer.vocabulary_)}, train nnz {train_matrix.nnz}, "
        f"dev nnz {dev_matrix.nnz}, fit {time.perf_counter() - tfidf_started:.1f}s",
        flush=True,
    )

    print("scoring the two k-NNs ...", flush=True)
    knn_started = time.perf_counter()
    embedding_scores = knn_scores(train_vectors, train_label_matrix, dev_vectors)
    print(f"  embedding k-NN {time.perf_counter() - knn_started:.1f}s", flush=True)
    lexical_started = time.perf_counter()
    tfidf_scores = lexical_scores(dev_matrix, train_matrix, train_label_matrix)
    print(f"  lexical k-NN {time.perf_counter() - lexical_started:.1f}s", flush=True)

    print("fusing the two rankings (RRF) ...", flush=True)
    fused_scores = reciprocal_rank_fusion(
        [embedding_scores, tfidf_scores],
        top_k=RRF_TOP_K,
        offset=RRF_OFFSET,
    )

    print("evaluating on every slice ...", flush=True)
    methods: dict[str, Any] = {
        "baseline_embedding_knn": {
            "kind": "embedding_knn",
            "arm": f"MiniLM whole document, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
            "cache_version": EMBEDDING_CACHE_VERSION,
        },
        "diagnostic_lexical_knn": {
            "kind": "lexical_knn",
            "arm": f"TF-IDF word 1-2 grams, cosine, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
            "vocabulary": int(len(vectorizer.vocabulary_)),
        },
        "treatment_rrf": {
            "kind": "rank_fusion",
            "arm": (f"RRF(embedding, lexical), top_k={RRF_TOP_K}, offset={RRF_OFFSET}"),
        },
    }
    for name, scores in (
        ("baseline_embedding_knn", embedding_scores),
        ("diagnostic_lexical_knn", tfidf_scores),
        ("treatment_rrf", fused_scores),
    ):
        methods[name]["slices"] = {
            slice_name: evaluate_slice(scores, gold, bucket_of_label, mask)
            for slice_name, mask in slices.items()
        }

    print("paired bootstrap over documents (2,000 resamples) ...", flush=True)
    comparisons: dict[str, Any] = {}
    for name, scores in (
        ("fusion", fused_scores),
        ("lexical", tfidf_scores),
    ):
        comparisons[name] = {
            slice_name: slice_bootstrap(embedding_scores, scores, gold, mask)
            for slice_name, mask in slices.items()
        }
    # Post-hoc diagnostic, not part of the pre-registered bar: how much the
    # fusion adds over its stronger single arm (the lexical k-NN).
    comparisons["fusion_minus_lexical"] = {
        slice_name: slice_bootstrap(tfidf_scores, fused_scores, gold, mask)
        for slice_name, mask in slices.items()
    }

    verdict = read_bar(
        comparisons["fusion"],
        methods["baseline_embedding_knn"],
        methods["treatment_rrf"],
    )

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-9 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 3)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "embedding_cache_version": EMBEDDING_CACHE_VERSION,
            "tfidf": TFIDF_CONFIG,
            "lexical_knn": {"cosine": True, "k": KNN_NEIGHBORS, "power": VOTE_POWER},
            "rank_fusion": f"reciprocal-rank fusion, top_k={RRF_TOP_K}, offset={RRF_OFFSET}",
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
            "tfidf_vocabulary": int(len(vectorizer.vocabulary_)),
            "tfidf_train_nnz": int(train_matrix.nnz),
            "tfidf_dev_nnz": int(dev_matrix.nnz),
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparisons,
        "verdict": verdict,
        "notes": {
            "bar": (
                "a win requires family_new R@50 +0.01 with the paired 95% bootstrap "
                "interval excluding 0, and R-precision and nDCG@10 losing at most 0.01"
            ),
            "lexical_alone": (
                "the lexical k-NN is a diagnostic (its error profile and tail recall), "
                "not a competing treatment; the fusion is the treatment"
            ),
            "fusion_candidate_pool": (
                "the union of the two top-50 lists, so R@100 is capped by that union; "
                "R@50 is the primary and is not capped"
            ),
            "bm25": (
                "TF-IDF is the stand-in available without a new dependency; BM25 proper "
                "is not tested here"
            ),
            "fusion_minus_lexical": (
                "a post-hoc diagnostic, not part of the bar: it isolates how much RRF "
                "adds over its stronger single arm"
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"comparison": comparisons, "verdict": verdict}, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
