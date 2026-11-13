"""TC-13 = menu item 7: cross-encoder rerank of the lexical top-50.

The lexical k-NN is the strongest single EURLEX ranking but its ordering is the
weak point.  This script scores the lexical base's top-50 candidates with two
cross-encoders (ms-marco MiniLM L-6 primary, L-12 secondary), reorders the
top-50 by the cross-encoder score (positions 51+ unchanged, so R@50 is
identical by construction), and reads the ordering metrics on all_dev /
family_new / family_new_clean.

Pre-registered in ``docs/topic_classification.md`` section 19 before this script
runs.  Artifact: ``results/eurlex-cross-encoder-rerank/metrics.json``.  Splits:
train = index (45,000), dev = evaluation (6,000); test is never read.  Requires
GPU device access.
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

from audit_eurlex import read_jsonl, title_skeleton  # noqa: E402
from lexical_rrf import TFIDF_CONFIG, lexical_scores  # noqa: E402
from shortlist_baselines import (  # noqa: E402
    NEAR_DUP_THRESHOLD,
    full_text,
    label_buckets,
    load_split,
    near_duplicate_cosines,
)

from cluster_rag.config import RerankerModelConfig  # noqa: E402
from cluster_rag.rerank.base import RerankRequest  # noqa: E402
from cluster_rag.rerank.cache import CachedReranker  # noqa: E402
from cluster_rag.rerank.hf import HFReranker  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    evaluate_ranked,
    paired_bootstrap_ci,
    per_document_ndcg_at_k,
    per_document_r_precision,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-cross-encoder-rerank"
RUN_ID = "eurlex-cross-encoder-rerank"
CANDIDATE_DEPTH = 50
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
NDCG_BAR = 0.01
R_PRECISION_TOLERANCE = 0.005
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")

L6 = RerankerModelConfig(
    name="msmarco-l6",
    model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    revision="c5ee24cb16019beea0893ab7796b1df96625c6b8",
    kind="cross_encoder",
    dtype="float32",
    max_length=512,
    batch_size=128,
    device="cuda",
    trust_remote_code=False,
)
L12 = RerankerModelConfig(
    name="msmarco-l12",
    model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
    revision="7b0235231ca2674cb8ca8f022859a6eba2b1c968",
    kind="cross_encoder",
    dtype="float32",
    max_length=512,
    batch_size=64,
    device="cuda",
    trust_remote_code=False,
)


def build_reranker(config: RerankerModelConfig) -> CachedReranker:
    """The client plus its own disk cache (the UPR track's own wiring shape)."""
    shape = {
        "max_length": config.max_length,
        "max_documents": config.max_documents,
        "max_query_length": config.max_query_length,
        "max_doc_length": config.max_doc_length,
        "template": config.template,
        "padding_side": config.padding_side,
        "batch_size": config.batch_size,
    }
    return CachedReranker(HFReranker(config), ROOT / "cache", shape=shape)


def slice_view(
    ranked: np.ndarray, gold: list[set[int]], mask: np.ndarray, bucket_of_label: dict[int, str]
) -> dict[str, Any]:
    """Evaluate a ranked matrix on one document mask."""
    indexes = np.flatnonzero(mask)
    return evaluate_ranked(ranked[indexes], [gold[index] for index in indexes], bucket_of_label)


def bootstrap_ranked(
    base_ranked: np.ndarray,
    treatment_ranked: np.ndarray,
    gold: list[set[int]],
    mask: np.ndarray,
    *,
    metric: str,
    k: int = 10,
) -> dict[str, Any]:
    """Paired per-document difference of one metric between two ranked matrices."""
    indexes = np.flatnonzero(mask)
    slice_gold = [gold[index] for index in indexes]

    def _values(ranked: np.ndarray) -> np.ndarray:
        if metric == "ndcg":
            return per_document_ndcg_at_k(ranked[indexes], slice_gold, k)
        return per_document_r_precision(ranked[indexes], slice_gold)

    base_values = _values(base_ranked)
    treatment_values = _values(treatment_ranked)
    return {
        "baseline": round(float(base_values.mean()), 6),
        "treatment": round(float(treatment_values.mean()), 6),
        "delta": round(float(treatment_values.mean() - base_values.mean()), 6),
        "bootstrap": paired_bootstrap_ci(
            treatment_values - base_values, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
    }


def main() -> None:
    """Run the two rerankers and the comparison."""
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

    vocabulary = {
        str(record["id"]): record["title"]
        for record in read_jsonl(ROOT / "data" / "eurlex_4k" / "eurovoc_concepts.jsonl")
    }
    missing = [label for label in labels if not vocabulary.get(label, "").strip()]
    if missing:
        raise ValueError(f"{len(missing)} labels have no title: {missing[:5]}")
    label_titles = [vocabulary[label] for label in labels]

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

    print("building the lexical base ranking ...", flush=True)
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
    base_ranked = ranked_labels(lexical)
    candidates = base_ranked[:, :CANDIDATE_DEPTH]
    print(f"  vocabulary {len(vectorizer.vocabulary_)}, candidates {candidates.shape}", flush=True)

    requests = [
        RerankRequest(
            query=dev_texts[row],
            documents=tuple(label_titles[index] for index in candidates[row]),
        )
        for row in range(len(dev_records))
    ]

    methods: dict[str, Any] = {
        "baseline_lexical_knn": {
            "kind": "knn",
            "arm": "lexical TF-IDF top-50 (the base ranking)",
            "slices": {
                slice_name: slice_view(base_ranked, gold, mask, bucket_of_label)
                for slice_name, mask in slices.items()
            },
        }
    }
    comparison: dict[str, Any] = {}
    verdicts: dict[str, Any] = {}
    model_reports: dict[str, Any] = {}

    for role, config in (("primary", L6), ("secondary", L12)):
        print(f"=== {config.name}: scoring {len(requests)} requests ...", flush=True)
        model_started = time.perf_counter()
        reranker = build_reranker(config)
        scores = reranker.score(requests)
        usage = dict(getattr(reranker, "usage", {}))
        elapsed = time.perf_counter() - model_started
        print(
            f"  {elapsed:.1f}s (cache hits {reranker.cache_hits}, misses {reranker.cache_misses}, "
            f"truncated {usage.get('n_truncated_inputs', 0)})",
            flush=True,
        )

        reranked = np.empty_like(base_ranked)
        for row in range(len(dev_records)):
            document_scores = np.asarray(scores[row], dtype=np.float64)
            order = np.argsort(-document_scores, kind="stable")
            reranked[row, :CANDIDATE_DEPTH] = candidates[row][order]
            reranked[row, CANDIDATE_DEPTH:] = base_ranked[row, CANDIDATE_DEPTH:]

        methods[config.name] = {
            "kind": "cross_encoder_rerank",
            "arm": f"{config.name} rerank of the lexical top-{CANDIDATE_DEPTH}",
            "model": reranker.info.model_id,
            "revision": reranker.info.revision,
            "score_transform": reranker.info.extra.get("score_transform"),
            "device": reranker.info.extra.get("device"),
            "dtype": reranker.info.extra.get("dtype"),
            "slices": {
                slice_name: slice_view(reranked, gold, mask, bucket_of_label)
                for slice_name, mask in slices.items()
            },
        }
        model_reports[config.name] = {
            "role": role,
            "model_id": config.model_id,
            "revision": config.revision,
            "max_length": config.max_length,
            "batch_size": config.batch_size,
            "dtype": config.dtype,
            "device": config.device,
            "cache_hits": reranker.cache_hits,
            "cache_misses": reranker.cache_misses,
            "seconds": round(elapsed, 1),
            "usage": usage,
        }
        comparison[config.name] = {
            slice_name: {
                "ndcg@10": bootstrap_ranked(base_ranked, reranked, gold, mask, metric="ndcg", k=10),
                "r_precision": bootstrap_ranked(
                    base_ranked, reranked, gold, mask, metric="r_precision"
                ),
            }
            for slice_name, mask in slices.items()
        }

        ndcg_family_new = comparison[config.name]["family_new"]["ndcg@10"]
        r_precision_family_new = comparison[config.name]["family_new"]["r_precision"]
        r50_base = methods["baseline_lexical_knn"]["slices"]["family_new"]["primary"]["recall@50"]
        r50_model = methods[config.name]["slices"]["family_new"]["primary"]["recall@50"]
        conditions = {
            "ndcg10_gain_at_least_0.01": ndcg_family_new["delta"] >= NDCG_BAR,
            "ndcg10_ci_excludes_zero": ndcg_family_new["bootstrap"]["ci_low"] > 0.0,
            "r_precision_loss_at_most_0.005": r_precision_family_new["delta"]
            >= -R_PRECISION_TOLERANCE,
            "r50_unchanged": abs(r50_model - r50_base) <= 1e-9,
        }
        verdicts[config.name] = {
            "role": role,
            "bar_met": all(conditions.values()),
            "conditions": conditions,
            "family_new_ndcg@10": ndcg_family_new["treatment"],
            "family_new_ndcg@10_baseline": ndcg_family_new["baseline"],
            "family_new_ndcg@10_delta": ndcg_family_new["delta"],
            "family_new_ndcg@10_95ci": [
                ndcg_family_new["bootstrap"]["ci_low"],
                ndcg_family_new["bootstrap"]["ci_high"],
            ],
            "family_new_r_precision_delta": r_precision_family_new["delta"],
            "family_new_r50": r50_model,
            "family_new_r50_baseline": r50_base,
            "outcome": "win" if all(conditions.values()) else "null",
        }
        print(
            f"  family_new nDCG@10 {ndcg_family_new['baseline']:.4f} -> "
            f"{ndcg_family_new['treatment']:.4f} (delta {ndcg_family_new['delta']:+.4f}), "
            f"R-precision delta {r_precision_family_new['delta']:+.4f}",
            flush=True,
        )

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-13 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 7)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "base": "lexical TF-IDF k-NN (TC-9), top-50 candidates",
            "tfidf": TFIDF_CONFIG,
            "candidate_depth": CANDIDATE_DEPTH,
            "pair": "query = full_text; document = candidate label title; max_length 512",
            "rerank": "sort the top-50 by the cross-encoder score (ties: base rank); 51+ unchanged",
            "models": {
                "primary": {"model_id": L6.model_id, "revision": L6.revision},
                "secondary": {"model_id": L12.model_id, "revision": L12.revision},
            },
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "slices": list(SLICE_NAMES),
        },
        "slices": {
            "documents": {name: int(mask.sum()) for name, mask in slices.items()},
            "clean_documents": int(clean.sum()),
            "family_new_documents": int(family_new.sum()),
        },
        "cost": {"seconds_total": round(time.perf_counter() - started, 1), "llm_calls": 0},
        "models": model_reports,
        "methods": methods,
        "comparison": comparison,
        "verdict": {
            "primary_model": L6.name,
            "primary": verdicts[L6.name],
            "secondary_model": L12.name,
            "secondary": verdicts[L12.name],
            "outcome": (
                "win"
                if verdicts[L6.name]["bar_met"]
                else ("secondary_win_only" if verdicts[L12.name]["bar_met"] else "null")
            ),
        },
        "notes": {
            "bar": (
                "a win requires family_new nDCG@10 +0.01 with the paired 95% interval "
                "excluding 0, R-precision losing at most 0.005, and R@50 unchanged"
            ),
            "r50": "the candidate set is the base's top-50, so R@50 cannot move",
            "models": (
                "ms-marco cross-encoders trained for short-query -> long-passage; the "
                "query here is the document and the document is the label title, an "
                "off-distribution role assignment for the model"
            ),
            "cache": "scores in cache/rerank (the rerankers' own cache, not the embeddings')",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "comparison": {
                    name: {
                        slice_name: {
                            metric: values[metric]["delta"] for metric in ("ndcg@10", "r_precision")
                        }
                        for slice_name, values in by_slice.items()
                    }
                    for name, by_slice in comparison.items()
                },
                "verdict": artifact["verdict"],
            },
            indent=1,
        )
    )
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
