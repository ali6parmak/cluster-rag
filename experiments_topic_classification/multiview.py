"""TC-8 = menu item 2: multi-view document embeddings on EURLEX-4K.

The whole-document view mixes four fields of very different quality: a short
clean title/header and long noisy recitals/body.  This experiment embeds each
field separately with the cached MiniLM, then compares two fixed combinations
against the whole-document baseline:

- **view mean** (pre-registered primary): L2-normalize each field vector, mean
  the present views, L2-normalize the mean, one k-NN (k=100, power 1.0);
- **rank fusion** (secondary): one k-NN per field, fused by reciprocal-rank
  fusion (top-100 per view, offset 60), no weight tuned.

The four single-view k-NNs are reported as diagnostics (which field carries the
signal), not as competing treatments.

Pre-registered in ``docs/topic_classification.md`` section 14 before this script
runs.  Artifact: ``results/eurlex-multiview/metrics.json``.  Splits: train =
index (45,000), dev = evaluation (6,000); test is never read.
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
from embed_chunkmean import evaluate_slice, knn_scores  # noqa: E402
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
from cluster_rag.topic_classification.chunking import mean_pool  # noqa: E402
from cluster_rag.topic_classification.fusion import reciprocal_rank_fusion  # noqa: E402
from cluster_rag.topic_classification.metrics import (  # noqa: E402
    paired_bootstrap_ci,
    per_document_recall_at_k,
    ranked_labels,
)

OUT = ROOT / "results" / "eurlex-multiview"
RUN_ID = "eurlex-multiview"
BASELINE_CACHE_VERSION = "eurlex-4k-v1"
VIEW_CACHE_VERSION = "eurlex-4k-views-v1"
VIEWS = ("title", "header", "recitals", "main_body")
RRF_TOP_K = 100
RRF_OFFSET = 60
BOOTSTRAP_RESAMPLES = 2_000
SEED = 42
SLICE_NAMES = ("all_dev", "family_new", "family_new_clean")


def field_texts(record: dict[str, Any]) -> dict[str, str]:
    """The four field views of one document (the same text the baseline joins)."""
    return {
        "title": record["title"],
        "header": record["header"],
        "recitals": record["recitals"],
        "main_body": "\n".join(record["main_body"]),
    }


def embed_views(
    records: list[dict[str, Any]],
    embedder: CachedEmbedder,
    dimension: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Embed each field of every record; blank views stay zero rows.

    A blank field (two dataset documents have empty recitals) is **absent**: its
    row is a zero vector, so the mean skips it (row norm 0) and its k-NN votes
    are zero. The returned masks say which rows are present per view.

    Returns:
        ``(vectors, present)`` — per view, an ``(n_documents, dimension)``
        float32 matrix and an ``(n_documents,)`` boolean presence mask.
    """
    texts_by_view = {view: [field_texts(record)[view] for record in records] for view in VIEWS}
    vectors: dict[str, np.ndarray] = {}
    present: dict[str, np.ndarray] = {}
    for view in VIEWS:
        texts = texts_by_view[view]
        present[view] = np.array([bool(text.strip()) for text in texts])
        matrix = np.zeros((len(texts), dimension), dtype=np.float32)
        if present[view].any():
            matrix[present[view]] = embedder.encode(
                [text for text, keep in zip(texts, present[view], strict=True) if keep],
                "document",
            )
        vectors[view] = matrix
        print(
            f"  {view}: {int(present[view].sum())} texts "
            f"(mean words {np.mean([len(text.split()) for text in texts]):.0f})",
            flush=True,
        )
    return vectors, present


def combine_views(vectors: dict[str, np.ndarray], present: dict[str, np.ndarray]) -> np.ndarray:
    """Mean the present view vectors per document, then L2-normalize."""
    n_documents = len(present[VIEWS[0]])
    dimension = vectors[VIEWS[0]].shape[1]
    combined = np.empty((n_documents, dimension), dtype=np.float32)
    for row in range(n_documents):
        rows = [vectors[view][row] for view in VIEWS if present[view][row]]
        if not rows:
            raise ValueError(f"document {row} has no non-blank view")
        combined[row] = mean_pool(np.stack(rows))
    return combined


def slice_bootstrap(
    baseline: np.ndarray, treatment: np.ndarray, gold: list[set[int]], mask: np.ndarray
) -> dict[str, Any]:
    """Paired per-document R@50 difference on one slice."""
    indexes = np.flatnonzero(mask)
    slice_gold = [gold[index] for index in indexes]
    base_recall = per_document_recall_at_k(ranked_labels(baseline[indexes]), slice_gold, 50)
    treat_recall = per_document_recall_at_k(ranked_labels(treatment[indexes]), slice_gold, 50)
    return {
        "baseline_recall@50": round(float(base_recall.mean()), 6),
        "treatment_recall@50": round(float(treat_recall.mean()), 6),
        "delta_recall@50": round(float(treat_recall.mean() - base_recall.mean()), 6),
        "bootstrap": paired_bootstrap_ci(
            treat_recall - base_recall, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        ),
    }


def read_bar(
    slices: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, Any],
    treatment_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered family_new bar to one treatment."""
    family_new = slices["family_new"]
    ci = family_new["bootstrap"]
    treatment_family_new = treatment_metrics["slices"]["family_new"]
    baseline_family_new = baseline_metrics["slices"]["family_new"]
    r_precision_delta = round(
        treatment_family_new["primary"]["r_precision"]
        - baseline_family_new["primary"]["r_precision"],
        6,
    )
    ndcg_delta = round(
        treatment_family_new["primary"]["ndcg@10"] - baseline_family_new["primary"]["ndcg@10"],
        6,
    )
    conditions = {
        "r50_gain_at_least_0.01": family_new["delta_recall@50"] >= 0.01,
        "r50_ci_excludes_zero": ci["ci_low"] > 0.0,
        "r_precision_loss_at_most_0.01": r_precision_delta >= -0.01,
        "ndcg10_loss_at_most_0.01": ndcg_delta >= -0.01,
    }
    return {
        "bar_met": all(conditions.values()),
        "conditions": conditions,
        "family_new_delta_recall@50": family_new["delta_recall@50"],
        "family_new_bootstrap_95ci": [ci["ci_low"], ci["ci_high"]],
        "family_new_delta_r_precision": r_precision_delta,
        "family_new_delta_ndcg@10": ndcg_delta,
        "outcome": "win" if all(conditions.values()) else "null",
    }


def main() -> None:
    """Run the multi-view experiment and write the artifact."""
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

    backend = HFEmbedder(_embedder_config())
    dimension = backend.info.dimension

    print("baseline: loading the cached whole-document vectors ...", flush=True)
    baseline_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=BASELINE_CACHE_VERSION
    )
    train_base = baseline_embedder.encode(train_texts, "document")
    dev_base = baseline_embedder.encode(dev_texts, "document")
    print(
        f"  hits {baseline_embedder.cache_hits}, misses {baseline_embedder.cache_misses}",
        flush=True,
    )

    print("treatment: embedding the four field views ...", flush=True)
    view_started = time.perf_counter()
    view_embedder = CachedEmbedder(
        backend, ROOT / "cache", preprocessing_version=VIEW_CACHE_VERSION
    )
    view_vectors, view_present = embed_views(all_records, view_embedder, dimension)
    print(
        f"  embedded in {time.perf_counter() - view_started:.1f}s "
        f"(hits {view_embedder.cache_hits}, misses {view_embedder.cache_misses})",
        flush=True,
    )

    print("combining the views (mean) ...", flush=True)
    combined = combine_views(view_vectors, view_present)
    train_mean = combined[: len(train_records)]
    dev_mean = combined[len(train_records) :]
    del combined

    print("computing the k-NN score matrices ...", flush=True)
    knn_started = time.perf_counter()
    baseline_scores = knn_scores(train_base, train_label_matrix, dev_base)
    view_mean_scores = knn_scores(train_mean, train_label_matrix, dev_mean)
    print(f"  baseline + view mean {time.perf_counter() - knn_started:.1f}s", flush=True)

    view_scores: dict[str, np.ndarray] = {}
    for view in VIEWS:
        view_started = time.perf_counter()
        view_scores[view] = knn_scores(
            view_vectors[view][: len(train_records)],
            train_label_matrix,
            view_vectors[view][len(train_records) :],
        )
        print(f"  {view} k-NN {time.perf_counter() - view_started:.1f}s", flush=True)

    print("fusing the view rankings (RRF) ...", flush=True)
    valid = [view_present[view][len(train_records) :] for view in VIEWS]
    fused_scores = reciprocal_rank_fusion(
        [view_scores[view] for view in VIEWS],
        top_k=RRF_TOP_K,
        offset=RRF_OFFSET,
        valid=valid,
    )

    print("evaluating on every slice ...", flush=True)
    methods: dict[str, Any] = {
        "baseline_whole_document": {
            "kind": "embedding_knn",
            "arm": f"MiniLM whole document, k={KNN_NEIGHBORS}, power={VOTE_POWER}",
            "cache_version": BASELINE_CACHE_VERSION,
            "cache_hits": baseline_embedder.cache_hits,
            "cache_misses": baseline_embedder.cache_misses,
        },
        "treatment_view_mean": {
            "kind": "embedding_knn",
            "arm": (f"MiniLM mean of the 4 field views, k={KNN_NEIGHBORS}, power={VOTE_POWER}"),
            "cache_version": VIEW_CACHE_VERSION,
            "cache_hits": view_embedder.cache_hits,
            "cache_misses": view_embedder.cache_misses,
        },
        "treatment_rank_fusion": {
            "kind": "rank_fusion",
            "arm": (f"RRF of the 4 per-view k-NNs, top_k={RRF_TOP_K}, offset={RRF_OFFSET}"),
            "cache_version": VIEW_CACHE_VERSION,
        },
        "diagnostic_single_view": {},
    }
    for name, scores in (
        ("baseline_whole_document", baseline_scores),
        ("treatment_view_mean", view_mean_scores),
        ("treatment_rank_fusion", fused_scores),
    ):
        methods[name]["slices"] = {
            slice_name: evaluate_slice(scores, gold, bucket_of_label, mask)
            for slice_name, mask in slices.items()
        }
    for view in VIEWS:
        methods["diagnostic_single_view"][view] = {
            slice_name: evaluate_slice(view_scores[view], gold, bucket_of_label, mask)
            for slice_name, mask in slices.items()
        }
    del view_scores

    print("paired bootstrap over documents (2,000 resamples) ...", flush=True)
    comparisons: dict[str, Any] = {}
    for name, scores in (
        ("view_mean", view_mean_scores),
        ("rank_fusion", fused_scores),
    ):
        comparisons[name] = {
            slice_name: slice_bootstrap(baseline_scores, scores, gold, mask)
            for slice_name, mask in slices.items()
        }

    primary_verdict = read_bar(
        comparisons["view_mean"],
        methods["baseline_whole_document"],
        methods["treatment_view_mean"],
    )
    secondary_verdict = read_bar(
        comparisons["rank_fusion"],
        methods["baseline_whole_document"],
        methods["treatment_rank_fusion"],
    )

    artifact = {
        "run_id": RUN_ID,
        "dataset": "eurlex_4k",
        "split": "dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "change_set": "TC-8 (TOPIC_CLASSIFICATION_PLAN.md section 6, menu item 2)",
        "config": {
            "seed": SEED,
            "label_count": len(labels),
            "label_buckets": bucket_counts,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "baseline_cache_version": BASELINE_CACHE_VERSION,
            "view_cache_version": VIEW_CACHE_VERSION,
            "views": list(VIEWS),
            "view_definition": (
                "title; header; recitals; main_body paragraphs joined by newlines; "
                "each embedded as-is (the model truncates a view above 256 tokens)"
            ),
            "view_mean": "mean of the present L2-normalized view vectors, L2-normalized",
            "rank_fusion": f"reciprocal-rank fusion, top_k={RRF_TOP_K}, offset={RRF_OFFSET}",
            "blank_view_rule": (
                "a blank view is absent: excluded from the mean, no votes, no RRF rank"
            ),
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
            "view_present_documents": {view: int(view_present[view].sum()) for view in VIEWS},
        },
        "cost": {
            "seconds_total": round(time.perf_counter() - started, 1),
            "view_texts_embedded": int(sum(int(view_present[view].sum()) for view in VIEWS)),
            "baseline_cache_hits": baseline_embedder.cache_hits,
            "baseline_cache_misses": baseline_embedder.cache_misses,
            "view_cache_hits": view_embedder.cache_hits,
            "view_cache_misses": view_embedder.cache_misses,
            "llm_calls": 0,
        },
        "methods": methods,
        "comparison": comparisons,
        "verdict": {
            "primary_treatment": "view_mean",
            "primary": primary_verdict,
            "secondary_treatment": "rank_fusion",
            "secondary": secondary_verdict,
            "outcome": (
                "win"
                if primary_verdict["bar_met"]
                else ("secondary_win_only" if secondary_verdict["bar_met"] else "null")
            ),
        },
        "notes": {
            "bar": (
                "a win requires family_new R@50 +0.01 with the paired 95% bootstrap "
                "interval excluding 0, and R-precision and nDCG@10 losing at most 0.01; "
                "view_mean is the pre-registered primary and decides the item"
            ),
            "single_view": (
                "the four per-view k-NNs are diagnostics (which field carries the "
                "signal), not competing treatments and not a selection step"
            ),
            "baseline": (
                "the same whole-document MiniLM k-NN as sections 8-9, served from the "
                "warm eurlex-4k-v1 cache; it reproduces the section-9 numbers"
            ),
            "bootstrap": (
                "percentile interval of the mean per-document R@50 difference, "
                "resampling documents within the slice"
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"comparison": comparisons, "verdict": artifact["verdict"]}, indent=2))
    print(f"wrote {OUT / 'metrics.json'} in {artifact['cost']['seconds_total']}s")


if __name__ == "__main__":
    main()
