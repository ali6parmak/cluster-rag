"""The **union of the lexical and the embedding kNN** on UPR (the track's CS-46).

Two retrievers reach nearly the same breadth — the lexical kNN (CS-41) a
shortlist R@50 of 0.9437, the embedding kNN 0.9576 — and differ in their
failures. This sweep measures their **union**: the per-row **z-max** of the two
rankings (CS-14's rule), at both halves' primary k = 10 (the champion's own
kNN k), on the **val** split — the candidate generator's ceiling, which is the
number PLAN.md's deferred BM25 decision is waiting for.

**The falsification, registered before the call.** The union's **recall@10
must be at least 0.9120** (the decision model's union at the same k). At or
above it, the two lexical retrievers together supply the breadth the decision
model's union supplies and the BM25 question gets its evidence; below it, the
shortlist gap is not a breadth gap. The **recall@50** must not fall below the
better arm's 0.9576 (a fusion that dilutes the breadth is a null regardless of
its recall@10).

**The label-blind counterpart** (the track's own yardstick) permutes the train
records' label sets once and hands the same permuted labels to both halves: it
is the fusion's empirical floor.

The cells are measured by the **existing** ``evaluate`` command at the
**champion's own operating point** (``top_k`` k=2) — the same harness as every
other predictor. ``evaluate`` is a main-pipeline command, so the rows land in
the **main** UPR table (``docs/experiments_table_upr.md``, the section
``## UPR — knn_union``); the narrative goes to the track log.

    uv run cluster-rag sweep --file experiments/knn_union_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The train embedding artifact the embedding half reads (the lexical half needs
# no embedding; the union declares the dependency because one half has it).
EMBEDDING_RUN = "upr-embed-embeddinggemma"

# The TF-IDF knobs are CS-29's floor's / CS-41's own (the pre-registered
# defaults, written out so the frozen config records them).
TFIDF_PARAMS: dict[str, object] = {
    "ngram_max": 2,
    "min_df": 1,
    "sublinear_tf": True,
    "lowercase": True,
    "stop_words": None,
}

# Both halves at the champion's own kNN k (the like-for-like operating point).
K = 10

cells = [
    # the union itself: the candidate generator's ceiling
    EvalCell(
        label="upr-eval-knn-union-val",
        predictor="knn_union",
        params={"k": K, **TFIDF_PARAMS},
        k=2,
        split="val",
        embedding_run=EMBEDDING_RUN,
    ),
    # the label-blind counterpart: the fusion's empirical floor
    EvalCell(
        label="upr-eval-knn-union-blind-val",
        predictor="knn_union",
        params={"k": K, "label_blind": True, **TFIDF_PARAMS},
        k=2,
        split="val",
        embedding_run=EMBEDDING_RUN,
    ),
]

SWEEP = define_sweep(cells, dataset="upr", label="knn_union")
