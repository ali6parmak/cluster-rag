"""The lexical kNN on UPR (the track's CS-41): how much of the retrieval is words alone?

The retrieval side's lexical counterpart. The lexical floor (CS-29) measured
the **classifier** the words alone can support (micro-F1 0.3845); this sweep
measures the **retrieval** they alone can support: the TF-IDF nearest
neighbours of the text over the train split, their labels voted with the main
pipeline's own kNN machinery — the same votes, the same power, the same top-k
as the embedding kNN reads.

**The grid** is the pool-cut family's own (k ∈ {1, 3, 5, 10, 20}): the
embedding kNN was measured on exactly it, so the two families are read at the
same operating points. **k = 10** is the primary arm — the champion's own kNN
k, so the like-for-like comparison is with ``upr-eval-knn-embeddinggemma-k10``
(shortlist R@50 0.9576 against the champion's fusion's 0.7423).

**The label-blind control** (k = 10) is the track's own yardstick: the train
records' label sets are permuted among the records, so the retrieval is
untouched and every label↔neighbour association is destroyed. It is the
empirical floor of the shortlist.

The cells are measured by the **existing** ``evaluate`` command, at the
**champion's own operating point** (``top_k`` k=2), on the **val** split — the
same metric code, the same decision rules and the same harness as every other
predictor. The rows land in the **main** UPR table (``docs/experiments_table_upr.md``)
— ``evaluate`` is a main-pipeline command, so the table is command-derived,
never a heuristic. The narrative goes to the track log
(``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments/lexical_knn_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The TF-IDF knobs are CS-29's floor's own (the pre-registered defaults,
# TEXT_PROCESSING_PLAN.md section 7.4); they are written out here so the
# frozen config records them and the run never depends on a code default.
TFIDF_PARAMS: dict[str, object] = {
    "ngram_max": 2,
    "min_df": 1,
    "sublinear_tf": True,
    "lowercase": True,
    "stop_words": None,
}

# The pool-cut family's own grid: the embedding kNN was measured on exactly it.
KS: tuple[int, ...] = (1, 3, 5, 10, 20)

# The primary arm: the champion's own kNN k (the fusion's a-term).
PRIMARY_K = 10

cells = [
    EvalCell(
        label=f"upr-eval-lexical-knn-k{k}-val",
        predictor="lexical_knn",
        params={"k": k, **TFIDF_PARAMS},
        k=2,
        split="val",
    )
    for k in KS
]

# The label-blind counterpart (the track's own yardstick): the retrieval is
# untouched, every label<->neighbour association is destroyed.
cells.append(
    EvalCell(
        label=f"upr-eval-lexical-knn-k{PRIMARY_K}-blind-val",
        predictor="lexical_knn",
        params={"k": PRIMARY_K, "label_blind": True, **TFIDF_PARAMS},
        k=2,
        split="val",
    )
)

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="lexical_knn")
