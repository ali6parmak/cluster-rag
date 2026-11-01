"""The lexical floor on UPR (the track's CS-29): TF-IDF + one-vs-rest.

One cell: the bag-of-words reference the whole lexical methods round is read
against. It is measured by the **existing** ``evaluate`` command, at the
**champion's own operating point** (``top_k`` k=2), on the **val** split — the
same metric code, the same decision rules and the same harness as every other
predictor, so the number is comparable with the champion's 0.5479 without a
second implementation of anything.

The row lands in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``baselines`` section — ``evaluate`` is a main-pipeline command, so the
table is command-derived, never a heuristic. The narrative goes to the track
log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments/lexical_tfidf_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The predictor's parameters are the pre-registered defaults
# (TEXT_PROCESSING_PLAN.md section 7.4); they are written out here so the
# frozen config records them and the run never depends on a code default.
PARAMS: dict[str, object] = {
    "ngram_max": 2,
    "min_df": 1,
    "sublinear_tf": True,
    "lowercase": True,
    "stop_words": None,
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": None,
}

cells = [
    EvalCell(
        label="upr-eval-lexical-tfidf-val",
        predictor="lexical_tfidf",
        params=PARAMS,
        k=2,
        split="val",
    )
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="baselines")
