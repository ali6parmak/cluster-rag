"""The corpus map as a classifier on UPR (the track's CS-30, extended by CS-31):
the arms.

One cell per scoring function, all on the **val** split at the champion's own
operating point (``top_k`` k=2), all measured by the **existing** ``evaluate``
command — the same instrument that measured CS-29's lexical floor and the
champion.

The **choice** between the arms is not made here: it is made on the fit side
(the out-of-fold train micro-F1, which every artifact carries), so nothing in
this sweep picks a winner on val. The val rows are the map's own **measure
ablation**: does the log-odds weighting beat a plain count of the label's table
words? — and (CS-31) does the z-scored weight move the classifier?

The rows land in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``corpus_map`` section — ``evaluate`` is a main-pipeline command, so the
table is command-derived, never a heuristic. The narrative goes to the track
log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments/corpus_map_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The four pre-registered scoring functions (TEXT_PROCESSING_PLAN.md section
# 7.4) plus CS-31's two z-scored arms (the map's weights read as the z-score
# instead of the log-odds). The run id names the arm, so the table's rows are
# readable without the artifact.
ARMS: tuple[tuple[str, str], ...] = (
    ("sum_logodds", "sum-logodds"),
    ("sum_positive", "sum-positive"),
    ("max_logodds", "max-logodds"),
    ("support_count", "support-count"),
    ("sum_zscore", "sum-zscore"),
    ("sum_zscore_positive", "sum-zscore-positive"),
)

# The fit side's selection metric is measured at the same operating point.
FIT_SIDE_K = 2

cells = [
    EvalCell(
        label=f"upr-eval-corpus-map-{short}-val",
        predictor="corpus_map",
        params={"scoring": scoring, "fit_side_k": FIT_SIDE_K, "folds": 5},
        k=2,
        split="val",
    )
    for scoring, short in ARMS
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="corpus_map")
