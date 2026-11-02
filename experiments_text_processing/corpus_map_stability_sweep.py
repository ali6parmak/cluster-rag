"""The **stability map** as a classifier (the track's CS-36): the filtered map.

The corpus map's bulk is fragile — two thirds of its 18,707 (label, word)
pairs rest on a **single** train text — and until now nothing had measured
how much of it is sampling noise. CS-36 resamples the train split, rebuilds the
map on each resample and reads every pair's survival; this sweep applies the
**filter** the readout licenses: only the pairs whose sign survives in at
least ``stability_min`` of the resamples contribute to a cell's score.

Five runs — the four filtered arms and the control's own baseline — through the
**existing** scorer and the **existing** `evaluate` command, on the same texts
at the same operating point as the full map — so the comparison against it is
paired (``tools/lexical_compare.py``):

- **stable75** — the filter at ``tau = 0.75`` (a link resting on at least two
  train texts; a link resting on one survives about 63 % of the resamples, one
  resting on two about 86 %);
- **stable90** — the filter at ``tau = 0.90`` (roughly at least three train
  texts);
- and each of them with its **label-blind counterpart** (the map built with
  the labels permuted among the train records): the control that says whether
  a gain is label-driven.

The **blind map without the filter** is the control's own baseline (the
unfiltered arm of the same blind map), so the control's paired difference is
measured the same way as the treatment's: same texts, same operating point,
same tool.

The arm is CS-31's fit-side selection (``sum_zscore_positive``), unchanged: the
filter is the **only** difference between these runs and the full map's run.

The rows land in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``corpus_map_stability`` section — ``evaluate`` is a main-pipeline command,
so the table is command-derived, never a heuristic. The narrative goes to the
track log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments_text_processing/corpus_map_stability_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The arm: CS-31's fit-side selection, unchanged. The filter is the only
# difference between these runs and the full map's run.
SCORING = "sum_zscore_positive"

# The fit side's selection metric is measured at the same operating point.
FIT_SIDE_K = 2

# The resamples the stability estimate rests on (B = 500, the pre-registered
# value) — the same number the map's own artifact uses.
STABILITY_RESAMPLES = 500

# The arms: the run id's short name, the survival threshold (``None`` = no
# filter) and whether the map is the label-blind counterpart. The two
# thresholds are the pre-registered ones (0.75 and 0.90).
ARMS: tuple[tuple[str, float | None, bool], ...] = (
    ("stable75", 0.75, False),
    ("stable90", 0.90, False),
    ("stable75-blind", 0.75, True),
    ("stable90-blind", 0.90, True),
    # the control's own baseline: the blind map WITHOUT the filter, so the
    # control's paired difference is measured like the treatment's.
    ("blind", None, True),
)

cells = [
    EvalCell(
        label=f"upr-eval-corpus-map-{SCORING.replace('_', '-')}-{short}-val",
        predictor="corpus_map",
        params={
            "scoring": SCORING,
            "stability_min": threshold,
            "stability_resamples": STABILITY_RESAMPLES,
            "label_blind": blind,
            "fit_side_k": FIT_SIDE_K,
            "folds": 5,
        },
        k=2,
        split="val",
    )
    for short, threshold, blind in ARMS
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="corpus_map_stability")
