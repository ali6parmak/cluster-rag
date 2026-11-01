"""The **word classes** as a classifier (the track's CS-33): the class-restricted
maps.

CS-33's readout answers the reviewer's part-of-speech question from the corpus
map's own structure (a word's class: **distinctive** — it sits in one or two of
the labelled tables; **generic** — it sits in at least a quarter of them; the
middle is **shared**), crossed with the sign of the map's log-odds. The
pre-registered falsification does not fire (the classes separate widely), so
the **payoff** is measured: a **class-restricted map** as a classifier.

Two restrictions, through the **existing** scorer and the **existing**
``evaluate`` command, on the same texts at the same operating point as the full
map — so the comparison against it is paired (``tools/lexical_compare.py``):

- **nogeneric** — the map with the **generic** class dropped (the register: the
  readout says it is the weak and often negative class);
- **distinctive** — the **distinctive-only** map (the other extreme).

Both use the same arm as CS-31's fit-side selection (``sum_zscore_positive``:
the map's weights read as the z-score, only the positive ones summed), so the
**only** difference from the full map's run is the restriction itself.

The rows land in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``corpus_map_classes`` section — ``evaluate`` is a main-pipeline command,
so the table is command-derived, never a heuristic. The narrative goes to the
track log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments_text_processing/corpus_map_classes_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The arm: CS-31's fit-side selection, unchanged. The restriction is the only
# difference between these two runs and the full map's run.
SCORING = "sum_zscore_positive"

# The two restrictions, and the run id's short name for each. The classes kept
# are the ones that still contribute to a cell's score; the dropped ones are as
# if they were not in the tables at all.
RESTRICTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nogeneric", ("distinctive", "shared")),
    ("distinctive", ("distinctive",)),
)

# The fit side's selection metric is measured at the same operating point.
FIT_SIDE_K = 2

cells = [
    EvalCell(
        label=f"upr-eval-corpus-map-{SCORING.replace('_', '-')}-{short}-val",
        predictor="corpus_map",
        params={
            "scoring": SCORING,
            "restrict": list(classes),
            "fit_side_k": FIT_SIDE_K,
            "folds": 5,
        },
        k=2,
        split="val",
    )
    for short, classes in RESTRICTIONS
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="corpus_map_classes")
