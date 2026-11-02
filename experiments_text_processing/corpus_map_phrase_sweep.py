"""The **phrase map** as a classifier (the track's CS-39): the union arm.

The corpus map's unit is a single word; UPR's genre is formulaic, so the
phrase table (the map over the corpus's bigrams) is where the genre's
structure would show. This sweep measures the **union** — the word map's score
plus the phrase map's own, through the same scorer — against the word map
alone.

Two arms, through the **existing** scorer and the **existing** `evaluate`
command, on the same texts at the same operating point as the word map — so the
comparison against it is paired (``tools/lexical_compare.py``):

- **phrase** — the union of the word map and the phrase map;
- **phrase-blind** — the union on the **label-blind** maps (the labels permuted
  among the train records, the per-label counts preserved): the control that
  says whether the gain is label-driven.

The union is the arm, not the phrase map alone: the topical phrases are rare,
so a phrase map alone would inherit the hapax problem (CS-33's
distinctive-only map: 0.0798).

**The guard.** The pack's fit side is **cross-fit** over 5 folds, and the
cross-fit rides into the phrase map too: each fold's phrase map is built from
the fold's training texts, so the fit side never sees its own texts.

The rows land in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``corpus_map_phrase`` section — ``evaluate`` is a main-pipeline command, so
the table is command-derived, never a heuristic. The narrative goes to the
track log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments_text_processing/corpus_map_phrase_sweep.py
"""

from __future__ import annotations

from cluster_rag.sweep import EvalCell, define_sweep

# The arm: the union of the word map's score and the phrase map's own (both
# under the z-scored "evidence for" sum, CS-31's fit-side selection).
SCORING = "sum_zscore_positive_phrase"

# The fit side's selection metric is measured at the same operating point.
FIT_SIDE_K = 2

cells = [
    EvalCell(
        label=f"upr-eval-corpus-map-{SCORING.replace('_', '-')}{'-blind' if blind else ''}-val",
        predictor="corpus_map",
        params={
            "scoring": SCORING,
            "label_blind": blind,
            "fit_side_k": FIT_SIDE_K,
            "folds": 5,
        },
        k=2,
        split="val",
    )
    for blind in (False, True)
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="corpus_map_phrase")
