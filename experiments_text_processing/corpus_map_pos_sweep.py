"""The **grammatical classes** as a classifier (the track's CS-34): the
POS-restricted maps.

CS-33's readout answered the reviewer's part-of-speech question from the corpus
map's own **frequency** classes; CS-34 answers it on the **grammatical** axis,
with a real tagger (``nltk``'s ``averaged_perceptron_tagger_eng``, at the edge
in ``src/cluster_rag/pos/``). The pre-registered payoff is measured here: a
**POS-restricted map** as a classifier.

Two restrictions, through the **existing** scorer and the **existing**
``evaluate`` command, on the same texts at the same operating point as the full
map — so the comparison against it is paired (``tools/lexical_compare.py``):

- **noun** — the map with the **nouns only** (the topic words);
- **verb** — the map with the **verbs only** (the predicate words: this
  genre's register).

Both use the same arm as CS-31's fit-side selection (``sum_zscore_positive``),
so the **only** difference from the full map's run is the restriction itself.
The tagger is switched on by the cell's ``pos:`` section (off by default
everywhere else, so every earlier run's artifact keeps its exact content).

The rows land in the **main** UPR table (``docs/experiments_table_upr.md``) in
the ``corpus_map_pos`` section — ``evaluate`` is a main-pipeline command, so
the table is command-derived, never a heuristic. The narrative goes to the
track log (``docs/experiments_text_processing.md``).

    uv run cluster-rag sweep --file experiments_text_processing/corpus_map_pos_sweep.py
"""

from __future__ import annotations

from cluster_rag.config import PosConfig
from cluster_rag.sweep import EvalCell, define_sweep

# The arm: CS-31's fit-side selection, unchanged. The restriction is the only
# difference between these two runs and the full map's run.
SCORING = "sum_zscore_positive"

# The two restrictions, and the run id's short name for each. The classes kept
# are the ones that still contribute to a cell's score; the dropped ones are as
# if they were not in the tables at all.
RESTRICTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("noun", ("noun",)),
    ("verb", ("verb",)),
)

# The fit side's selection metric is measured at the same operating point.
FIT_SIDE_K = 2

# The tagger (the track's CS-34): the edge client, the model id and the
# language code are the config's, never the code's.
POS = PosConfig(enabled=True)

cells = [
    EvalCell(
        label=f"upr-eval-corpus-map-{SCORING.replace('_', '-')}-{short}-val",
        predictor="corpus_map",
        params={
            "scoring": SCORING,
            "restrict_pos": list(classes),
            "fit_side_k": FIT_SIDE_K,
            "folds": 5,
        },
        k=2,
        split="val",
        pos=POS,
    )
    for short, classes in RESTRICTIONS
]

# One sweep = one dataset ("upr": the label prefix and the allowed commands
# are enforced); `label` names the sweep's table section.
SWEEP = define_sweep(cells, dataset="upr", label="corpus_map_pos")
