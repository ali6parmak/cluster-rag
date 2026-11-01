"""The lexical machinery of the track's lexical methods round (CS-29 … CS-35).

The round's pure side: the scorers that turn the **corpus map** (CS-23 — the
train split's own word↔label association) into numbers, and the estimator
behind the map's measures (CS-31).

A **pure core** in the AGENTS.md sense: numpy and the standard library only —
no model client, no I/O, no scikit-learn (the track's one documented
scikit-learn exception is its ``combining/`` package, §2). The model clients
and the harness wiring live outside: the adapter in
``src/cluster_rag/predictors/`` and the CLI in ``src/cluster_rag/commands/``.

Note the import order: this package's ``__init__`` re-exports the
**estimator** only. The scorer (:mod:`corpus_map_score`) reads the map, and
the map reads the estimator — so re-exporting the scorer here would make the
package's import cycle.
"""

from __future__ import annotations

from cluster_rag.text_processors.lexical.lexicon import (
    ALPHA,
    PairMeasures,
    conditional_log_odds,
    g2_p_value,
    g2_statistic,
    pair_measures,
    prior_total,
)

__all__ = [
    "ALPHA",
    "PairMeasures",
    "conditional_log_odds",
    "g2_p_value",
    "g2_statistic",
    "pair_measures",
    "prior_total",
]
