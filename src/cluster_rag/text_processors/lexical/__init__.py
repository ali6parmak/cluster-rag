"""The lexical machinery of the track's lexical methods round (CS-29 … CS-35).

The round's pure side: the scorers that turn the **corpus map** (CS-23 — the
train split's own word↔label association) into numbers, and the estimator
behind the map's measures (CS-31).

A **pure core** in the AGENTS.md sense: numpy and the standard library only —
no model client, no I/O, no scikit-learn (the track's one documented
scikit-learn exception is its ``combining/`` package, §2). The model clients
and the harness wiring live outside: the adapter in
``src/cluster_rag/predictors/`` and the CLI in ``src/cluster_rag/commands/``.
"""

from __future__ import annotations

from cluster_rag.text_processors.lexical.corpus_map_score import (
    FLOOR_GAP,
    SCORING_FUNCTIONS,
    cross_fit_scores,
    score_cells,
)

__all__ = [
    "FLOOR_GAP",
    "SCORING_FUNCTIONS",
    "cross_fit_scores",
    "score_cells",
]
