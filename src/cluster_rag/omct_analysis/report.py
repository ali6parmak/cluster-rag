"""The report: the artifact one run writes.

``build_analysis_report`` assembles everything the other modules measured
into the ``analysis.json`` the track's runs produce.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.omct_analysis.helpers import (
    BASELINE_ARM,
    SEMANTIC_CRITERION_AUC,
)
from cluster_rag.omct_analysis.view import (
    DatasetView,
    paragraph_length_report,
)
from cluster_rag.text_processors.control import SAMPLING_RULE_DESCRIPTIONS

logger = logging.getLogger(__name__)


def build_analysis_report(
    view: DatasetView,
    models: Sequence[dict[str, Any]],
    *,
    run_id: str,
    source: str,
    source_sha256: str,
    seed: int,
    control_size: int,
    sampling_rule: str,
    question_transform: str,
    chunker_name: str,
    chunker_params: Mapping[str, Any] | None = None,
    question_chunker_name: str | None = None,
    question_chunker_params: Mapping[str, Any] | None = None,
    semantic_thresholds: Sequence[float] = (),
    semantic_selection: str = SEMANTIC_CRITERION_AUC,
    aggregations: Sequence[str] = (),
    question_transforms: Sequence[str] = (),
    bootstrap: Mapping[str, Any] | None = None,
    stage3: Mapping[str, Any] | None = None,
    baseline_chunker_name: str = BASELINE_ARM,
    n_valid_pairs: int,
    n_malformed_pairs: int,
    total_seconds: float,
) -> dict[str, Any]:
    """Assemble the Stage-0 artifact (the JSON document of one run)."""
    excluded = [
        {
            "cell_id": cell["cell_id"],
            "pair_id": cell["pair_id"],
            "document": cell["document"],
            "label": cell["label"],
        }
        for cell in view.judged_cells
    ]
    return {
        "run_id": run_id,
        "command": "analyze-omct",
        "source": source,
        "source_sha256": source_sha256,
        "n_valid_pairs": n_valid_pairs,
        "n_malformed_pairs": n_malformed_pairs,
        "seed": seed,
        "analysis": {
            "control": {
                **view.control.describe(),
                "n_judged_cells_excluded": len(excluded),
                "excluded_judged_cells": excluded,
                "numpy_version": np.__version__,
            },
            "grid": {
                "n_documents": view.pooled["n_documents"],
                "n_paragraphs": view.pooled["n_paragraphs"],
                "n_questions": view.pooled["n_questions"],
                "n_cells": view.pooled["grid_cells"],
                "n_judged_cells": view.pooled["judged_cells"],
                "n_unjudged_cells": view.pooled["unjudged_cells"],
                "coverage": view.pooled["grid_coverage"],
            },
            "strata": list(view.strata),
            "pooled": view.pooled,
            "documents": list(view.documents),
            "sampling_rule": sampling_rule,
            "sampling_rule_description": SAMPLING_RULE_DESCRIPTIONS.get(
                sampling_rule, sampling_rule
            ),
            "question_transform": question_transform,
            "question_transforms": list(question_transforms),
            "chunker": chunker_name,
            "chunker_params": dict(chunker_params or {}),
            "question_chunker": question_chunker_name,
            "question_chunker_params": dict(question_chunker_params or {}),
            "semantic_thresholds": list(semantic_thresholds),
            "semantic_selection": semantic_selection,
            "baseline_chunker": baseline_chunker_name,
            "aggregations": list(aggregations),
            "bootstrap": dict(bootstrap or {}),
            "stage3": None if stage3 is None else dict(stage3),
            "paragraph_lengths": paragraph_length_report(view),
            "notes": [
                "OMCT is auxiliary data (PLAN.md section 2): these numbers annotate "
                "method choice, they are never a UPR metric.",
                "Subset A (the reviewer-verified hard negatives) and subset B (the "
                "unjudged control) are never pooled: every number names its subset.",
                "The control is 'unjudged' (presumed-negative only where a label-shaped "
                "number is unavoidable); its positive rate is unmeasured.",
                "rank_percentile is the position from the top of the paragraph's own "
                "same-document candidate list (0 = best candidate, 1 = worst; a random "
                "candidate sits at about 0.5).",
                "The model's top-level fields are the baseline arm (full text against "
                "full text). The other arms are under 'arms' and their comparisons "
                "against the baseline under 'comparisons' (one per arm).",
                "A swept parameter (the semantic threshold) is chosen on the fit "
                "side only: the model's 'threshold_selection' block records the "
                "grid, the per-rotation fit-side numbers and the chosen value, so "
                "no document ever sees its own fitted parameter.",
                "The paragraph lengths are recorded in 'paragraph_lengths' (the "
                "length-vs-shift question); they are never a feature, because "
                "length is a document proxy on this data.",
                "A learned arm (the Stage-3 block) scores probabilities and the "
                "baseline arm scores cosines: the level numbers of the two arms "
                "are in their own units and are not comparable, while the rank "
                "numbers (the rank percentiles) are.",
            ],
        },
        "models": list(models),
        "total_seconds": round(total_seconds, 3),
    }
