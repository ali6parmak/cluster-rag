"""Predictor registry: build the configured predictor (Phase 3).

``build_predictor`` resolves the ``eval.predictor`` config section into a
:class:`~cluster_rag.predictors.base.Predictor` instance. Every predictor
takes the same constructor signature ``(context, seed, params)`` and declares
its supported parameter keys in ``ALLOWED_PARAMS``, so an unknown name or
parameter fails here — before any run starts. Phase 4 predictors register
themselves in :data:`PREDICTORS` the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cluster_rag.predictors.base import Predictor, PredictorContext
from cluster_rag.predictors.trivial import MostFrequentPredictor, RandomPredictor

PREDICTORS: dict[str, type] = {
    "most_frequent": MostFrequentPredictor,
    "random": RandomPredictor,
}


def build_predictor(
    name: str,
    params: Mapping[str, Any],
    context: PredictorContext,
    *,
    seed: int,
) -> Predictor:
    """Instantiate the named predictor with its config params.

    Args:
        name: Registry key (``eval.predictor.name``).
        params: Predictor-specific parameters (``eval.predictor.params``);
            each class validates its own keys via ``ALLOWED_PARAMS``.
        context: The label space plus train-only records.
        seed: Top-level config seed (baselines are deterministic under it).

    Raises:
        ValueError: For an unknown predictor name or unsupported parameter.
    """
    predictor_class = PREDICTORS.get(name)
    if predictor_class is None:
        raise ValueError(f"Unknown predictor {name!r}; expected one of {sorted(PREDICTORS)}")
    predictor: Predictor = predictor_class(context, seed, dict(params))
    return predictor
