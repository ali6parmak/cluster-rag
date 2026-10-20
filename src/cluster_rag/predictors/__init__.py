"""Predictor registry: build the configured predictor (Phase 3 port; Phase 4+ members).

``build_predictor`` resolves the ``eval.predictor`` config section into a
:class:`~cluster_rag.predictors.base.Predictor` instance. Every predictor
takes the same constructor signature ``(context, seed, params)`` and declares
its supported parameter keys in ``ALLOWED_PARAMS``, so an unknown name or
parameter fails here — before any run starts. Embedding-dependent predictors
further declare ``REQUIRES_EMBEDDINGS``/``REQUIRED_KINDS``
(:func:`predictor_embedding_requirements` reads them for the evaluate
command's artifact wiring).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cluster_rag.predictors.base import Predictor, PredictorContext
from cluster_rag.predictors.fusion import FusionPredictor
from cluster_rag.predictors.knn import KNNVotingPredictor
from cluster_rag.predictors.maxsim import MaxSimPredictor
from cluster_rag.predictors.prototype import PrototypePredictor
from cluster_rag.predictors.trivial import MostFrequentPredictor, RandomPredictor

PREDICTORS: dict[str, type] = {
    "most_frequent": MostFrequentPredictor,
    "random": RandomPredictor,
    "knn": KNNVotingPredictor,
    "prototype": PrototypePredictor,
    "maxsim": MaxSimPredictor,
    "fusion": FusionPredictor,
}


def _registry_class(name: str) -> type:
    """Look up the predictor class for ``name`` (shared error contract)."""
    predictor_class = PREDICTORS.get(name)
    if predictor_class is None:
        raise ValueError(f"Unknown predictor {name!r}; expected one of {sorted(PREDICTORS)}")
    return predictor_class


def predictor_embedding_requirements(name: str) -> tuple[bool, tuple[str, ...]]:
    """The embedding needs the named predictor declares (evaluate-side wiring).

    Returns:
        ``(requires, kinds)`` — ``requires`` is the class's
        ``REQUIRES_EMBEDDINGS`` flag (default False for predictors that never
        declared it), ``kinds`` its ``REQUIRED_KINDS`` (document/query/label).

    Raises:
        ValueError: For an unknown predictor name (same message as
            :func:`build_predictor`).
    """
    predictor_class = _registry_class(name)
    return (
        bool(getattr(predictor_class, "REQUIRES_EMBEDDINGS", False)),
        tuple(getattr(predictor_class, "REQUIRED_KINDS", ())),
    )


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
