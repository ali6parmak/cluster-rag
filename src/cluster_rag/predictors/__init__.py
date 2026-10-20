"""Predictor registry: build the configured predictor (Phase 3 port; Phase 4+ members).

``build_predictor`` resolves the ``eval.predictor`` config section into a
:class:`~cluster_rag.predictors.base.Predictor` instance. Every predictor
takes the same constructor signature ``(context, seed, params)`` and declares
its supported parameter keys in ``ALLOWED_PARAMS``, so an unknown name or
parameter fails here — before any run starts. Embedding-dependent predictors
further declare ``REQUIRES_EMBEDDINGS``/``REQUIRED_KINDS``
(:func:`predictor_requirements` reads them, params included, for the evaluate
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
from cluster_rag.predictors.zeroshot import ZeroShotPredictor

PREDICTORS: dict[str, type] = {
    "most_frequent": MostFrequentPredictor,
    "random": RandomPredictor,
    "knn": KNNVotingPredictor,
    "prototype": PrototypePredictor,
    "maxsim": MaxSimPredictor,
    "fusion": FusionPredictor,
    "zeroshot": ZeroShotPredictor,
}


def _registry_class(name: str) -> type:
    """Look up the predictor class for ``name`` (shared error contract)."""
    predictor_class = PREDICTORS.get(name)
    if predictor_class is None:
        raise ValueError(f"Unknown predictor {name!r}; expected one of {sorted(PREDICTORS)}")
    return predictor_class


def predictor_requirements(
    name: str, params: Mapping[str, Any] | None = None
) -> tuple[bool, tuple[str, ...], bool]:
    """The embedding needs the named predictor declares (evaluate-side wiring).

    Args:
        name: Registry key (``eval.predictor.name``).
        params: The configured predictor params (``eval.predictor.params``) —
            some predictors declare *conditional* needs from them
            (:meth:`FusionPredictor.needs_label_artifact` wants label vectors
            only when ``c > 0``).

    Returns:
        ``(requires_embedding, kinds, needs_label_artifact)`` — the class's
        ``REQUIRES_EMBEDDINGS`` flag (default False) and ``REQUIRED_KINDS``
        (document/query/label), plus whether a prior ``upr-labelembed-*``
        artifact must be wired (class attr ``REQUIRES_LABEL_EMBEDDING``, or a
        params-aware ``needs_label_artifact(params)`` classmethod when defined).

    Raises:
        ValueError: For an unknown predictor name (same message as
            :func:`build_predictor`).
    """
    predictor_class = _registry_class(name)
    params = dict(params or {})
    label_hook = getattr(predictor_class, "needs_label_artifact", None)
    needs_labels = (
        bool(label_hook(params))
        if callable(label_hook)
        else bool(getattr(predictor_class, "REQUIRES_LABEL_EMBEDDING", False))
    )
    return (
        bool(getattr(predictor_class, "REQUIRES_EMBEDDINGS", False)),
        tuple(getattr(predictor_class, "REQUIRED_KINDS", ())),
        needs_labels,
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
