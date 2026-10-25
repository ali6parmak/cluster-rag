"""The track's combiners: a learned scorer over the feature pack's rows.

``logistic`` is the Stage-3.1 baseline (a scikit-learn logistic regression
behind the ``Combiner`` port of ``text_processors.base``). The
gradient-boosted combiner (Stage 3.2) and the chunk-profile features
(Stage 3.3) come later; ``lightgbm`` is deferred by the plan until
scikit-learn's ``HistGradientBoostingClassifier`` has been beaten (a new
dependency needs the AGENTS.md section-3 approval).

``build_combiner(name, params)`` builds a combiner *with its parameters*, the
same way the chunker registry does: a combiner that takes none refuses any
parameter, and the parameters are validated against the combiner's own config
class, so a typo fails before a run starts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cluster_rag.text_processors.base import Combiner
from cluster_rag.text_processors.combining.logistic import (
    COMBINER_LOGISTIC,
    LogisticCombiner,
    LogisticCombinerConfig,
)

# The config-facing name of every combiner (the registry key).
COMBINERS: dict[str, type[LogisticCombiner]] = {
    COMBINER_LOGISTIC: LogisticCombiner,
}

# The config class of every combiner (``None`` = the combiner takes no
# parameters).
COMBINER_CONFIGS: dict[str, type[Any] | None] = {
    COMBINER_LOGISTIC: LogisticCombinerConfig,
}


def combiner_config_class(name: str) -> type[Any] | None:
    """The config class of a combiner (``None`` when it takes no parameters).

    Raises:
        ValueError: If the name is not in the registry.
    """
    if name not in COMBINERS:
        raise ValueError(f"Unknown combiner {name!r}; expected one of {sorted(COMBINERS)}")
    return COMBINER_CONFIGS[name]


def build_combiner(name: str, params: Mapping[str, Any] | None = None) -> Combiner:
    """Build the combiner of a config-facing name, with its parameters.

    Args:
        name: The registry key (e.g. ``"logistic"``).
        params: The combiner's parameters (e.g. ``{"C": 1.0}``).

    Returns:
        The configured combiner.

    Raises:
        ValueError: If the name is unknown, the combiner takes no parameters
            but some were given, or a parameter is not one of the combiner's.
    """
    config_class = combiner_config_class(name)
    combiner_class = COMBINERS[name]
    payload = dict(params or {})
    if config_class is None:
        if payload:
            raise ValueError(f"The combiner {name!r} takes no parameters: {sorted(payload)}")
        return combiner_class()  # type: ignore[call-arg]
    known = {field.name for field in config_class.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} for the combiner {name!r}; "
            f"expected a subset of {sorted(known)}"
        )
    return combiner_class(config_class(**payload))  # type: ignore[arg-type]


__all__ = [
    "COMBINERS",
    "COMBINER_CONFIGS",
    "COMBINER_LOGISTIC",
    "LogisticCombiner",
    "LogisticCombinerConfig",
    "build_combiner",
    "combiner_config_class",
]
