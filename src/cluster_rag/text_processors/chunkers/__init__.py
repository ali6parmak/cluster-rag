"""The track's chunkers: one file per method, one class per file.

``CHUNKERS`` maps a config-facing name to the chunker class, so the CLI can
build a chunker from a config section without importing the modules by hand.
The registry grows with the track's methods (word/sentence windows, semantic
boundaries, question parts — TEXT_PROCESSING_PLAN.md section 2).

``build_chunker(name, params)`` builds a chunker *with its parameters*: a
chunker that takes none (``full_text``) refuses any parameter, and a chunker
that takes some (``atoms``) is configured from the config-facing mapping
(``{"split_clauses": false}``). The parameters are part of the chunker's
cache-facing version, so two differently configured chunkers can never share
cached chunks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cluster_rag.text_processors.base import BaseChunker
from cluster_rag.text_processors.chunkers.atoms import (
    DEFAULT_ABBREVIATIONS,
    AtomChunker,
    AtomChunkerConfig,
    split_clauses,
    split_sentences,
)
from cluster_rag.text_processors.chunkers.full_text import FullTextChunker

CHUNKERS: dict[str, type[BaseChunker]] = {
    "full_text": FullTextChunker,
    "atoms": AtomChunker,
}

# The config class of each chunker (``None`` = the chunker takes no
# parameters). The registry is the single place that knows them.
CHUNKER_CONFIGS: dict[str, type[Any] | None] = {
    "full_text": None,
    "atoms": AtomChunkerConfig,
}


def chunker_config_class(name: str) -> type[Any] | None:
    """The config class of a chunker (``None`` when it takes no parameters).

    Raises:
        ValueError: If the name is not in the registry.
    """
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunker {name!r}; expected one of {sorted(CHUNKERS)}")
    return CHUNKER_CONFIGS[name]


def build_chunker(name: str, params: Mapping[str, Any] | None = None) -> BaseChunker:
    """Build the chunker of a config-facing name, with its parameters.

    Args:
        name: The registry key of the chunker (e.g. ``"atoms"``).
        params: The chunker's parameters (e.g. ``{"split_clauses": false}``).

    Returns:
        The configured chunker.

    Raises:
        ValueError: If the name is unknown, the chunker takes no parameters
            but some were given, or a parameter is not one of the chunker's.
    """
    chunker_config_class(name)  # raises for an unknown name
    chunker_class = CHUNKERS[name]
    config_class = CHUNKER_CONFIGS[name]
    payload = dict(params or {})
    if config_class is None:
        if payload:
            raise ValueError(f"The chunker {name!r} takes no parameters: {sorted(payload)}")
        return chunker_class()  # type: ignore[call-arg]
    known = {field.name for field in config_class.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {unknown} for the chunker {name!r}; "
            f"expected a subset of {sorted(known)}"
        )
    return chunker_class(config_class(**payload))  # type: ignore[call-arg]


__all__ = [
    "CHUNKERS",
    "CHUNKER_CONFIGS",
    "DEFAULT_ABBREVIATIONS",
    "AtomChunker",
    "AtomChunkerConfig",
    "BaseChunker",
    "FullTextChunker",
    "build_chunker",
    "chunker_config_class",
    "split_clauses",
    "split_sentences",
]
