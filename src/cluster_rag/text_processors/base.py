"""Ports of the text-processing track (TEXT_PROCESSING_PLAN.md section 2).

The track studies the least-explored layer of the pipeline: how a text is
chunked, segmented and structured before it is embedded. Three protocols
carry the work:

- :class:`Chunker` — splits texts into chunks (pure, deterministic). The
  chunker's ``version`` plugs the chunk *inputs* into the **existing**
  embedding-cache version slot, so the effective cache key stays
  ``(model, prefix, version, sha1(chunk))`` and the warm cache of the 14
  pool models stays valid (no new cache-key component).
- :class:`FeatureExtractor` — turns a (paragraph, question) pair into a row of
  floats. Pure and pair-local: no corpus-level state, so no feature can
  silently encode a corpus statistic (the track's leakage rules: no
  question-level priors, no document identity).
- :class:`Combiner` — fits a scorer over feature rows and scores new rows.
  The split handling stays with the caller: a combiner never learns which
  document (or which split) a row came from.

Pure core (AGENTS.md dependency direction): this package may import
``cluster_rag.embed.base`` (a ``typing.Protocol`` plus numpy) and nothing
heavier — no torch, sentence-transformers, httpx or sqlite3. Model clients
(embedders, Ollama LLMs) are wired in the CLI, never inside these modules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class TextPair:
    """One (paragraph, question) pair — the unit a feature extractor consumes.

    The pair carries text only: a feature extractor must never see a label,
    a document name or a corpus statistic, so that the track's leakage rules
    (TEXT_PROCESSING_PLAN.md Stage 3) hold by construction.
    """

    paragraph_text: str
    question_text: str


def config_version(name: str, payload: Mapping[str, Any] | None = None) -> str:
    """Cache-facing version string of a configuration.

    ``<name>-v1-<sha1(payload)[:12]>`` — a content hash of the parameters, so
    two configurations that differ in any parameter can never share cached
    chunks, while the same configuration always does.

    Raises:
        ValueError: If ``name`` is empty.
    """
    if not name:
        raise ValueError("config_version needs a non-empty name")
    body = json.dumps(payload or {}, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    return f"{name}-v1-{digest}"


@runtime_checkable
class Chunker(Protocol):
    """Splits texts into chunks (pure, deterministic, no I/O).

    ``segment`` takes a sequence of texts and returns one chunk list per text
    (same order, same length). ``version`` is the cache-facing string of the
    chunker's own parameters; it goes into the *existing* embedding-cache
    version slot, so the cache key layout never changes.
    """

    @property
    def version(self) -> str:
        """Cache-facing version of this chunker's parameters."""

    def cache_version(self, base_version: str) -> str:
        """The embedding-cache version slot for chunks of ``base_version`` texts.

        The chunking version is composed into the *existing* version slot, so
        the effective cache key stays ``(model, prefix, version, sha1(chunk))``:
        the cache layout never changes (an identity chunker returns
        ``base_version`` unchanged, which keeps the warm cache valid).
        """

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into its chunks."""


class BaseChunker:
    """Shared behaviour of the chunkers: version composition for the cache.

    Subclasses set ``name`` and implement :meth:`segment`; the version string
    is a hash of :meth:`version_payload` (the parameters that define the
    behaviour), so a changed parameter set can never reuse cached chunks.
    """

    name: ClassVar[str] = "base"

    def version_payload(self) -> dict[str, Any]:
        """The parameters that define this chunker's behaviour."""
        return {}

    @property
    def slug(self) -> str:
        """A short, parameter-aware handle of this chunker (e.g. ``"words15"``).

        The slug names the chunker inside an arm name and inside a table row.
        It defaults to the chunker's registry name, so a chunker without
        parameters keeps its plain name; a chunker whose parameters change the
        output (the word and sentence windows) overrides it.
        """
        return self.name

    @property
    def version(self) -> str:
        """Cache-facing version string (a hash of the parameters)."""
        return config_version(self.name, self.version_payload())

    def cache_version(self, base_version: str) -> str:
        """The embedding-cache version slot for chunks of ``base_version`` texts.

        The chunking version is composed into the *existing* version slot, so
        the effective cache key is ``(model, prefix, version, sha1(chunk))``:
        the layout does not change and the warm cache of the 14 pool models
        stays valid.

        Args:
            base_version: The version of the text transform applied before
                chunking (e.g. ``raw-v1`` for paragraphs, the question
                transform's version for questions).
        """
        return f"{base_version}+{self.version}"

    def segment(self, texts: Sequence[str]) -> list[list[str]]:
        """Split each text into chunks (implemented by the subclasses)."""
        raise NotImplementedError


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns (paragraph, question) pairs into numeric feature rows (pure).

    The extractor sees one pair at a time and keeps no corpus-level state.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the features, in column order."""

    def extract(self, pairs: Sequence[TextPair]) -> np.ndarray:
        """Feature matrix of shape ``(n_pairs, n_features)``, float32."""


@runtime_checkable
class Combiner(Protocol):
    """Fits a scorer over feature rows and scores new rows.

    The split handling stays with the caller: a combiner is handed the fit
    side's rows and labels and never learns which document a row came from
    (TEXT_PROCESSING_PLAN.md Stage 3 leakage rules).
    """

    def fit(self, features: np.ndarray, labels: Sequence[int]) -> Combiner:
        """Fit the scorer on the fit side's rows (returns the fitted combiner)."""

    def score(self, features: np.ndarray) -> np.ndarray:
        """One score per row (higher = more likely relevant)."""
