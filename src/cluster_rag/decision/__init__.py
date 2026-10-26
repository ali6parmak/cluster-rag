"""The decision models at the edge (the text-processing track's Stage 3.5).

Ollama's **decision model** type: a *state* (the text to judge) plus 1-64
named, typed *questions* (``noul`` / ``choice`` / ``score``), answered with
**probabilities** in one forward pass — no decoder, no generation, no
relabelling.

Three modules:

- :mod:`cluster_rag.decision.base` — the :class:`~cluster_rag.decision.base.Decider`
  port and the answer shapes (no model client, no I/O);
- :mod:`cluster_rag.decision.systemone` — the client for Ollama's
  ``POST /v1/systemone`` endpoint (which is in neither the Ollama CLI nor its
  Python/JS libraries, so it is a raw HTTP call);
- :mod:`cluster_rag.decision.cache` — the decision models' **own** disk cache
  (keyed by the model id, the digest, the prompt version, the state and the
  question specs), which never mixes with the embedding cache.

The pure side of the track (the question specs, the call plan, the
competence control, the measurement) lives in
:mod:`cluster_rag.text_processors.decision`.
"""

from cluster_rag.decision.base import (
    Decider,
    DecisionAnswer,
    DecisionInfo,
    DecisionState,
    answer_probability,
    parse_answer,
)
from cluster_rag.decision.cache import CachedDecider, decision_cache_key, decision_cache_path
from cluster_rag.decision.systemone import SystemOneDecider

__all__ = [
    "CachedDecider",
    "DecisionAnswer",
    "DecisionInfo",
    "DecisionState",
    "Decider",
    "SystemOneDecider",
    "answer_probability",
    "decision_cache_key",
    "decision_cache_path",
    "parse_answer",
]
