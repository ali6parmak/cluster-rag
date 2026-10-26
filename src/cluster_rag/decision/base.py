"""The decision-model port (the text-processing track's Stage 3.5).

Ollama's new **decision model** type: a *state* (the text to judge — a string
or a JSON object) plus 1-64 named, typed *questions* (``noul`` = true/false,
``choice`` = pick from 2-26 options, ``score`` = a rubric), answered with
**probabilities** in one forward pass. There is no decoder, no generation and
no relabelling: a decision model only **scores**.

This module holds the port's data shapes (the :class:`Decider` protocol, the
:class:`DecisionAnswer` and the :class:`DecisionInfo`) — no model client, no
I/O. The client lives in :mod:`cluster_rag.decision.systemone` (the raw HTTP
caller for ``POST /v1/systemone``, an endpoint that is in neither the Ollama
CLI nor its Python/JS libraries) and the cache in
:mod:`cluster_rag.decision.cache`.

**Why a port at all** (AGENTS.md dependency direction): the pure side of the
track (the question specs, the arm's construction, the measurement) must be
testable without a server, so the pure core talks to a ``Decider`` protocol
and the CLI wires the real client in.

**The hygiene the port carries** (TEXT_PROCESSING_PLAN.md section 2): the model
is pinned by id **and digest**, the prompt version is recorded, every call is
cached (no re-judging on a re-run) and there is no temperature (there is no
sampling — the model answers with probabilities).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from cluster_rag.text_processors.decision.questions import QuestionSpec

# The decision models' question types (the API's own vocabulary).
QUESTION_TYPE_NOUL = "noul"
QUESTION_TYPE_CHOICE = "choice"
QUESTION_TYPE_SCORE = "score"

# The state of a decision call: a string or a JSON object (the API accepts
# both). The track uses a JSON object with the pair's parts as its keys.
DecisionState = str | Mapping[str, Any]


@dataclass(frozen=True)
class DecisionAnswer:
    """One question's answer: the value, the option probabilities and the confidence.

    - ``type`` is the question's type (``noul`` / ``choice`` / ``score``);
    - ``value`` is the ``noul`` probability for a boolean question, the **picked
      option** for a choice question and the rubric score for a score question;
    - ``probabilities`` holds the per-option probabilities (a choice question)
      or the per-level probabilities (a score question);
    - ``confidence`` is the model's own confidence — **not** the chance of
      being right, and never read as a probability;
    - ``raw`` is the server's own answer object, kept for the artifact.
    """

    name: str
    type: str
    value: float | str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """The answer as a JSON-serializable dict (the artifact and the cache)."""
        return {
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DecisionInfo:
    """The provenance of a decision model: the id, the pinned digest, the host.

    ``extra`` carries the server-side details (the model's size, its
    capabilities, the host the client reached) and is written into the run's
    artifact.
    """

    model_id: str
    digest: str
    backend: str = "ollama"
    host: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Decider(Protocol):
    """A decision model: a state plus a question spec -> the answers.

    ``decide`` takes **one** state and a mapping of question name -> question
    spec, so a caller can ask up to 64 questions about the same state in one
    call (the "paragraph as the state" shape of the prompt grid, which is the
    cheap one).
    """

    @property
    def info(self) -> DecisionInfo:
        """The provenance of the model (its id and its pinned digest)."""

    def decide(
        self, state: DecisionState, questions: Mapping[str, QuestionSpec]
    ) -> Mapping[str, DecisionAnswer]:
        """Answer the questions about the state; one answer per question name."""


def parse_answer(name: str, payload: Mapping[str, Any]) -> DecisionAnswer:
    """Parse one answer object of the ``/v1/systemone`` response.

    The three question types answer differently: a ``noul`` question carries the
    probability that the statement is true, a ``choice`` question carries the
    picked option and the per-option probabilities, and a ``score`` question
    carries the rubric score and the per-level probabilities.

    Raises:
        RuntimeError: If the answer object carries no usable probability.
    """
    kind = str(payload.get("type") or "")
    raw = dict(payload)
    if kind == QUESTION_TYPE_NOUL:
        value = payload.get("noul")
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"the decision answer {name!r} carries no 'noul' probability: {payload}"
            )
        return DecisionAnswer(
            name=name,
            type=kind,
            value=float(value),
            confidence=_optional_float(payload.get("confidence")),
            raw=raw,
        )
    if kind == QUESTION_TYPE_CHOICE:
        picked = payload.get("choice")
        probabilities = _probability_map(payload.get("probabilities"))
        if not probabilities:
            raise RuntimeError(
                f"the decision answer {name!r} carries no option probabilities: {payload}"
            )
        return DecisionAnswer(
            name=name,
            type=kind,
            value=str(picked) if picked is not None else None,
            probabilities=probabilities,
            confidence=_optional_float(payload.get("confidence")),
            raw=raw,
        )
    if kind == QUESTION_TYPE_SCORE:
        score = payload.get("score")
        probabilities = _probability_map(payload.get("probabilities"))
        return DecisionAnswer(
            name=name,
            type=kind,
            value=float(score) if isinstance(score, (int, float)) else None,
            probabilities=probabilities,
            confidence=_optional_float(payload.get("confidence")),
            raw=raw,
        )
    raise RuntimeError(f"unknown decision answer type {kind!r} for the answer {name!r}")


def answer_probability(answer: DecisionAnswer, positive_option: str | None = None) -> float:
    """The arm's score of one answer: the probability that the pair matches.

    - a ``noul`` question's score is its own probability (the probability that
      the statement is true);
    - a ``choice`` question's score is the probability of the **positive
      option** (the config names it; the picked option and the confidence are
      *not* used — the confidence is not the chance of being right).

    Raises:
        RuntimeError: If the positive option is missing from the answer.
    """
    if answer.type == QUESTION_TYPE_NOUL:
        return float(answer.value or 0.0)
    if answer.type == QUESTION_TYPE_CHOICE:
        if positive_option is None:
            raise RuntimeError(
                "a choice answer needs the positive option to be read as a probability"
            )
        if positive_option not in answer.probabilities:
            raise RuntimeError(
                f"the answer carries no probability for the positive option "
                f"{positive_option!r}: {sorted(answer.probabilities)}"
            )
        return float(answer.probabilities[positive_option])
    raise RuntimeError(f"cannot read a probability from a {answer.type!r} answer")


def _optional_float(value: Any) -> float | None:
    """A float or ``None`` (the server may omit the field)."""
    return float(value) if isinstance(value, (int, float)) else None


def _probability_map(value: Any) -> dict[str, float]:
    """The per-option probabilities of an answer (empty when absent)."""
    if not isinstance(value, Mapping):
        return {}
    probabilities: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)):
            probabilities[str(key)] = float(item)
    return probabilities
