"""The ``/v1/systemone`` client: Ollama's decision-model endpoint.

A **decision model** takes a *state* (the text to judge — a string or a JSON
object) plus 1-64 named, typed *questions* (``noul`` = true/false, ``choice`` =
pick from 2-26 options, ``score`` = a rubric) and answers with
**probabilities** in one forward pass: no decoder, no generation, no
relabelling. The endpoint is ``POST /v1/systemone`` with the body
``{"model": ..., "state": ..., "questions": {...}}`` and the answer
``{"answers": {...}, "usage": {...}}``.

**The endpoint is in neither the Ollama CLI nor its Python/JS libraries**, so
this is a raw HTTP call (``httpx``, already a dependency).

**The hygiene** (TEXT_PROCESSING_PLAN.md section 2): the model is pinned by id
**and digest** — the client reads the server's ``/api/tags`` and *fails the run*
when the digest differs from the pinned one — and there is no temperature
(there is no sampling: the model answers with probabilities).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import httpx

from cluster_rag.config import DecisionModelConfig
from cluster_rag.decision.base import (
    DecisionAnswer,
    DecisionInfo,
    DecisionState,
    parse_answer,
)
from cluster_rag.text_processors.decision.questions import QuestionSpec

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# The endpoint's path (not part of the Ollama CLI or its libraries).
SYSTEMONE_PATH = "/v1/systemone"


def normalize_ollama_host(host: str) -> str:
    """Accept hosts with or without a scheme (``0.0.0.0:11434`` -> ``http://...``)."""
    host = host.strip()
    if not host:
        return DEFAULT_OLLAMA_HOST
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def _resolve_host(config: DecisionModelConfig, host: str) -> str:
    """The host to call: the config's value, else ``OLLAMA_HOST``, else the default."""
    if host:
        return normalize_ollama_host(host)
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        return normalize_ollama_host(env_host)
    return DEFAULT_OLLAMA_HOST


class SystemOneDecider:
    """A decision model over Ollama's ``/v1/systemone`` endpoint."""

    def __init__(
        self,
        config: DecisionModelConfig,
        *,
        host: str = "",
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Connect, check the pinned digest and record the server-side model info.

        Args:
            config: The validated :class:`~cluster_rag.config.DecisionModelConfig`
                (the model id, the pinned digest and the prompt).
            host: The Ollama host (empty = the ``OLLAMA_HOST`` env var, else the
                default).
            timeout: The HTTP timeout in seconds (the first call pays the model
                load, so the default is generous).
            client: An optional pre-built ``httpx.Client`` (tests inject a
                ``MockTransport``-backed client here instead of a real server).

        Raises:
            RuntimeError: If the server is unreachable, the model is not
                installed, or the **pinned digest does not match** the server's.
        """
        self._config = config
        self._model_id = config.model_id
        self._host = _resolve_host(config, host)
        self._client = client or httpx.Client(base_url=self._host, timeout=timeout)
        self._timeout = timeout
        self._input_tokens = 0
        self._output_tokens = 0
        self._n_calls = 0
        digest, model_info = self._fetch_model_info()
        if digest != config.digest:
            raise RuntimeError(
                f"The digest of {self._model_id!r} at {self._host} is {digest!r}, but the "
                f"config pins {config.digest!r}. Refusing to run against a different "
                "model build — re-pin the digest (or pull the pinned one) and try again."
            )
        self._digest = digest
        self._info = DecisionInfo(
            model_id=self._model_id,
            digest=digest,
            backend="ollama",
            host=self._host,
            extra={
                "endpoint": SYSTEMONE_PATH,
                "timeout": timeout,
                "size": model_info.get("size"),
                "capabilities": model_info.get("capabilities"),
                "details": model_info.get("details"),
                "host_from_env": not host and "OLLAMA_HOST" in os.environ,
            },
        )
        logger.info(
            "Decision model ready: %s (digest %s, host %s)", self._model_id, digest, self._host
        )

    @property
    def info(self) -> DecisionInfo:
        """The provenance of the model (its id and its pinned digest)."""
        return self._info

    @property
    def usage(self) -> dict[str, int]:
        """The tokens the calls spent so far (the artifact's cost record)."""
        return {
            "n_calls": self._n_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }

    # -- request plumbing ------------------------------------------------

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        """Send JSON and return the parsed body, with actionable error translation.

        Raises:
            RuntimeError: Unreachable server (``ollama serve`` hint), unknown
                model (``ollama pull`` hint), a rejected request (the server's
                own message) or a non-JSON answer.
        """
        try:
            response = self._client.request(method, path, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._host} — is `ollama serve` running? "
                f"(also check `ollama pull {self._model_id}`) Original error: {exc}"
            ) from exc
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{self._model_id}' was not found at {self._host} — "
                f"run `ollama pull {self._model_id}` (installed models: `ollama list`)"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama {path} returned HTTP {response.status_code} for model "
                f"'{self._model_id}': {response.text[:300]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Ollama {path} returned a non-JSON response: {response.text[:300]}"
            ) from exc
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(
                f"Ollama {path} rejected the request for model '{self._model_id}': "
                f"{str(body['error'])[:300]}"
            )
        return body

    def _fetch_model_info(self) -> tuple[str, dict[str, Any]]:
        """The model's digest and its server-side info (from ``/api/tags``).

        Raises:
            RuntimeError: If the model is not installed on the server.
        """
        data = self._request("GET", "/api/tags", None)
        for entry in data.get("models", []):
            name = entry.get("model") or entry.get("name") or ""
            if name == self._model_id or (
                ":" not in self._model_id and name.rsplit(":", 1)[0] == self._model_id
            ):
                digest = entry.get("digest")
                if not digest:
                    raise RuntimeError(
                        f"Ollama reports no digest for {self._model_id!r} at {self._host}"
                    )
                return str(digest), entry
        raise RuntimeError(
            f"Ollama model '{self._model_id}' is not installed at {self._host} — "
            f"run `ollama pull {self._model_id}` (installed models: `ollama list`)"
        )

    # -- protocol --------------------------------------------------------

    def decide(
        self, state: DecisionState, questions: Mapping[str, QuestionSpec]
    ) -> Mapping[str, DecisionAnswer]:
        """Answer the questions about the state in one call.

        Raises:
            RuntimeError: If the call carries no question, more questions than
                the API accepts, or the answer carries no usable probability.
        """
        if not questions:
            raise ValueError("a decision call must carry at least one question")
        if len(questions) > 64:
            raise ValueError(f"a decision call accepts at most 64 questions, got {len(questions)}")
        payload = {
            "model": self._model_id,
            "state": state,
            "questions": {name: spec.to_payload() for name, spec in questions.items()},
        }
        body = self._request("POST", SYSTEMONE_PATH, payload)
        answers = body.get("answers")
        if not isinstance(answers, Mapping):
            raise RuntimeError(
                f"Ollama {SYSTEMONE_PATH} returned no answers for model "
                f"'{self._model_id}': {str(body)[:300]}"
            )
        usage = body.get("usage") or {}
        self._n_calls += 1
        if isinstance(usage, Mapping):
            self._input_tokens += int(usage.get("input_tokens") or 0)
            self._output_tokens += int(usage.get("output_tokens") or 0)
        return {name: parse_answer(name, answers[name]) for name in questions if name in answers}
