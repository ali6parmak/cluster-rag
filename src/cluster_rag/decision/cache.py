"""The decision models' disk cache (a **separate** cache, never the embedding one).

A decision answer is not a vector, so the decision cache never mixes with the
embedding cache (TEXT_PROCESSING_PLAN.md section 2). It has its own root under
``cache/`` (``cache/decisions/``) and its own key:

    (model id, digest, prompt version, the state, the question specs)

The prompt version is part of the key, so a re-prompted arm never reads the old
arm's answers. The digest is part of the key, so a re-pulled model never reads
the old build's answers. The state and the question specs are hashed together,
because a decision model's answer depends on the whole call (a batched call
answers 64 questions about one state, and the batch is part of the question).

The track's LLM hygiene: the calls are cached, so a re-run is reproducible
modulo model drift and never re-judges a pair.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cluster_rag.decision.base import (
    DecisionAnswer,
    DecisionInfo,
    DecisionState,
    answer_probability,
)
from cluster_rag.text_processors.decision.questions import QuestionSpec

logger = logging.getLogger(__name__)

# The cache's own format version (its layout, not the embedding cache's).
CACHE_FORMAT_VERSION = "v1"

# The directory name of the decision cache under the run's cache root.
DECISION_CACHE_DIRNAME = "decisions"

_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _slug(value: str) -> str:
    """Readable path component (``[A-Za-z0-9._-]``); empty collapses to "-"."""
    slug = "".join(character if character in _SLUG_SAFE else "_" for character in value)
    return slug or "-"


def _model_dir(model_id: str) -> str:
    """The model's cache directory: a readable slug plus a uniqueness prefix."""
    digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{digest}-{_slug(model_id)}"


def decision_cache_key(
    *,
    model_id: str,
    digest: str,
    prompt_version: str,
    state: DecisionState,
    questions: Mapping[str, QuestionSpec],
) -> str:
    """The cache key of one call: the model's identity, the prompt and the call itself.

    The key is the canonical JSON of the tuple, hashed — so the same call
    always lands on the same file and a different call never does.
    """
    payload = {
        "model_id": model_id,
        "digest": digest,
        "prompt_version": prompt_version,
        "state": state,
        "questions": {name: spec.to_payload() for name, spec in sorted(questions.items())},
    }
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


def decision_cache_path(
    cache_root: Path | str,
    *,
    model_id: str,
    prompt_version: str,
    key: str,
    cache_format_version: str = CACHE_FORMAT_VERSION,
) -> Path:
    """The cache file of one call."""
    return (
        Path(cache_root)
        .joinpath(
            DECISION_CACHE_DIRNAME,
            cache_format_version,
            _model_dir(model_id),
            _slug(prompt_version),
            key,
        )
        .with_suffix(".json")
    )


def _store(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one answer (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class CachedDecider:
    """Decider decorator that serves answers from disk and delegates the misses.

    The wrapped decider's provenance is passed through, so the artifact records
    the real model's id and digest even when the answers come from the cache.
    """

    def __init__(
        self,
        decider: Any,
        cache_root: Path | str,
        *,
        prompt_version: str,
        cache_format_version: str = CACHE_FORMAT_VERSION,
    ) -> None:
        """Wrap ``decider``; the prompt version is the run's (config-driven)."""
        self._decider = decider
        self._root = Path(cache_root)
        self._prompt_version = prompt_version
        self._cache_format_version = cache_format_version
        self.cache_hits = 0
        self.cache_misses = 0
        self.n_stored = 0

    @property
    def info(self) -> DecisionInfo:
        """The wrapped decider's provenance plus the cache's location."""
        info = self._decider.info
        extra = {
            **info.extra,
            "cache_root": str(self._root),
            "cache_format_version": self._cache_format_version,
            "prompt_version": self._prompt_version,
        }
        return DecisionInfo(
            model_id=info.model_id,
            digest=info.digest,
            backend=info.backend,
            host=info.host,
            extra=extra,
        )

    @property
    def usage(self) -> dict[str, int]:
        """The wrapped decider's token counters (the calls that actually ran)."""
        return dict(getattr(self._decider, "usage", {}))

    def decide(
        self, state: DecisionState, questions: Mapping[str, QuestionSpec]
    ) -> Mapping[str, DecisionAnswer]:
        """Answer the questions about the state: the cache first, the model on a miss."""
        if not questions:
            raise ValueError("a decision call must carry at least one question")
        info = self._decider.info
        key = decision_cache_key(
            model_id=info.model_id,
            digest=info.digest,
            prompt_version=self._prompt_version,
            state=state,
            questions=questions,
        )
        path = decision_cache_path(
            self._root,
            model_id=info.model_id,
            prompt_version=self._prompt_version,
            key=key,
            cache_format_version=self._cache_format_version,
        )
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return {
                name: _answer_from_json(name, entry) for name, entry in payload["answers"].items()
            }
        self.cache_misses += 1
        answers = self._decider.decide(state, questions)
        _store(
            path,
            {
                "key": {
                    "model_id": info.model_id,
                    "digest": info.digest,
                    "prompt_version": self._prompt_version,
                    "key": key,
                },
                "answers": {name: answer.to_json() for name, answer in answers.items()},
            },
        )
        self.n_stored += 1
        return answers


def _answer_from_json(name: str, entry: Mapping[str, Any]) -> DecisionAnswer:
    """Rebuild an answer from its cached JSON."""
    return DecisionAnswer(
        name=str(entry.get("name") or name),
        type=str(entry.get("type") or ""),
        value=entry.get("value"),
        probabilities={
            str(key): float(value) for key, value in (entry.get("probabilities") or {}).items()
        },
        confidence=(
            float(entry["confidence"])
            if isinstance(entry.get("confidence"), (int, float))
            else None
        ),
    )


def cached_probability(answer: DecisionAnswer, positive_option: str | None = None) -> float:
    """The probability of one cached answer (the arm's score).

    A thin re-export of :func:`cluster_rag.decision.base.answer_probability`, so
    the cache's module is the only import a caller needs for the score.
    """
    return answer_probability(answer, positive_option)
