"""Tests for the decision models at the edge (the track's Stage 3.5).

Contracts under test:

- the ``Decider`` port's data shapes: the answer parsing of the three question
  types and the probability the arm's score is read from (the confidence is
  **not** the chance of being right);
- the ``/v1/systemone`` client: the request body, the digest check (a
  mismatch **fails the run**), the error translation and the token counters;
- the decision cache: its own root (never the embedding cache), its key (the
  model id, the digest, the prompt version, the state and the question specs)
  and its hit/miss counters.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cluster_rag.config import DecisionModelConfig
from cluster_rag.decision import (
    CachedDecider,
    DecisionAnswer,
    DecisionInfo,
    SystemOneDecider,
    answer_probability,
    decision_cache_key,
    decision_cache_path,
    parse_answer,
)
from cluster_rag.text_processors.decision.questions import (
    QuestionSpec,
    build_question_spec,
    render_instructions,
)

DIGEST = "a" * 64


def _model_config(**overrides) -> DecisionModelConfig:
    """A minimal, valid decision-model config."""
    payload = {
        "name": "fake-decider",
        "model_id": "fake:latest",
        "digest": DIGEST,
        "instructions": "Is the paragraph about the same topic as the question?",
    }
    payload.update(overrides)
    return DecisionModelConfig(**payload)


def _tags_handler(digest: str = DIGEST) -> httpx.MockTransport:
    """A ``/api/tags`` transport that reports one model with the given digest."""
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "models": [
                    {
                        "model": "fake:latest",
                        "digest": digest,
                        "size": 1234,
                        "capabilities": ["decision"],
                    }
                ]
            },
        )
    )


# -- the port's data shapes ------------------------------------------------


def test_parse_answer_reads_the_noul_probability() -> None:
    """A boolean question answers with the probability that it is true."""
    answer = parse_answer("same_topic", {"type": "noul", "noul": 0.75, "confidence": 0.6})
    assert answer.type == "noul"
    assert answer.value == 0.75
    assert answer_probability(answer) == 0.75


def test_parse_answer_reads_the_choice_probabilities() -> None:
    """A choice question answers with the picked option and the option probabilities."""
    answer = parse_answer(
        "same_topic",
        {
            "type": "choice",
            "choice": "yes",
            "probabilities": {"yes": 0.9, "no": 0.1},
            "confidence": 0.8,
        },
    )
    assert answer.value == "yes"
    assert answer.probabilities == {"yes": 0.9, "no": 0.1}
    assert answer_probability(answer, "yes") == 0.9
    assert answer_probability(answer, "no") == 0.1


def test_the_confidence_is_not_read_as_a_probability() -> None:
    """The confidence is the model's own confidence, never the score."""
    answer = parse_answer(
        "same_topic",
        {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.2, "no": 0.8}},
    )
    assert answer.confidence is None
    assert answer_probability(answer, "yes") == 0.2


def test_parse_answer_rejects_an_answer_without_a_probability() -> None:
    """An unusable answer fails loudly instead of scoring as 0."""
    with pytest.raises(RuntimeError):
        parse_answer("same_topic", {"type": "noul"})
    with pytest.raises(RuntimeError):
        parse_answer("same_topic", {"type": "bogus"})


def test_answer_probability_needs_the_positive_option() -> None:
    """A choice answer cannot be read without knowing the positive option."""
    answer = DecisionAnswer(name="q", type="choice", value="yes", probabilities={"yes": 0.4})
    with pytest.raises(RuntimeError):
        answer_probability(answer)
    with pytest.raises(RuntimeError):
        answer_probability(answer, "maybe")


# -- the client ------------------------------------------------------------


def test_the_client_refuses_a_digest_mismatch() -> None:
    """The pinned digest is checked against the server's before any call."""
    with pytest.raises(RuntimeError, match="digest"):
        SystemOneDecider(
            _model_config(),
            client=httpx.Client(
                base_url="http://localhost:11434", transport=_tags_handler("b" * 64)
            ),
        )


def test_the_client_sends_the_state_and_the_questions() -> None:
    """The request body is the documented one: the model, the state, the questions."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"model": "fake:latest", "digest": DIGEST}]}
            )
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": {"q": {"type": "noul", "noul": 0.5}},
                "usage": {"input_tokens": 12, "output_tokens": 0},
            },
        )

    transport = httpx.MockTransport(handler)
    client = SystemOneDecider(
        _model_config(), client=httpx.Client(base_url="http://localhost:11434", transport=transport)
    )
    spec = build_question_spec("noul", "Is it about the same topic?")
    answers = client.decide({"paragraph": "p", "question": "q"}, {"q": spec})

    assert seen[0]["model"] == "fake:latest"
    assert seen[0]["state"] == {"paragraph": "p", "question": "q"}
    assert seen[0]["questions"]["q"]["type"] == "noul"
    assert answers["q"].value == 0.5
    assert client.usage["n_calls"] == 1
    assert client.usage["input_tokens"] == 12


def test_the_client_translates_a_missing_model() -> None:
    """An unknown model gets the ``ollama pull`` hint."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": "not found"})

    with pytest.raises(RuntimeError, match="ollama pull"):
        SystemOneDecider(
            _model_config(),
            client=httpx.Client(
                base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
            ),
        )


def test_the_client_translates_a_rejected_request() -> None:
    """The server's own error message reaches the reader."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"model": "fake:latest", "digest": DIGEST}]}
            )
        return httpx.Response(200, json={"error": 'questions.q: "instructions" must be provided'})

    client = SystemOneDecider(
        _model_config(),
        client=httpx.Client(
            base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
        ),
    )
    with pytest.raises(RuntimeError, match="instructions"):
        client.decide({"paragraph": "p"}, {"q": build_question_spec("noul", "x")})


# -- the cache -------------------------------------------------------------


class _CountingDecider:
    """A decider that counts its calls and answers with a fixed probability."""

    def __init__(self) -> None:
        self.calls = 0
        self._info = DecisionInfo(model_id="fake:latest", digest=DIGEST, backend="fake")

    @property
    def info(self) -> DecisionInfo:
        """The decider's provenance."""
        return self._info

    def decide(self, state, questions):
        """Answer every question with 0.25."""
        self.calls += 1
        return {name: DecisionAnswer(name=name, type="noul", value=0.25) for name in questions}


def test_the_decision_cache_hits_on_the_second_call(tmp_path: Path) -> None:
    """A cached call never reaches the model again."""
    decider = _CountingDecider()
    cached = CachedDecider(decider, tmp_path, prompt_version="v1")
    spec = build_question_spec("noul", "Is it about the same topic?")
    state = {"paragraph": "p", "question": "q"}

    first = cached.decide(state, {"q": spec})
    second = cached.decide(state, {"q": spec})

    assert decider.calls == 1
    assert cached.cache_hits == 1
    assert cached.cache_misses == 1
    assert first["q"].value == second["q"].value == 0.25


def test_the_decision_cache_has_its_own_root(tmp_path: Path) -> None:
    """The decision cache never mixes with the embedding cache."""
    decider = _CountingDecider()
    cached = CachedDecider(decider, tmp_path, prompt_version="v1")
    spec = build_question_spec("noul", "Is it about the same topic?")
    cached.decide({"paragraph": "p", "question": "q"}, {"q": spec})

    files = list(tmp_path.rglob("*.json"))
    assert files, "the cache wrote nothing"
    assert all("decisions" in file.parts for file in files)


def test_the_cache_key_covers_the_prompt_version_and_the_digest() -> None:
    """A re-prompted or re-pulled model never reads the old answers."""
    spec = build_question_spec("noul", "Is it about the same topic?")
    base = decision_cache_key(
        model_id="fake:latest",
        digest=DIGEST,
        prompt_version="v1",
        state={"paragraph": "p", "question": "q"},
        questions={"q": spec},
    )
    other_prompt = decision_cache_key(
        model_id="fake:latest",
        digest=DIGEST,
        prompt_version="v2",
        state={"paragraph": "p", "question": "q"},
        questions={"q": spec},
    )
    other_digest = decision_cache_key(
        model_id="fake:latest",
        digest="b" * 64,
        prompt_version="v1",
        state={"paragraph": "p", "question": "q"},
        questions={"q": spec},
    )
    other_state = decision_cache_key(
        model_id="fake:latest",
        digest=DIGEST,
        prompt_version="v1",
        state={"paragraph": "other", "question": "q"},
        questions={"q": spec},
    )
    assert len({base, other_prompt, other_digest, other_state}) == 4


def test_the_cache_path_is_under_the_prompt_version(tmp_path: Path) -> None:
    """The path carries the prompt version, so a re-prompted arm is a new file."""
    path = decision_cache_path(tmp_path, model_id="fake:latest", prompt_version="v1", key="abc")
    assert path.parent.name == "v1"
    assert "decisions" in path.parts


# -- the pure question specs ----------------------------------------------


def test_render_instructions_puts_the_question_in_front_of_the_task() -> None:
    """The paragraph shape carries the question in the instructions."""
    rendered = render_instructions("Same topic?", "Why?", shape="paragraph")
    assert rendered == "Question: Why?\n\nSame topic?"


def test_render_instructions_leaves_the_task_alone_in_the_pair_shape() -> None:
    """The pair shape carries the question in the state, not in the instructions."""
    assert render_instructions("Same topic?", "Why?", shape="pair") == "Same topic?"


def test_render_instructions_rejects_a_task_that_repeats_the_question() -> None:
    """A task that repeats the question would put it into the call twice."""
    with pytest.raises(ValueError, match="placeholder"):
        render_instructions("Question: {question}", "Why?")
    with pytest.raises(ValueError, match="unknown decision state shape"):
        render_instructions("Same topic?", "Why?", shape="pairrot")


def test_a_choice_spec_carries_its_options_and_criteria() -> None:
    """The choice question's payload carries the options and their descriptions."""
    spec = build_question_spec(
        "choice",
        "Same topic?",
        options=("yes", "no"),
        criteria={"yes": "the same topic", "no": "a different topic"},
    )
    payload = spec.to_payload()
    assert payload["options"] == ["yes", "no"]
    assert payload["criteria"]["yes"] == "the same topic"


def test_a_noul_spec_takes_no_options() -> None:
    """A boolean question has no options."""
    with pytest.raises(ValueError):
        build_question_spec("noul", "Same topic?", options=("yes", "no"))
    assert QuestionSpec(type="noul", instructions="x").to_payload() == {
        "type": "noul",
        "instructions": "x",
    }
