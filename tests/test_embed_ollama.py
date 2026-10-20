"""Tests for the Ollama backend (Phase 2).

Default suite: pure unit tests over an ``httpx.MockTransport`` (batching,
payloads, digest/context discovery, actionable errors). The marked
integration tests hit a real `ollama serve` and are skipped by default.
"""

from __future__ import annotations

import json
import logging

import httpx
import numpy as np
import pytest

from cluster_rag.config import EmbedderConfig, PrefixesConfig
from cluster_rag.embed.ollama import OllamaEmbedder, normalize_ollama_host

MODEL_ID = "fake-emb"
TAG_ENTRY = {
    "model": "fake-emb:latest",
    "name": "fake-emb:latest",
    "digest": "aa" * 8,
    "details": {"parameter_size": "1B"},
}
SHOW_BODY = {"model_info": {"bert.embedding_length": 4, "bert.context_length": 512}}


def make_config(**overrides) -> EmbedderConfig:
    """Config for the fake model; explicit host so tests never read the env."""
    return EmbedderConfig(
        backend="ollama",
        model_id=MODEL_ID,
        host="http://mock:11434",
        batch_size=2,
        **overrides,
    )


def mock_client(routes: dict[tuple[str, str], object]) -> httpx.Client:
    """Client over MockTransport. Route values are dicts (200 JSON bodies),
    ``httpx.Response``/``Exception`` instances (returned/raised) or callables
    taking the request and returning a Response."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key not in routes:
            return httpx.Response(404, json={"error": "not found"})
        response = routes[key]
        if callable(response):
            return response(request)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(200, json=response)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock:11434")


def happy_routes() -> tuple[dict[tuple[str, str], object], list]:
    """Standard routes plus a request log, for asserting payloads/batching."""
    embed_requests: list[object] = []

    def embed_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        embed_requests.append(body)
        inputs = body["input"]
        vectors = [[1.0, 0.0, 0.0, 0.5] for _ in inputs]
        return httpx.Response(200, json={"model": body["model"], "embeddings": vectors})

    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): SHOW_BODY,
        ("POST", "/api/embed"): embed_handler,
    }
    return routes, embed_requests


def test_normalize_ollama_host() -> None:
    """Scheme-less hosts get http://; empty falls back to the documented default."""
    assert normalize_ollama_host("0.0.0.0:11434") == "http://0.0.0.0:11434"
    assert normalize_ollama_host(" http://h:1234/ ") == "http://h:1234"
    assert normalize_ollama_host("") == "http://localhost:11434"


def test_construction_records_digest_dimension_context() -> None:
    """Digest(es), parameter size, dimension and context come from the server."""
    routes, _ = happy_routes()
    embedder = OllamaEmbedder(make_config(), client=mock_client(routes))
    info = embedder.info
    assert info.backend == "ollama" and info.dimension == 4
    assert info.context_tokens == 512
    assert info.revision_or_digest == "aa" * 8
    assert info.extra["digests"] == ["aa" * 8]
    assert info.extra["parameter_size"] == "1B"


def test_construction_unreachable_server_mentions_ollama_serve() -> None:
    """Connection errors explain `ollama serve` (AGENTS.md error discipline)."""
    routes = {("GET", "/api/tags"): httpx.ConnectError("refused")}
    with pytest.raises(RuntimeError, match=r"ollama serve") as excinfo:
        OllamaEmbedder(make_config(), client=mock_client(routes))
    assert "http://mock:11434" in str(excinfo.value)


def test_construction_unknown_model_mentions_ollama_pull() -> None:
    """A model missing from /api/tags asks for `ollama pull`."""
    routes = {("GET", "/api/tags"): {"models": [{"model": "other:latest", "digest": "x"}]}}
    with pytest.raises(RuntimeError, match=r"ollama pull fake-emb"):
        OllamaEmbedder(make_config(), client=mock_client(routes))


def test_construction_missing_dimension_falls_back_to_config() -> None:
    """Without an exposed embedding_length the explicit config value is used."""
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): {"model_info": {"bert.context_length": 512}},
        ("POST", "/api/embed"): lambda request: httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0, 0.0, 0.5]] * len(json.loads(request.read())["input"])},
        ),
    }
    embedder = OllamaEmbedder(make_config(dimension=4), client=mock_client(routes))
    assert embedder.info.extra["dimension_source"] == "config_fallback"


def test_construction_dimension_override_trusts_config(caplog) -> None:
    """A set config dimension wins over a disagreeing /api/show (warn only).

    covers the embeddinggemma-2 case: the server metadata reports the
    pre-projection hidden size while the endpoint serves the projected
    vectors; the encode-time contract check still validates real output.
    """
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): {
            "model_info": {"bert.embedding_length": 512, "bert.context_length": 512}
        },
        ("POST", "/api/embed"): lambda request: httpx.Response(
            200,
            json={"embeddings": [[1.0] + [0.0] * 767] * len(json.loads(request.read())["input"])},
        ),
    }
    with caplog.at_level(logging.WARNING, logger="cluster_rag.embed.ollama"):
        embedder = OllamaEmbedder(make_config(dimension=768), client=mock_client(routes))
    assert "trusting the config" in caplog.text
    assert embedder.info.dimension == 768
    assert embedder.info.extra["dimension_source"] == "config_override"
    assert embedder.info.extra["api_embedding_length"] == 512
    out = embedder.encode(["one"], "query")
    assert out.shape == (1, 768) and out.dtype == np.float32


def test_encode_rejects_vectors_mismatching_pinned_dimension() -> None:
    """A pinned dimension that mismatches the real output vectors fails hard."""
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): {
            "model_info": {"bert.embedding_length": 512, "bert.context_length": 512}
        },
        ("POST", "/api/embed"): lambda request: httpx.Response(
            200,
            json={"embeddings": [[0.0] * 4] * len(json.loads(request.read())["input"])},
        ),
    }
    embedder = OllamaEmbedder(make_config(dimension=768), client=mock_client(routes))
    with pytest.raises(RuntimeError, match="returned 4-dim embeddings"):
        embedder.encode(["one"], "query")


def test_construction_context_override_beats_server_value() -> None:
    """An explicit embed.context_tokens replaces the server-reported context."""
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): {
            "model_info": {"bert.embedding_length": 4, "bert.context_length": 262144}
        },
        ("POST", "/api/embed"): lambda request: httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0, 0.0, 0.5]] * len(json.loads(request.read())["input"])},
        ),
    }
    embedder = OllamaEmbedder(make_config(context_tokens=8192), client=mock_client(routes))
    assert embedder.info.context_tokens == 8192
    assert embedder.info.extra["discovered_context_tokens"] == 262144
    assert embedder.info.extra["configured_context_tokens"] == 8192


def test_encode_batches_and_applies_prefix() -> None:
    """Requests are batched (batch_size=2 over 3 texts) and carry the prefix."""
    routes, embed_requests = happy_routes()
    config = make_config(prefixes=PrefixesConfig(query="query: "))
    embedder = OllamaEmbedder(config, client=mock_client(routes))
    out = embedder.encode(["one", "two", "three"], "query")

    assert out.shape == (3, 4) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert all(request["model"] == MODEL_ID for request in embed_requests)
    assert [len(request["input"]) for request in embed_requests] == [2, 1]
    assert embed_requests[0]["input"] == ["query: one", "query: two"]


def test_encode_empty_list_returns_empty_matrix() -> None:
    """An empty batch returns the valid (0, d) contract shape."""
    routes, _ = happy_routes()
    embedder = OllamaEmbedder(make_config(dimension=4), client=mock_client(routes))
    out = embedder.encode([], "document")
    assert out.shape == (0, 4) and out.dtype == np.float32


def test_encode_rejects_unknown_kind_and_blank_texts() -> None:
    """Contract guards fire before any request is sent."""
    routes, _ = happy_routes()
    embedder = OllamaEmbedder(make_config(), client=mock_client(routes))
    with pytest.raises(ValueError, match="kind"):
        embedder.encode(["a"], "search")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blank"):
        embedder.encode(["a", "  "], "query")


def test_encode_detects_wrong_embedding_count() -> None:
    """A mismatched /api/embed response fails instead of silently misaligning."""
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): SHOW_BODY,
        ("POST", "/api/embed"): httpx.Response(200, json={"embeddings": [[1.0, 0, 0, 0]]}),
    }
    embedder = OllamaEmbedder(make_config(), client=mock_client(routes))
    with pytest.raises(RuntimeError, match="embeddings for a batch"):
        embedder.encode(["one", "two"], "query")


def test_encode_surfaces_server_errors_with_body() -> None:
    """Non-404 HTTP errors include the status and a body snippet."""
    routes = {
        ("GET", "/api/tags"): {"models": [TAG_ENTRY]},
        ("POST", "/api/show"): SHOW_BODY,
        ("POST", "/api/embed"): httpx.Response(500, text="loader failure"),
    }
    embedder = OllamaEmbedder(make_config(), client=mock_client(routes))
    with pytest.raises(RuntimeError, match="HTTP 500.*loader failure"):
        embedder.encode(["one"], "query")


def test_host_resolution_precedence(tmp_path, monkeypatch) -> None:
    """Explicit config host wins; otherwise the OLLAMA_HOST env var applies."""
    routes, _ = happy_routes()

    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:9999")
    unconfigured = EmbedderConfig(backend="ollama", model_id=MODEL_ID, batch_size=2)
    embedder = OllamaEmbedder(unconfigured, client=mock_client({**routes}))
    assert embedder.info.extra["host"] == "http://0.0.0.0:9999"
    assert embedder.info.extra["host_from_env"] is True

    explicit = EmbedderConfig(
        backend="ollama", model_id=MODEL_ID, batch_size=2, host="http://config-server:1234"
    )
    embedder = OllamaEmbedder(explicit, client=mock_client({**routes}))
    assert embedder.info.extra["host"] == "http://config-server:1234"
    assert embedder.info.extra["host_from_env"] is False

    # unset config and env: the documented default
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    default = EmbedderConfig(backend="ollama", model_id=MODEL_ID, batch_size=2)
    embedder = OllamaEmbedder(default, client=mock_client({**routes}))
    assert embedder.info.extra["host"] == "http://localhost:11434"


@pytest.mark.integration
def test_ollama_real_server_smoke() -> None:
    """Real `ollama serve` + real small model: batched encode obey the contract."""
    config = EmbedderConfig(backend="ollama", model_id="granite-embedding", batch_size=4)
    embedder = OllamaEmbedder(config)
    out = embedder.encode(["forced displacement rose", "a ratification note"], "document")
    assert out.dtype == np.float32 and out.shape[0] == 2
    assert out.shape[1] == embedder.info.dimension
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert embedder.info.revision_or_digest  # a digest was recorded


# --- Phase 4 pool additions (real-server wiring checks) ----------------------

RAW_TEXT = "Ensure that all children have access to free education."


@pytest.mark.integration
def test_ollama_embeddinggemma2_wiring() -> None:
    """embeddinggemma-2 serves 768-d despite /api/show reporting 512 (config pins it);

    template prefixes apply exactly once; the configured 8192 context overrides the
    server's 256K loader context in recorded metadata.
    """
    config = EmbedderConfig(
        backend="ollama",
        model_id="embeddinggemma-2:latest",
        batch_size=4,
        dimension=768,
        context_tokens=8192,
        prefixes=PrefixesConfig(
            query="task: search result | query: {content}",
            document="title: none | text: {content}",
        ),
    )
    embedder = OllamaEmbedder(config)
    assert embedder.info.dimension == 768
    assert embedder.info.context_tokens == 8192
    assert embedder.info.extra["api_embedding_length"] == 512
    assert embedder.info.extra["dimension_source"] == "config_override"
    prefixed = embedder.encode([RAW_TEXT], "query")
    assert prefixed.shape == (1, 768)
    assert np.allclose(np.linalg.norm(prefixed, axis=1), 1.0, atol=1e-5)
    # exactly-once: the manually expanded template embedded prefix-free equals
    # the prefixed encode through the backend
    bare = OllamaEmbedder(
        EmbedderConfig(
            backend="ollama",
            model_id="embeddinggemma-2:latest",
            batch_size=4,
            dimension=768,
            context_tokens=8192,
        )
    )
    manual = bare.encode(["task: search result | query: " + RAW_TEXT], "query")[0]
    assert float(manual @ prefixed[0]) > 0.999999
    bare_raw = bare.encode([RAW_TEXT], "query")[0]
    assert abs(float(bare_raw @ prefixed[0])) < 0.999999  # the template is not a no-op


@pytest.mark.integration
def test_ollama_nomic_v2_moe_wiring() -> None:
    """nomic-embed-text-v2-moe: auto-discovered 768d/512 ctx; search prefixes exact-once."""
    config = EmbedderConfig(
        backend="ollama",
        model_id="nomic-embed-text-v2-moe:latest",
        batch_size=4,
        prefixes=PrefixesConfig(
            query="search_query: ", document="search_document: ", label="search_query: "
        ),
    )
    embedder = OllamaEmbedder(config)
    assert embedder.info.dimension == 768
    assert embedder.info.context_tokens == 512
    out = embedder.encode([RAW_TEXT], "query")
    assert out.shape == (1, 768) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    bare = OllamaEmbedder(
        EmbedderConfig(backend="ollama", model_id="nomic-embed-text-v2-moe:latest", batch_size=4)
    )
    manual = bare.encode(["search_query: " + RAW_TEXT], "query")[0]
    assert float(manual @ out[0]) > 0.999999
    bare_raw = bare.encode([RAW_TEXT], "query")[0]
    assert abs(float(bare_raw @ out[0])) < 0.999999
