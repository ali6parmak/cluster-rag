"""The Phase 4 embedding pool: card-verified configs shared by sweep files.

`experiments/omct_probe_sweep.py` (OMCT probe) and `experiments/upr_embed_sweep.py`
(UPR document,query embeddings) import this table so both runs always use the
same model ids, prefixes, revisions and backend knobs — drift between the two
sweeps would break the "same pool on both datasets" comparison.

The `sweep` command's file loader puts the sweep file's directory on `sys.path`
while executing it, which is what makes this sibling import possible.

Sources (verified against the model cards at wiring time, 2026-10-20):
HF revisions are the local snapshot hashes (`~/.cache/huggingface/.../refs/main`);
Ollama digests are recorded into run metadata by the backend at run time.
`dimension` is set only where the backend cannot discover the true output size
(jina-v5's custom ST module) or reports the pre-projection size
(embeddinggemma-2: /api/show says 512, /api/embed serves 768) or where the
card context differs from the server-reported loader context
(embeddinggemma-2 context override 8192). The encode-time embedding contract
hard-fails whenever a pinned value mismatches the real vectors.
"""

from __future__ import annotations

from cluster_rag.config import EmbedderConfig, PrefixesConfig

NO_PREFIXES = PrefixesConfig()  # every kind: no prefix


def _hf(
    model_id: str,
    revision: str,
    prefixes: PrefixesConfig,
    batch_size: int = 64,
    *,
    trust_remote_code: bool = False,
    model_kwargs: dict[str, object] | None = None,
    dimension: int | None = None,
) -> EmbedderConfig:
    """Embedder config for a sentence-transformers backend (revision pinned)."""
    return EmbedderConfig(
        backend="hf",
        model_id=model_id,
        revision=revision,
        batch_size=batch_size,
        device=None,
        prefixes=prefixes,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs or {},
        dimension=dimension,
    )


def _ollama(
    model_id: str,
    prefixes: PrefixesConfig,
    *,
    dimension: int | None = None,
    context_tokens: int | None = None,
) -> EmbedderConfig:
    """Embedder config for an Ollama backend (local models only)."""
    return EmbedderConfig(
        backend="ollama",
        model_id=model_id,
        batch_size=64,
        prefixes=prefixes,
        dimension=dimension,
        context_tokens=context_tokens,
    )


MODELS: dict[str, EmbedderConfig] = {
    "minilm": _hf(
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        NO_PREFIXES,
    ),
    "e5-base-v2": _hf(
        "intfloat/e5-base-v2",
        "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        PrefixesConfig(query="query: ", document="passage: ", label="query: "),
    ),
    "bge-m3": _hf(
        "BAAI/bge-m3",
        "9a0624b896d81da7492a910ffa53731274b6cf3d",
        NO_PREFIXES,
        batch_size=32,
    ),
    "arctic2": _ollama(
        "snowflake-arctic-embed2:latest",
        PrefixesConfig(query="query: ", document="", label="query: "),
    ),
    "embeddinggemma": _ollama(
        "embeddinggemma:latest",
        PrefixesConfig(
            query="task: search result | query: {content}",
            document="title: none | text: {content}",
            label="task: classification | query: {content}",
        ),
    ),
    "mxbai": _ollama(
        "mxbai-embed-large:latest",
        PrefixesConfig(
            query="Represent this sentence for searching relevant passages: ",
            document="",
            label="",
        ),
    ),
    # --- Phase 4 pool widening: 8 new embedders (cards re-verified before
    # wiring, 2026-10-20).
    "e5-large-v2": _hf(
        "intfloat/e5-large-v2",
        "f169b11e22de13617baa190a028a32f3493550b6",
        PrefixesConfig(query="query: ", document="passage: ", label="query: "),
    ),
    "mpnet": _hf(
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        NO_PREFIXES,
    ),
    "nomic-v2-moe": _ollama(
        "nomic-embed-text-v2-moe:latest",
        PrefixesConfig(
            query="search_query: ", document="search_document: ", label="search_query: "
        ),
    ),
    "jina-v5": _hf(
        "jinaai/jina-embeddings-v5-text-small",
        "dd76d535f5447ca3897a9c893fb1e612ead98192",
        # ST config prompts (config_sentence_transformers.json): "Query: " /
        # "Document: "; the label kind takes the query-side prompt by
        # convention (labels are matched like queries). The remote-code ST
        # module needs default_task at load or every encode() raises, and does
        # not report its output dimension (pinned here, card value).
        PrefixesConfig(query="Query: ", document="Document: ", label="Query: "),
        batch_size=16,
        trust_remote_code=True,
        model_kwargs={"default_task": "retrieval"},
        dimension=1024,
    ),
    "voyage-4-nano": _hf(
        "voyageai/voyage-4-nano",
        "433ece5e27a1611e9360dd7134ecaaf79fb3681d",
        # Card prompts (also in config_sentence_transformers.json); labels use
        # the query-side prompt (no separate label prompt is documented).
        PrefixesConfig(
            query="Represent the query for retrieving supporting documents: ",
            document="Represent the document for retrieval: ",
            label="Represent the query for retrieving supporting documents: ",
        ),
        batch_size=16,
        trust_remote_code=True,
    ),
    "granite-english-r2": _hf(
        "ibm-granite/granite-embedding-english-r2",
        "47ea694b257b703fee9253d75c2b1f2985180498",
        # No prefixes anywhere (card usage embeds queries and passages bare).
        NO_PREFIXES,
        batch_size=32,
    ),
    "f2llm-80m": _hf(
        "codefuse-ai/F2LLM-v2-80M",
        "ad88d7a126711f1490cd4bad645dc9d3acc2af6a",
        # Card/ST-config query prompt; documents and labels stay bare.
        PrefixesConfig(
            query=(
                "Instruct: Given a question, retrieve passages that can help answer "
                "the question.\nQuery: "
            ),
            document="",
            label="",
        ),
        batch_size=32,
    ),
    "embeddinggemma2": _ollama(
        "embeddinggemma-2:latest",
        # Google card task templates (same shapes as embeddinggemma v1);
        # document uses "title: none" (we carry no title field), labels the
        # classification instruction. dimension: /api/show reports the
        # pre-projection 512, the served vectors are 768 (the projection is in
        # the GGUF); context: /api/show reports the 256K loader context, the
        # card's text window is 8192.
        PrefixesConfig(
            query="task: search result | query: {content}",
            document="title: none | text: {content}",
            label="task: classification | query: {content}",
        ),
        dimension=768,
        context_tokens=8192,
    ),
}
