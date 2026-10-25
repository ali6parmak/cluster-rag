"""Text-processing track: chunkers, feature extractors and combiners.

The satellite track of ``TEXT_PROCESSING_PLAN.md`` studies how text is
chunked, segmented and structured before embedding — the least-explored layer
of the pipeline.

Pure core (AGENTS.md dependency direction): this package may import
``cluster_rag.embed.base`` (a ``typing.Protocol`` plus numpy) and nothing
heavier — no torch, sentence-transformers, httpx or sqlite3. Model clients
(embedders, Ollama LLMs) are wired in the CLI code, never inside these
modules. The one exception, reviewed and recorded in
``TEXT_PROCESSING_PLAN.md`` section 2: the *combiner*
(``combining/logistic.py``) imports scikit-learn — a core dependency of the
main pipeline, with no model download, no network and no I/O.
"""

from cluster_rag.text_processors.base import (
    BaseChunker,
    Chunker,
    Combiner,
    FeatureExtractor,
    TextPair,
    config_version,
)

__all__ = [
    "BaseChunker",
    "Chunker",
    "Combiner",
    "FeatureExtractor",
    "TextPair",
    "config_version",
]
