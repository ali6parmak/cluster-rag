"""UPR embedding sweep: all 14 pool models, kinds document+query → `upr-embed-*` runs.

Phase 4 opening checkbox B: every pool model encodes the full UPR dataset
twice — kind ``document`` (the indexed side: labeled examples) and kind
``query`` (the side the text being classified embeds on) — so the Phase 4
retrieval predictors (kNN voting, label prototypes, later fusion) can run
symmetric and asymmetric models alike with real query/document handling.

        uv run cluster-rag sweep --file experiments/upr_embed_sweep.py

Cell labels are filesystem-safe run ids (`upr-embed-<model>`; the two Phase 2
run ids refresh their folders in place). Model configs come from
`experiments/embedder_pool.py` — the table shared with the OMCT probe sweep,
so "same pool on both datasets" cannot drift. The `document` side of
`upr-embed-e5-base-v2` / `upr-embed-arctic2` is warm in the disk cache.
"""

from __future__ import annotations

from embedder_pool import MODELS

from cluster_rag.sweep import EmbedCell, define_sweep

# Types are the encode kinds (embed/base.py KINDS): documents = indexed
# examples / label-side texts indexed later, queries = the text being
# classified. One matrix per kind lands in the run's embeddings.npz
# (`vectors_document`, `vectors_query`).
UPR_KINDS: tuple[str, ...] = ("document", "query")

cells = [
    EmbedCell(label=f"upr-embed-{name}", embed=embedder, kinds=UPR_KINDS)
    for name, embedder in MODELS.items()
]

SWEEP = define_sweep(cells, dataset="upr", label="embed_pool")
