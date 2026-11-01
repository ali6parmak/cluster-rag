"""The **part-of-speech tagger** at the edge (the track's CS-34).

A model client: it downloads a model (``nltk``'s
``averaged_perceptron_tagger_eng``), so it lives beside ``embed/``,
``decision/`` and ``rerank/``. The **port** is :mod:`cluster_rag.pos.base` (the
track's pure package imports that one only), the nltk backend is
:mod:`cluster_rag.pos.nltk_tagger` and the disk cache is
:mod:`cluster_rag.pos.cache`.
"""

from __future__ import annotations

from cluster_rag.pos.base import Tagger, TaggerInfo

__all__ = ["Tagger", "TaggerInfo"]
