"""The **part-of-speech tagger's port** (the track's CS-34).

The tagger is a **model client**: it downloads a model (``nltk``'s
``averaged_perceptron_tagger_eng``), so it lives at the edge, next to
``embed/``, ``decision/`` and ``rerank/``: the port here, the nltk backend in
:mod:`cluster_rag.pos.nltk_tagger` and the disk cache in
:mod:`cluster_rag.pos.cache`.

**Why a port at all** (AGENTS.md's dependency direction): the track's pure
package must stay importable without a model client, so the pure side talks to
a ``Tagger`` protocol and the CLI wires the real client in. The track's package
imports **this module only**.

**The port's contract.** ``tag(text)`` returns the text's tokens with their
tags, **in the token order**, and the tokens are the **track's own
tokenization** (:func:`cluster_rag.text_processors.features.pair_features.tokenize`
— the lower-cased alphanumeric tokens). The contract matters: the pure side
aligns the tags with the corpus map's words **by position**, so the tagger's
tokens and the map's words are the same units and no join is ever guessed.

**The hygiene the port carries** (the decision models' precedent): the model is
pinned by id **and digest**, every call is cached (a re-run never re-tags) and
the provenance is written into the run's artifact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaggerInfo:
    """The provenance of a tagger: the id, the pinned digest and the backend.

    ``extra`` carries the client's own details (the model directory, the
    cache's location and counters) and is written into the run's artifact.
    """

    model_id: str
    digest: str
    backend: str = "nltk"
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Tagger(Protocol):
    """A part-of-speech tagger: a text in, its tokens with their tags out."""

    @property
    def info(self) -> TaggerInfo:
        """The provenance of the model (its id and its pinned digest)."""

    def tag(self, text: str) -> list[tuple[str, str]]:
        """The text's tokens with their part-of-speech tags, in the token order.

        The tokens are the track's own tokenization (see the module's docstring).

        Args:
            text: The text to tag.

        Returns:
            One ``(token, tag)`` pair per token, in order.
        """


def tagged_tokens(tagged: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """The tokens of a tagged sequence (the tags dropped).

    Args:
        tagged: The ``(token, tag)`` pairs.

    Returns:
        The tokens, in order.
    """
    return tuple(token for token, _tag in tagged)
