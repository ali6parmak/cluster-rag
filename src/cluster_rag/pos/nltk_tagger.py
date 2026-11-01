"""The **nltk** backend of the part-of-speech tagger (the track's CS-34).

The tagger the plan names: ``nltk`` (MIT) with the
``averaged_perceptron_tagger_eng`` model. The model is **downloaded** into the
run's own cache root (``cache/pos/nltk_data/``), so the download is
reproducible and never lands in a random home directory, and the model's
**digest** (the sha256 over its own files) is computed and recorded — the
embedder's and the decision model's discipline.

**The tokenization is the track's own** (:func:`tokenize`: the lower-cased
alphanumeric tokens), so the tagger's tokens and the corpus map's words are
the same units and the pure side can align them by position. No ``punkt``
download and no second tokenizer: the tagger is handed the token stream.

Pure core: **not** pure — this is the edge. It imports ``nltk`` (the new
dependency) and reads the model's files; the track's package never imports it.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from cluster_rag.pos.base import TaggerInfo
from cluster_rag.text_processors.features.pair_features import tokenize

logger = logging.getLogger(__name__)

# The model's default id (the plan's): the English averaged perceptron tagger.
DEFAULT_MODEL_ID = "averaged_perceptron_tagger_eng"

# The language code the model's id ends with (the tagger's own naming).
DEFAULT_LANGUAGE = "eng"


def model_digest(location: Path | str) -> str:
    """The model's **digest**: the sha256 over its own files.

    The files are hashed in name order and their hashes are hashed together, so
    the digest changes when any of the model's files changes — the same
    discipline the embedder's and the decision model's digests carry.

    Args:
        location: The model's directory.

    Returns:
        The digest (64 hex characters), or an empty string when the directory
        holds no file.
    """
    root = Path(location)
    if not root.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parts.append(f"{path.relative_to(root).as_posix()}:{digest}")
    if not parts:
        return ""
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class NltkTagger:
    """The nltk part-of-speech tagger (the edge client).

    The model is downloaded into ``data_root`` on the first use (the download
    is a network call; an existing model is never re-downloaded), and the
    tagger is loaded once and reused.

    Args:
        model_id: The model's id (nltk's resource name).
        data_root: The directory the model lives in (the run's cache root).
        language: The language code the model's id ends with.

    Raises:
        RuntimeError: If the model cannot be found or downloaded.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        data_root: Path | str,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._model_id = model_id
        self._language = language
        # the absolute path: nltk resolves a *relative* location as a resource
        # name, not as a directory.
        self._data_root = Path(data_root).resolve()
        self._location = self._ensure_model()
        self._digest = model_digest(self._location)
        if not self._digest:
            raise RuntimeError(
                f"the tagger model at {self._location} holds no file: its digest is empty"
            )
        self._tagger = self._load()

    @property
    def info(self) -> TaggerInfo:
        """The model's provenance (its id, its digest and the backend)."""
        return TaggerInfo(
            model_id=self._model_id,
            digest=self._digest,
            backend="nltk",
            extra={
                "language": self._language,
                "model_location": str(self._location),
                "data_root": str(self._data_root),
                "tokenizer": (
                    "the track's own tokenization (the lower-cased alphanumeric "
                    "tokens), so the tagger's tokens and the corpus map's words are "
                    "the same units"
                ),
            },
        )

    def tag(self, text: str) -> list[tuple[str, str]]:
        """The text's tokens with their tags, in the token order.

        Args:
            text: The text to tag.

        Returns:
            One ``(token, tag)`` pair per token.
        """
        tokens = tokenize(text)
        if not tokens:
            return []
        return [(token, tag) for token, tag in self._tagger.tag(tokens)]

    def _ensure_model(self) -> Path:
        """The model's directory, downloading it when it is not there yet."""
        location = self._data_root / "taggers" / self._model_id
        if location.is_dir() and any(location.iterdir()):
            logger.info("the tagger model is already there: %s", location)
            return location
        try:
            import nltk
        except ImportError as error:  # pragma: no cover - the dependency's guard
            raise RuntimeError(
                "nltk is not installed: the part-of-speech tagger needs it "
                "(the track's CS-34). Install it with `uv sync` (it is a "
                "declared dependency) and try again."
            ) from error
        logger.info("downloading the tagger model %s into %s", self._model_id, self._data_root)
        self._data_root.mkdir(parents=True, exist_ok=True)
        ok = nltk.download(self._model_id, download_dir=str(self._data_root), quiet=True)
        if not ok or not location.is_dir():
            raise RuntimeError(
                f"could not download the tagger model {self._model_id!r} into "
                f"{self._data_root}: nltk reported {'success' if ok else 'a failure'}"
            )
        return location

    def _load(self) -> Any:
        """The loaded tagger (nltk's own class, loaded once)."""
        try:
            import nltk
            from nltk.tag.perceptron import PerceptronTagger
        except ImportError as error:  # pragma: no cover - the dependency's guard
            raise RuntimeError(
                "nltk is not installed: the part-of-speech tagger needs it (the track's CS-34)."
            ) from error
        # nltk only opens files under its own data roots, so the run's cache
        # root is authorized as one (its own documented way: the data path).
        path = str(self._data_root)
        if path not in nltk.data.path:
            nltk.data.path.insert(0, path)
        return PerceptronTagger(lang=self._language, loc=str(self._location))
