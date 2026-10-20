"""Preprocessing variants applied before embedding (Phase 2, PLAN.md).

Pure text -> text functions, no model or I/O concerns: the cache keys on the
preprocessing *version*, so every variant that can change text output carries
a version string that changes with its parameters (the opener list of
``strip_leading_formulaic`` is hashed into the version, and the omct question
block markers likewise).

Variants (``PreprocessConfig.variant``):

- ``raw`` — identity. Baseline for every comparison.
- ``lowercase`` — collapse case only.
- ``strip_leading_formulaic`` — remove formulaic openers ("Continue to",
  "Ensure that", ...) from the start of a recommendation (Phase 1 found 17.8%
  of UPR texts start with "continue"; the opener list comes from the config,
  longest match wins; a stripped-away-everything edge falls back to the
  original text so embeddings never see an empty string).

The auxiliary probe additionally strips the "Question: ... / Answer choices:
... / Notes: ..." furniture from GTI question texts (PLAN.md section 7 open
question); that is :func:`strip_question_block`, explicit and config-markered,
not part of the UPR variants.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cluster_rag.config import PreprocessConfig

QUESTION_PREFIX_PATTERN = re.compile(r"^\s*Question:\s*")
VARIANT_NAMES: tuple[str, str, str] = ("raw", "lowercase", "strip_leading_formulaic")
_STRIP_VERSION = "strip_leading_formulaic-v1"


@dataclass(frozen=True)
class PreprocessSpec:
    """A chosen preprocessing variant with its cache-facing version string.

    ``version`` is a cache-key component (``embed/cache.py``): two runs must
    share it if and only if they apply identical text transformations.
    """

    variant: str
    version: str
    openers: tuple[str, ...] = ()  # only meaningful for strip_leading_formulaic

    def apply_one(self, text: str) -> str:
        """Transform a single text (never returns blank)."""
        return _APPLY[self.variant](text, self.openers)

    def apply(self, texts: Sequence[str]) -> list[str]:
        """Transform texts in order."""
        if self.variant == "raw":
            return list(texts)
        return [self.apply_one(text) for text in texts]

    def describe(self, texts: Sequence[str]) -> dict[str, Any]:
        """JSON-serializable diagnostics: coverage and mean chars removed."""
        if self.variant == "raw":
            return {"variant": self.variant, "version": self.version, "n_changed": 0}
        output = self.apply(texts)
        changed = [
            (original, processed)
            for original, processed in zip(texts, output, strict=True)
            if original != processed
        ]
        removed = [len(original) - len(processed) for original, processed in changed]
        return {
            "variant": self.variant,
            "version": self.version,
            "n_texts": len(texts),
            "n_changed": len(changed),
            "share_changed": len(changed) / len(texts) if texts else 0.0,
            "mean_chars_removed": statistics.fmean(removed) if removed else 0.0,
            "openers": list(self.openers),
        }


def _raw(text: str, openers: tuple[str, ...]) -> str:
    return text


def _lowercase(text: str, openers: tuple[str, ...]) -> str:
    return text.lower()


def _strip_leading_formulaic(text: str, openers: tuple[str, ...]) -> str:
    """Longest-matching opener removed from the start; boundary-checked.

    Returns the whitespace-stripped original when no opener matches or when
    stripping would empty the text (an empty string cannot be embedded).
    """
    head = text.strip()
    if not head or not openers:
        return head
    lowered = head.lower()
    for opener in sorted(set(openers), key=len, reverse=True):
        phrase = opener.strip().lower()
        if not phrase or not lowered.startswith(phrase):
            continue
        after = len(phrase)
        if after < len(head) and head[after].isalnum():
            continue  # would end mid-word ("continue" vs "continuous")
        remainder = head[after:]
        remainder = remainder.lstrip(" \t,.:;-")  # leftover opener punctuation
        if remainder.strip():
            return remainder.strip()
        return head  # opener was the whole text: keep it rather than embedding ""
    return head


def strip_question_block(text: str, markers: Sequence[str] = ("Answer choices:", "Notes:")) -> str:
    """OMCT-only transform (PLAN.md section 7 open question): keep the question
    sentence, drop the "Question:" label and the trailing answer-choice/notes
    blocks so embeddings see the question content only.

    Example: 'Question: How often were people displaced?\\n\\nAnswer
    choices:\\n- Very often\\n- Often' -> 'How often were people displaced?'.
    """
    stripped = text.strip()
    stripped = QUESTION_PREFIX_PATTERN.sub("", stripped, count=1)
    cut_points = [stripped.find(marker) for marker in markers]
    cut_points = [point for point in cut_points if point > 0]
    if cut_points:
        stripped = stripped[: min(cut_points)]
    return stripped.strip()


def question_transform_version(transform: str, markers: Sequence[str]) -> str:
    """Cache-facing version string of an omct question transform.

    Includes a content hash of the markers so a changed strip rule cannot
    silently reuse cached embeddings under the old rule.
    """
    if transform == "keep":
        return "question-transform-keep-v1"
    if transform == "strip_block":
        content = "\x00".join(markers)
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        return f"question-transform-strip_block-v1-{digest}"
    raise ValueError(f"Unknown question transform {transform!r}; expected 'keep' or 'strip_block'")


def apply_question_transform(
    texts: Sequence[str], transform: str, markers: Sequence[str]
) -> list[str]:
    """Apply an omct question transform to texts ('keep' = identity)."""
    if transform == "keep":
        return list(texts)
    if transform == "strip_block":
        return [strip_question_block(text, markers) for text in texts]
    raise ValueError(f"Unknown question transform {transform!r}; expected 'keep' or 'strip_block'")


def build_preprocess_spec(config: PreprocessConfig) -> PreprocessSpec:
    """Build the cache-facing :class:`PreprocessSpec` from the resolved config."""
    variant = config.variant
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown preprocess variant {variant!r}; expected one of {VARIANT_NAMES}")
    if variant == "strip_leading_formulaic":
        openers = tuple(config.strip_openers)
        content = "\x00".join(sorted(openers))
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        version = f"{_STRIP_VERSION}-{digest}"
        return PreprocessSpec(variant=variant, version=version, openers=openers)
    return PreprocessSpec(variant=variant, version=f"{variant}-v1")


_APPLY = {
    "raw": _raw,
    "lowercase": _lowercase,
    "strip_leading_formulaic": _strip_leading_formulaic,
}
