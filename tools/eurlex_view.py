"""The EURLEX-4K result viewer's **data layer** (read-only).

The viewer asks four questions of the track's existing artifacts:

* *what does one dev document look like* — its gold labels, the train
  neighbours the two retrievers find, and the rankings five methods give it;
* *which methods were run, and what did they score* — one tidy table read from
  a small, **explicit** registry of ``results/eurlex-*/metrics.json``;
* *where are the label-space holes* — the label-title embedding map with the
  labels a seed's 240-document corpus does not see highlighted;
* *what does the best arm miss* — the TC-17 pack ranking's misses per bucket.

Nothing here produces a new result. The per-document scores are **computed on
demand** from the documented caches with the track's own rules (the same
``full_text`` field, the same TF-IDF config, the same cosine vote at
``k = 100``, ``power = 1.0``); the run metrics are **read** from the artifacts.
The layer never writes anything and never reads ``data/eurlex_4k/test.jsonl``.

The guard follows the ``error_app`` precedent: before the app shows a view,
the ranking's shape, the 4,197-label space, the catalogue coverage, the
chosen seed's 240-row sample and the pack's own ``family_new`` R@50 are all
checked. A mismatch raises a clear message instead of showing stale numbers
beside fresh ones.

Pure: numpy + scikit-learn + the standard library. No model client, no
Streamlit, no network, no download.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from cluster_rag.embed.cache import cache_key_path
from cluster_rag.retrieve.knn import knn_vote_scores
from cluster_rag.topic_classification.retrieval import vote_scores_from_similarity

# --- the tool's own configuration surface ----------------------------------
#
# Every value below has a CLI/env override in the app or is a documented track
# constant; the logic never guesses. Paths are relative to the repository root.

DEFAULT_RESULTS = Path("results")
DEFAULT_CACHE = Path("cache")
DEFAULT_DATA = Path("data/eurlex_4k")
DEFAULT_CATALOG = Path("catalogs/eurlex_labels.jsonl")
DEFAULT_VOCABULARY = DEFAULT_DATA / "eurovoc_concepts.jsonl"
DEFAULT_RANKING = Path("results/eurlex-embed-finetune/ranked_dev.npz")

#: The track's pinned MiniLM (the cache's model directory is keyed by this id).
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_CACHE_VERSION = "eurlex-4k-v1"
LABEL_CACHE_V1 = "eurlex-4k-label-v1"
LABEL_CACHE_V2 = "eurlex-4k-label-v2"

#: TC-21's design (docs/topic_classification.md section 27.1).
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
SOURCE_DOCUMENTS = 45_000
LABELLED_DOCUMENTS = 240

#: The fixed label space: the sorted train+dev union (test is never read).
EXPECTED_LABELS = 4_197
EXPECTED_DOCUMENTS = 6_000

#: The track's k-NN rule (TC-9 / TC-14 / TC-21): cosine votes, k=100, power 1.
KNN_K = 100
VOTE_POWER = 1.0

#: The track's TF-IDF configuration (TC-9, reused by every retrieval arm).
TFIDF_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200_000,
    "sublinear_tf": True,
    "dtype": np.float32,
}

TOP_K = 10
DEFAULT_SEED = 42

#: The support-bucket cuts of the full 45,000-document train split (TC-3).
HEAD_MIN = 100
TORSO_MIN = 10

#: The TC-21 240-document support buckets (docs section 27.1).
SPARSE_BUCKETS: tuple[str, ...] = ("1", "2-4", "5+", "unseen")

SLICE_NAMES: tuple[str, ...] = ("all_dev", "family_new", "family_new_clean")
REGIME_45K = "45k"
REGIME_240 = "240"
REGIME_LABELS = {REGIME_45K: "45,000 train documents", REGIME_240: "240-document seed"}

#: The columns of the one-document label x source matrix.
MATRIX_SOURCES: tuple[tuple[str, str], ...] = (
    ("tfidf_nn", "TF-IDF neighbours"),
    ("embedding_nn", "MiniLM neighbours"),
    ("pack", "TC-17 pack"),
    ("lexical_knn", "lexical k-NN"),
    ("embedding_knn", "embedding k-NN"),
    ("label_text_title", "label text (title)"),
    ("label_text_title_parents", "label text (title+parents)"),
)


# --- the sources ------------------------------------------------------------


@dataclass(frozen=True)
class Sources:
    """The read-only files and cache roots the viewer uses.

    Attributes:
        results: The run-artifact folder.
        cache: The embedding cache root (``<cache>/v1/...``).
        data_dir: The EURLEX-4K split folder (``train.jsonl`` and ``dev.jsonl``
            are read; ``test.jsonl`` is never opened).
        catalog_path: The committed EUROVOC catalogue.
        vocabulary_path: The label-id -> title vocabulary.
        ranking_path: The TC-17 pack's ``ranked_dev.npz``.
        model_id: The embedding model whose cache keys are read.
        document_version: The document cache version.
        label_v1_version: The title-only label cache version.
        label_v2_version: The enriched label cache version.
    """

    results: Path
    cache: Path
    data_dir: Path
    catalog_path: Path
    vocabulary_path: Path
    ranking_path: Path
    model_id: str = MODEL_ID
    document_version: str = DOCUMENT_CACHE_VERSION
    label_v1_version: str = LABEL_CACHE_V1
    label_v2_version: str = LABEL_CACHE_V2


def resolve_sources(
    *,
    results: Path | str = DEFAULT_RESULTS,
    cache: Path | str = DEFAULT_CACHE,
    data_dir: Path | str = DEFAULT_DATA,
    catalog_path: Path | str = DEFAULT_CATALOG,
    vocabulary_path: Path | str | None = None,
    ranking_path: Path | str | None = None,
    model_id: str = MODEL_ID,
) -> Sources:
    """Resolve the viewer's sources from plain paths.

    Args:
        results: The results folder.
        cache: The embedding cache root.
        data_dir: The EURLEX-4K split folder.
        catalog_path: The committed label catalogue.
        vocabulary_path: The label vocabulary (default: inside ``data_dir``).
        ranking_path: The TC-17 ranking (default: inside ``results``).
        model_id: The embedding model id used for the cache keys.

    Returns:
        The resolved sources.
    """
    results_path = Path(results)
    return Sources(
        results=results_path,
        cache=Path(cache),
        data_dir=Path(data_dir),
        catalog_path=Path(catalog_path),
        vocabulary_path=Path(vocabulary_path or Path(data_dir) / "eurovoc_concepts.jsonl"),
        ranking_path=Path(
            ranking_path or results_path / "eurlex-embed-finetune" / "ranked_dev.npz"
        ),
        model_id=model_id,
    )


# --- the documents ----------------------------------------------------------


@dataclass(frozen=True)
class DocumentSection:
    """One coloured part of a document's ``full_text``.

    Attributes:
        kind: ``title`` / ``header`` / ``recitals`` / ``main_body``.
        text: The section's own text (``main_body`` is newline-joined).
    """

    kind: str
    text: str


@dataclass(frozen=True)
class Document:
    """One EURLEX-4K document, as the track's retrievers see it.

    Attributes:
        celex_id: The CELEX identifier (the document's id in every artifact).
        document_type: The record's ``document_type``.
        title: The title.
        header: The header block.
        recitals: The recitals.
        main_body: The body paragraphs.
        labels: The gold EUROVOC label ids.
    """

    celex_id: str
    document_type: str
    title: str
    header: str
    recitals: str
    main_body: tuple[str, ...]
    labels: tuple[str, ...]

    @property
    def sections(self) -> tuple[DocumentSection, ...]:
        """The four ``full_text`` parts, in the text's own order.

        The parts are for display only: the retrievers see the concatenated
        :attr:`text`, exactly as the cache keyed it.
        """
        return (
            DocumentSection("title", self.title),
            DocumentSection("header", self.header),
            DocumentSection("recitals", self.recitals),
            DocumentSection("main_body", "\n".join(self.main_body)),
        )

    @property
    def text(self) -> str:
        """The document's ``full_text`` — exactly what the caches embedded."""
        return "\n".join([self.title, self.header, self.recitals, "\n".join(self.main_body)])


def full_text(record: Mapping[str, Any]) -> str:
    """The track's ``full_text``: title, header, recitals, body, newline-joined."""
    return "\n".join(
        [
            str(record["title"]),
            str(record["header"]),
            str(record["recitals"]),
            "\n".join(str(part) for part in record["main_body"]),
        ]
    )


def _document(record: Mapping[str, Any]) -> Document:
    """Build one :class:`Document` from a split record."""
    return Document(
        celex_id=str(record["celex_id"]),
        document_type=str(record.get("document_type", "")),
        title=str(record["title"]),
        header=str(record["header"]),
        recitals=str(record["recitals"]),
        main_body=tuple(str(part) for part in record["main_body"]),
        labels=tuple(str(label) for label in record["eurovoc_concepts"]),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file (the track's own splits and catalogue format).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no artifact at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CatalogEntry:
    """One EUROVOC catalogue row (TC-19's join of the official SKOS).

    Attributes:
        label_id: The dataset's label id.
        name: The dataset vocabulary title.
        parents: The immediate ``skos:broader`` ids.
        ancestors: The full transitive ``skos:broader`` chain.
        parent_names: The English ``skos:prefLabel`` of each parent.
        definition: The English definition/scope note, when present.
        definition_kind: ``definition`` / ``scope_note`` / ``None``.
    """

    label_id: str
    name: str
    parents: tuple[str, ...]
    ancestors: tuple[str, ...]
    parent_names: tuple[str, ...]
    definition: str | None
    definition_kind: str | None


@dataclass(frozen=True)
class Dataset:
    """The two readable splits, the label space, the catalogue and the support.

    Attributes:
        train: The 45,000 train documents.
        dev: The 6,000 dev documents (the evaluation split).
        labels: The sorted train+dev label union (4,197 ids).
        label_index: The label id -> column map.
        catalog: The label id -> catalogue row (4,271 labels, metadata only).
        vocabulary: The label id -> title map.
        gold: Every dev row's gold label columns.
        train_gold: Every train row's gold label columns.
        support: The full-train document frequency of every label.
        bucket: The full-train support bucket of every label
            (``head`` / ``torso`` / ``tail`` / ``unseen``).
    """

    train: tuple[Document, ...]
    dev: tuple[Document, ...]
    labels: tuple[str, ...]
    label_index: Mapping[str, int]
    catalog: Mapping[str, CatalogEntry]
    vocabulary: Mapping[str, str]
    gold: tuple[frozenset[int], ...]
    train_gold: tuple[frozenset[int], ...]
    support: Mapping[str, int]
    bucket: Mapping[str, str]


def support_bucket(support: int) -> str:
    """The full-train bucket from a document frequency (the track's TC-3 cuts)."""
    if support >= HEAD_MIN:
        return "head"
    if support >= TORSO_MIN:
        return "torso"
    if support >= 1:
        return "tail"
    return "unseen"


def load_dataset(sources: Sources) -> Dataset:
    """Read ``train.jsonl`` and ``dev.jsonl``, the catalogue and the vocabulary.

    ``test.jsonl`` is never opened: the file name does not occur in this
    function and the ``Sources.data_dir`` is only joined with the two names.

    Raises:
        FileNotFoundError: If a source is missing.
        ValueError: If a documented label has no catalogue/vocabulary row.
    """
    train = tuple(_document(record) for record in read_jsonl(sources.data_dir / "train.jsonl"))
    dev = tuple(_document(record) for record in read_jsonl(sources.data_dir / "dev.jsonl"))
    labels = tuple(
        sorted({label for doc in (*train, *dev) for label in doc.labels}),
    )
    label_index = {label: index for index, label in enumerate(labels)}
    vocabulary = {str(row["id"]): str(row["title"]) for row in read_jsonl(sources.vocabulary_path)}
    catalog = {
        str(row["id"]): CatalogEntry(
            label_id=str(row["id"]),
            name=str(row["name"]),
            parents=tuple(str(parent) for parent in row.get("parents") or ()),
            ancestors=tuple(str(parent) for parent in row.get("ancestors") or ()),
            parent_names=tuple(str(name) for name in row.get("parent_names") or ()),
            definition=(None if row.get("definition") is None else str(row.get("definition"))),
            definition_kind=(
                None if row.get("definition_kind") is None else str(row.get("definition_kind"))
            ),
        )
        for row in read_jsonl(sources.catalog_path)
    }
    missing = [label for label in labels if label not in catalog]
    if missing:
        raise ValueError(f"{len(missing)} labels have no catalogue row: {missing[:5]}")
    support: Counter[str] = Counter()
    for doc in train:
        for label in set(doc.labels):
            support[label] += 1
    gold = tuple(
        frozenset(label_index[label] for label in doc.labels if label in label_index) for doc in dev
    )
    train_gold = tuple(
        frozenset(label_index[label] for label in doc.labels if label in label_index)
        for doc in train
    )
    bucket = {label: support_bucket(support[label]) for label in labels}
    return Dataset(
        train=train,
        dev=dev,
        labels=labels,
        label_index=label_index,
        catalog=catalog,
        vocabulary=vocabulary,
        gold=gold,
        train_gold=train_gold,
        support=dict(support),
        bucket=bucket,
    )


def label_title(dataset: Dataset, label: str) -> str:
    """The label's title (the vocabulary, else the catalogue's name)."""
    title = dataset.vocabulary.get(label, "")
    if title:
        return title
    entry = dataset.catalog.get(label)
    return entry.name if entry else label


def label_text(dataset: Dataset, label: str, variant: str = "title_parents") -> str:
    """The label text for one of TC-21's three variants.

    Args:
        dataset: The loaded dataset (vocabulary + catalogue).
        label: The label id.
        variant: ``title`` / ``title_parents`` / ``title_parents_definition``.

    Returns:
        The exact text the cache key was built from.

    Raises:
        ValueError: On an unknown variant.
    """
    if variant not in ("title", "title_parents", "title_parents_definition"):
        raise ValueError(f"unknown label-text variant {variant!r}")
    parts = [label_title(dataset, label)]
    entry = dataset.catalog.get(label)
    if variant != "title" and entry is not None:
        parts.extend(dict.fromkeys(entry.parent_names))
    if variant == "title_parents_definition" and entry is not None and entry.definition:
        parts.append(entry.definition)
    return " | ".join(parts)


def build_label_matrix(docs: Sequence[Document], labels: Sequence[str]) -> np.ndarray:
    """The ``(n_documents, n_labels)`` multi-hot matrix (the k-NN's index side)."""
    index = {label: column for column, label in enumerate(labels)}
    matrix = np.zeros((len(docs), len(labels)), dtype=bool)
    for row, doc in enumerate(docs):
        for label in doc.labels:
            column = index.get(label)
            if column is not None:
                matrix[row, column] = True
    return matrix


# --- the cached vectors -----------------------------------------------------


class MissingVectorError(RuntimeError):
    """A cache entry the viewer needs is absent (the viewer never downloads)."""


def document_vector_path(sources: Sources, text: str) -> Path:
    """The cache path of one document vector (MiniLM, no prefix)."""
    return cache_key_path(
        sources.cache,
        model_id=sources.model_id,
        prefix="",
        preprocessing_version=sources.document_version,
        text=text,
    )


def label_vector_path(sources: Sources, text: str, version: str) -> Path:
    """The cache path of one label-text vector (MiniLM, no prefix)."""
    return cache_key_path(
        sources.cache,
        model_id=sources.model_id,
        prefix="",
        preprocessing_version=version,
        text=text,
    )


def _read_vector(path: Path, what: str) -> np.ndarray:
    """Load one unit-norm float32 vector, or say exactly what is missing."""
    if not path.is_file():
        raise MissingVectorError(
            f"no cached vector for {what}: {path}. The viewer never downloads or "
            "encodes; warm the cache with the track's run first."
        )
    try:
        vector = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a stale entry
        raise MissingVectorError(f"unreadable cache entry for {what}: {path} ({exc})") from exc
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"cached vector for {what} is not a finite 1-D array: {path}")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0 or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"cached vector for {what} is not unit-norm (|v|={norm:g}): {path}")
    return array


def load_document_vectors(sources: Sources, docs: Sequence[Document]) -> np.ndarray:
    """Load the cached MiniLM vectors for a sequence of documents.

    Returns:
        A ``(n_documents, dimension)`` float32 unit-norm matrix.

    Raises:
        MissingVectorError: If any document vector is missing.
    """
    if not docs:
        raise ValueError("load_document_vectors needs at least one document")
    matrix: np.ndarray | None = None
    for row, doc in enumerate(docs):
        vector = _read_vector(
            document_vector_path(sources, doc.text),
            f"document {doc.celex_id}",
        )
        if matrix is None:
            matrix = np.empty((len(docs), vector.shape[0]), dtype=np.float32)
        matrix[row] = vector
    assert matrix is not None
    return matrix


def load_label_vectors(
    sources: Sources, dataset: Dataset, variant: str = "title_parents"
) -> np.ndarray:
    """Load the cached label-text vectors for every label of the space.

    ``variant="title"`` reads the title-only cache (``eurlex-4k-label-v1``); the
    two enriched variants read ``eurlex-4k-label-v2``.

    Returns:
        A ``(n_labels, dimension)`` float32 unit-norm matrix.

    Raises:
        MissingVectorError: If any label vector is missing.
    """
    version = sources.label_v1_version if variant == "title" else sources.label_v2_version
    matrix: np.ndarray | None = None
    for row, label in enumerate(dataset.labels):
        text = label_text(dataset, label, variant)
        vector = _read_vector(
            label_vector_path(sources, text, version),
            f"label {label} ({variant})",
        )
        if matrix is None:
            matrix = np.empty((len(dataset.labels), vector.shape[0]), dtype=np.float32)
        matrix[row] = vector
    assert matrix is not None
    return matrix


# --- the TC-17 ranking ------------------------------------------------------


@dataclass(frozen=True)
class Ranking:
    """The TC-17 pack's dev ranking and its own reproduction checkpoints.

    Attributes:
        ranked: The ``(6,000, 4,197)`` int32 ranked label matrix.
        baseline: The npz's own ``family_new`` baseline metrics.
        deltas: The npz's own deltas against the TC-17 artifact.
        seconds: The export's run time.
    """

    ranked: np.ndarray
    baseline: Mapping[str, float]
    deltas: Mapping[str, float]
    seconds: float


def _json_scalar(value: Any) -> dict[str, float]:
    """Parse the npz's JSON-in-a-scalar metadata strings."""
    if isinstance(value, Mapping):
        return {str(key): float(item) for key, item in value.items()}
    text = str(value)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): float(item) for key, item in payload.items()}


def load_ranking(path: Path) -> Ranking:
    """Load the TC-17 ``ranked_dev.npz`` once.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the archive lacks the ``ranked`` matrix.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no TC-17 ranking at {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "ranked" not in archive:
            raise ValueError(f"{path} has no 'ranked' matrix")
        ranked = np.asarray(archive["ranked"])
        baseline = _json_scalar(archive["baseline"]) if "baseline" in archive else {}
        deltas = _json_scalar(archive["deltas"]) if "deltas" in archive else {}
        seconds = float(archive["seconds"]) if "seconds" in archive else 0.0
    return Ranking(ranked=ranked, baseline=baseline, deltas=deltas, seconds=seconds)


# --- the metrics registry ---------------------------------------------------
#
# The artifacts do not share one schema: the standard ranking runs store
# ``methods.<arm>.slices.<slice>.primary``, TC-21 stores per-seed flat keys and
# a pooled mean, TC-12 stores decision blocks, and three runs store their
# comparison at the top level. The registry is **explicit per run id**; each
# extractor is a small function for one schema, never a generic walker.


@dataclass(frozen=True)
class RunSpec:
    """One run the viewer reads: its folder, its extractor and its regime."""

    run_id: str
    extractor: str
    regime: str


#: The registry. Every ``results/eurlex-*/metrics.json`` on disk must be listed
#: here (the guard refuses an unlisted artifact); every row in the Methods table
#: comes from one of these extractors.
RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec("eurlex-shortlist", "arm_slices", REGIME_45K),
    RunSpec("eurlex-family-heldout", "arm_slices", REGIME_45K),
    RunSpec("eurlex-embed-chunkmean", "arm_slices", REGIME_45K),
    RunSpec("eurlex-fusion", "fusion", REGIME_45K),
    RunSpec("eurlex-labelname", "arm_slices", REGIME_45K),
    RunSpec("eurlex-zeroshot-calibrated", "calibrated", REGIME_45K),
    RunSpec("eurlex-lexical-rrf", "arm_slices", REGIME_45K),
    RunSpec("eurlex-multiview", "arm_slices", REGIME_45K),
    RunSpec("eurlex-cooccurrence", "arm_slices", REGIME_45K),
    RunSpec("eurlex-zero-shot-quota", "arm_slices", REGIME_45K),
    RunSpec("eurlex-threshold-calibration", "threshold", REGIME_45K),
    RunSpec("eurlex-cross-encoder-rerank", "arm_slices", REGIME_45K),
    RunSpec("eurlex-candidate-pack", "arm_slices", REGIME_45K),
    RunSpec("eurlex-cluster-routing", "arm_slices", REGIME_45K),
    RunSpec("eurlex-hard-negative-discipline", "arm_slices", REGIME_45K),
    RunSpec("eurlex-parent-channel", "arm_slices", REGIME_45K),
    RunSpec("eurlex-embed-finetune", "arm_slices", REGIME_45K),
    RunSpec("eurlex-llm-rerank", "llm", REGIME_45K),
    RunSpec("eurlex-embeddinggemma-zeroshot", "arm_slices", REGIME_45K),
    RunSpec("eurlex-label-sparse", "sparse", REGIME_240),
    RunSpec("eurlex-label-sparse-transfer", "sparse", REGIME_240),
    RunSpec("eurlex-label-sparse-clusters", "sparse", REGIME_240),
)


@dataclass(frozen=True)
class MetricRow:
    """One (run, method, slice) row of the Methods table.

    Attributes:
        run_id: The run's folder name.
        change_set: The artifact's own ``change_set`` string.
        method: The artifact's arm key (with a variant suffix when needed).
        arm: The artifact's human arm description.
        regime: ``45k`` or ``240``.
        slice: The evaluation slice name.
        recall50: R@50 (macro over documents), when the artifact carries it.
        r_precision: R-precision, when carried.
        ndcg10: nDCG@10, when carried.
        unseen_recall50: micro recall@50 over the unseen gold assignments.
        micro_f1: The decision micro-F1 (TC-12 only).
        cost_seconds: The run's own ``cost.seconds_total``.
        artifact: The artifact path the row was read from.
    """

    run_id: str
    change_set: str
    method: str
    arm: str
    regime: str
    slice: str
    recall50: float | None
    r_precision: float | None
    ndcg10: float | None
    unseen_recall50: float | None
    micro_f1: float | None
    cost_seconds: float | None
    artifact: str


def _float(value: Any) -> float | None:
    """A JSON number as a float (``None`` stays ``None``)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _eval_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """The primary/secondary/diagnostics metrics of one evaluation block."""
    primary = block.get("primary") if isinstance(block.get("primary"), Mapping) else {}
    diagnostics = block.get("diagnostics") if isinstance(block.get("diagnostics"), Mapping) else {}
    label_recall = (
        diagnostics.get("label_recall@50")
        if isinstance(diagnostics.get("label_recall@50"), Mapping)
        else {}
    )
    unseen = label_recall.get("unseen") if isinstance(label_recall.get("unseen"), Mapping) else {}
    return {
        "recall50": _float(primary.get("recall@50")),
        "r_precision": _float(primary.get("r_precision")),
        "ndcg10": _float(primary.get("ndcg@10")),
        "unseen_recall50": _float(unseen.get("recall")),
    }


def _row(
    spec: RunSpec,
    payload: Mapping[str, Any],
    artifact: Path,
    *,
    method: str,
    arm: str,
    slice_name: str,
    block: Mapping[str, Any] | None = None,
    recall50: float | None = None,
    micro_f1: float | None = None,
    unseen_recall50: float | None = None,
) -> MetricRow:
    """Assemble one :class:`MetricRow` from one evaluation block."""
    values = _eval_block(block or {})
    cost = payload.get("cost") if isinstance(payload.get("cost"), Mapping) else {}
    return MetricRow(
        run_id=spec.run_id,
        change_set=str(payload.get("change_set", "")),
        method=method,
        arm=arm,
        regime=spec.regime,
        slice=slice_name,
        recall50=values["recall50"] if recall50 is None else recall50,
        r_precision=values["r_precision"],
        ndcg10=values["ndcg10"],
        unseen_recall50=(values["unseen_recall50"] if unseen_recall50 is None else unseen_recall50),
        micro_f1=micro_f1,
        cost_seconds=_float(cost.get("seconds_total")),
        artifact=str(artifact),
    )


def _slice_blocks(arm: Mapping[str, Any]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    """Yield ``(variant, slice, block)`` for one arm of a standard run."""
    slices = arm.get("slices")
    if isinstance(slices, Mapping):
        for slice_name, block in slices.items():
            if isinstance(block, Mapping) and "primary" in block:
                yield "", str(slice_name), block
        return
    flat = [
        (str(key), value)
        for key, value in arm.items()
        if isinstance(value, Mapping) and "primary" in value
    ]
    if flat:
        for slice_name, block in flat:
            yield "", "all_dev" if slice_name == "overall" else slice_name, block
        return
    overall = arm.get("overall")
    if isinstance(overall, Mapping):
        yield "", "all_dev", overall
        clean = arm.get("clean")
        if isinstance(clean, Mapping):
            yield "", "clean", clean
        return
    for variant, nested in arm.items():
        if not isinstance(nested, Mapping):
            continue
        nested_slices = nested.get("slices")
        if isinstance(nested_slices, Mapping):
            for slice_name, block in nested_slices.items():
                if isinstance(block, Mapping) and "primary" in block:
                    yield str(variant), str(slice_name), block
            continue
        for slice_name in SLICE_NAMES:
            block = nested.get(slice_name)
            if isinstance(block, Mapping) and "primary" in block:
                yield str(variant), slice_name, block


def extract_arm_slices(
    spec: RunSpec, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """The standard ranking runs: ``methods``/``arms`` -> slices -> primary."""
    arms = payload.get("methods") or payload.get("arms") or {}
    if not isinstance(arms, Mapping):
        return ()
    rows: list[MetricRow] = []
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        arm_text = str(arm.get("arm") or name)
        for variant, slice_name, block in _slice_blocks(arm):
            method = str(name) if not variant else f"{name} :: {variant}"
            rows.append(
                _row(
                    spec,
                    payload,
                    artifact,
                    method=method,
                    arm=arm_text,
                    slice_name=slice_name,
                    block=block,
                )
            )
    return tuple(rows)


def extract_fusion(
    spec: RunSpec, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """TC-5's fusion: the selected-weight CV blocks per dictionary arm."""
    results = payload.get("results") or {}
    if not isinstance(results, Mapping):
        return ()
    rows: list[MetricRow] = []
    blocks = (
        ("cv", "all_dev"),
        ("cv_family_new", "family_new"),
        ("cv_family_new_clean", "family_new_clean"),
    )
    for arm_name, arm in results.items():
        if not isinstance(arm, Mapping):
            continue
        for key, slice_name in blocks:
            block = arm.get(key)
            if isinstance(block, Mapping):
                rows.append(
                    _row(
                        spec,
                        payload,
                        artifact,
                        method=f"{arm_name} :: selected weights",
                        arm=str(arm_name),
                        slice_name=slice_name,
                        block=block,
                    )
                )
    return tuple(rows)


def extract_calibrated(
    spec: RunSpec, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """TC-6's calibrated zero-shot channel: the selected-alpha CV blocks."""
    cv = payload.get("cv") or {}
    if not isinstance(cv, Mapping):
        return ()
    rows: list[MetricRow] = []
    for key, slice_name in (
        ("metrics", "all_dev"),
        ("family_new", "family_new"),
        ("family_new_clean", "family_new_clean"),
    ):
        block = cv.get(key)
        if isinstance(block, Mapping):
            rows.append(
                _row(
                    spec,
                    payload,
                    artifact,
                    method="calibrated CV",
                    arm="zero-shot channel at the CV-selected alpha",
                    slice_name=slice_name,
                    block=block,
                )
            )
    return tuple(rows)


def extract_threshold(
    spec: RunSpec, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """TC-12: the shortlist R@50 and the decision arms' micro-F1."""
    methods = payload.get("methods") or {}
    if not isinstance(methods, Mapping):
        return ()
    rows: list[MetricRow] = []
    for base, arm in methods.items():
        if not isinstance(arm, Mapping):
            continue
        shortlist = arm.get("shortlist_r50")
        if isinstance(shortlist, Mapping):
            for slice_name, value in shortlist.items():
                rows.append(
                    _row(
                        spec,
                        payload,
                        artifact,
                        method=f"{base} :: shortlist",
                        arm="shortlist ranking (the decision's input)",
                        slice_name=str(slice_name),
                        recall50=_float(value),
                    )
                )
        for key, arm_text in (
            ("control_top5", "top-5 (control)"),
            ("treatment_threshold", "score >= tau (treatment)"),
        ):
            block = arm.get(key)
            if not isinstance(block, Mapping):
                continue
            slices = block.get("slices")
            if not isinstance(slices, Mapping):
                continue
            for slice_name, metrics in slices.items():
                if not isinstance(metrics, Mapping):
                    continue
                rows.append(
                    _row(
                        spec,
                        payload,
                        artifact,
                        method=f"{base} :: {key}",
                        arm=arm_text,
                        slice_name=str(slice_name),
                        micro_f1=_float(metrics.get("micro_f1")),
                    )
                )
    return tuple(rows)


def _pooled_mean(pooled: Mapping[str, Any], slice_name: str, key: str) -> float | None:
    """One pooled metric's mean (TC-21 stores ``slice/key -> {mean, sd, ...}``)."""
    entry = pooled.get(f"{slice_name}/{key}")
    return _float(entry.get("mean")) if isinstance(entry, Mapping) else None


def extract_sparse(
    spec: RunSpec, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """TC-21: per-seed metrics pooled to their mean (the artifact's own mean)."""
    arms = payload.get("arms") or {}
    if not isinstance(arms, Mapping):
        return ()
    rows: list[MetricRow] = []
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        pooled = arm.get("pooled")
        if not isinstance(pooled, Mapping):
            continue
        cost = payload.get("cost") if isinstance(payload.get("cost"), Mapping) else {}
        for slice_name in SLICE_NAMES:
            rows.append(
                MetricRow(
                    run_id=spec.run_id,
                    change_set=str(payload.get("change_set", "")),
                    method=str(name),
                    arm=str(arm.get("arm") or name),
                    regime=spec.regime,
                    slice=slice_name,
                    recall50=_pooled_mean(pooled, slice_name, "recall@50"),
                    r_precision=_pooled_mean(pooled, slice_name, "r_precision"),
                    ndcg10=_pooled_mean(pooled, slice_name, "ndcg@10"),
                    unseen_recall50=_pooled_mean(pooled, slice_name, "label_recall@50/unseen"),
                    micro_f1=None,
                    cost_seconds=_float(cost.get("seconds_total")),
                    artifact=str(artifact),
                )
            )
    return tuple(rows)


def extract_llm(spec: RunSpec, payload: Mapping[str, Any], artifact: Path) -> tuple[MetricRow, ...]:
    """TC-18: the LLM rerank's stage-1 ranking beside its pack baseline."""
    rows: list[MetricRow] = []
    stage1 = payload.get("stage1") or {}
    metrics = stage1.get("metrics") if isinstance(stage1, Mapping) else None
    if isinstance(metrics, Mapping):
        for slice_name, block in metrics.items():
            if isinstance(block, Mapping):
                rows.append(
                    _row(
                        spec,
                        payload,
                        artifact,
                        method="llm_rerank :: stage 1",
                        arm="tev1 rerank of the pack's top-64",
                        slice_name=str(slice_name),
                        block=block,
                    )
                )
    baseline = payload.get("baseline") or {}
    slices = baseline.get("slices") if isinstance(baseline, Mapping) else None
    if isinstance(slices, Mapping):
        for slice_name, block in slices.items():
            if isinstance(block, Mapping):
                rows.append(
                    _row(
                        spec,
                        payload,
                        artifact,
                        method="pack_baseline",
                        arm="TC-17 pack (the rerank's input)",
                        slice_name=str(slice_name),
                        block=block,
                    )
                )
    return tuple(rows)


EXTRACTORS = {
    "arm_slices": extract_arm_slices,
    "fusion": extract_fusion,
    "calibrated": extract_calibrated,
    "threshold": extract_threshold,
    "sparse": extract_sparse,
    "llm": extract_llm,
}


def run_spec(run_id: str) -> RunSpec | None:
    """The registry entry for one run id (``None`` when unlisted)."""
    return next((spec for spec in RUN_SPECS if spec.run_id == run_id), None)


def extract_metrics(
    run_id: str, payload: Mapping[str, Any], artifact: Path
) -> tuple[MetricRow, ...]:
    """Extract the Methods rows for one run id and one payload.

    Raises:
        KeyError: If the run id is not in the registry.
    """
    spec = run_spec(run_id)
    if spec is None:
        raise KeyError(f"run {run_id!r} is not in the registry")
    return EXTRACTORS[spec.extractor](spec, payload, artifact)


def load_metrics(sources: Sources) -> tuple[MetricRow, ...]:
    """Read every registry artifact and extract its rows.

    Raises:
        FileNotFoundError: If a registry artifact is missing.
        KeyError: If an artifact's ``run_id`` is not its folder name.
    """
    rows: list[MetricRow] = []
    for spec in RUN_SPECS:
        artifact = sources.results / spec.run_id / "metrics.json"
        payload = read_json(artifact)
        declared = payload.get("run_id")
        if declared is not None and str(declared) != spec.run_id:
            raise KeyError(
                f"{artifact} declares run_id {declared!r}, the folder is {spec.run_id!r}"
            )
        rows.extend(extract_metrics(spec.run_id, payload, artifact))
    return tuple(rows)


def unregistered_runs(results: Path) -> tuple[str, ...]:
    """The ``results/eurlex-*`` folders with a metrics artifact not in the registry."""
    listed = {spec.run_id for spec in RUN_SPECS}
    found = {
        path.parent.name
        for path in sorted(results.glob("eurlex-*/metrics.json"))
        if path.parent.name not in listed
    }
    return tuple(sorted(found))


# --- one document's neighbours and rankings --------------------------------


@dataclass(frozen=True)
class LexicalIndex:
    """The TF-IDF vectorizer and its normalized index matrix."""

    vectorizer: TfidfVectorizer
    matrix: Any
    docs: tuple[Document, ...]


def build_lexical_index(docs: Sequence[Document]) -> LexicalIndex:
    """Fit the track's TF-IDF configuration on the index documents."""
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    matrix = vectorizer.fit_transform([doc.text for doc in docs])
    return LexicalIndex(vectorizer=vectorizer, matrix=matrix, docs=tuple(docs))


def lexical_similarity(index: LexicalIndex, text: str) -> np.ndarray:
    """The query's cosine to every index document (TF-IDF rows are L2-normed)."""
    query = index.vectorizer.transform([text])
    return np.asarray((index.matrix @ query.T).todense()).ravel().astype(np.float64)


@dataclass(frozen=True)
class NeighbourRow:
    """One nearest train document for the query document."""

    rank: int
    doc_id: str
    title: str
    text: str
    sections: tuple[DocumentSection, ...]
    score: float
    labels: tuple[str, ...]
    shared: tuple[str, ...]


def _neighbour_rows(
    scores: np.ndarray,
    docs: Sequence[Document],
    gold_labels: Sequence[str],
    *,
    k: int,
) -> tuple[NeighbourRow, ...]:
    """The top-k index documents by a score vector (stable, descending)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")[:k]
    gold = set(gold_labels)
    rows: list[NeighbourRow] = []
    for rank, column in enumerate(order, start=1):
        doc = docs[int(column)]
        rows.append(
            NeighbourRow(
                rank=rank,
                doc_id=doc.celex_id,
                title=doc.title.strip(),
                text=doc.text,
                sections=doc.sections,
                score=float(scores[int(column)]),
                labels=doc.labels,
                shared=tuple(label for label in doc.labels if label in gold),
            )
        )
    return tuple(rows)


def embedding_neighbours(
    query_vector: np.ndarray,
    index_vectors: np.ndarray,
    index_docs: Sequence[Document],
    *,
    k: int = TOP_K,
    gold_labels: Sequence[str] = (),
) -> tuple[NeighbourRow, ...]:
    """The top-k train documents by cached MiniLM cosine.

    Args:
        query_vector: The query's unit-norm cached vector.
        index_vectors: The index's ``(n, d)`` unit-norm vectors.
        index_docs: The index documents, in row order.
        k: The number of neighbours.
        gold_labels: The query's gold labels (for the ``shared`` marks).

    Returns:
        The neighbours, closest first.
    """
    scores = np.asarray(index_vectors, dtype=np.float32) @ np.asarray(
        query_vector, dtype=np.float32
    )
    return _neighbour_rows(scores, index_docs, gold_labels, k=k)


def lexical_neighbours(
    index: LexicalIndex,
    text: str,
    *,
    k: int = TOP_K,
    gold_labels: Sequence[str] = (),
) -> tuple[NeighbourRow, ...]:
    """The top-k train documents by TF-IDF cosine."""
    return _neighbour_rows(
        lexical_similarity(index, text),
        index.docs,
        gold_labels,
        k=k,
    )


@dataclass(frozen=True)
class OverlapSummary:
    """The query's and its neighbours' label overlap."""

    shared: tuple[str, ...]
    document_only: tuple[str, ...]
    neighbours_only: tuple[str, ...]

    @property
    def n_shared(self) -> int:
        """The number of labels both sides carry."""
        return len(self.shared)

    @property
    def n_document_only(self) -> int:
        """The number of gold labels no neighbour carries."""
        return len(self.document_only)

    @property
    def n_neighbours_only(self) -> int:
        """The number of neighbour labels the document does not carry."""
        return len(self.neighbours_only)


def overlap_summary(
    gold_labels: Sequence[str],
    neighbour_label_sets: Sequence[Sequence[str]],
) -> OverlapSummary:
    """Summarise the label overlap of the query and its neighbours.

    Args:
        gold_labels: The query's gold labels.
        neighbour_label_sets: Each neighbour's gold labels.

    Returns:
        ``shared`` (gold ∩ neighbours), ``document_only`` (gold − neighbours)
        and ``neighbours_only`` (neighbours − gold), each sorted.
    """
    gold = set(gold_labels)
    neighbours: set[str] = set()
    for labels in neighbour_label_sets:
        neighbours.update(labels)
    return OverlapSummary(
        shared=tuple(sorted(gold & neighbours)),
        document_only=tuple(sorted(gold - neighbours)),
        neighbours_only=tuple(sorted(neighbours - gold)),
    )


@dataclass(frozen=True)
class SearchIndex:
    """One retrieval index: the full train split or a seed's 240 documents.

    Attributes:
        regime: ``45k`` or ``240``.
        seed: The seed of the 240-document sample (``None`` for the full split).
        rows: The train row numbers in the index (the artifact's own order).
        docs: The index documents.
        label_matrix: The ``(n_index, n_labels)`` multi-hot gold matrix.
        vectors: The cached MiniLM vectors, row-aligned with ``docs``.
        lexical: The fitted TF-IDF index.
    """

    regime: str
    seed: int | None
    rows: tuple[int, ...]
    docs: tuple[Document, ...]
    label_matrix: np.ndarray
    vectors: np.ndarray
    lexical: LexicalIndex

    @property
    def size(self) -> int:
        """The number of labelled documents in the index."""
        return len(self.docs)


def seed_sample(
    seed: int,
    *,
    n_sources: int = SOURCE_DOCUMENTS,
    n_labelled: int = LABELLED_DOCUMENTS,
) -> np.ndarray:
    """TC-21's 240-row sample: ``default_rng(seed).choice(n_sources, ...)``, sorted.

    Args:
        seed: The seed (42..46 in the track).
        n_sources: The train split size the rule samples from.
        n_labelled: The labelled-corpus size.

    Returns:
        The sorted int64 row numbers.

    Raises:
        ValueError: If the sample size is larger than the source size.
    """
    if n_labelled > n_sources:
        raise ValueError(f"cannot draw {n_labelled} rows from {n_sources}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_sources, size=n_labelled, replace=False))


def build_search_index(
    dataset: Dataset,
    sources: Sources,
    *,
    seed: int | None = None,
    vectors: np.ndarray | None = None,
    lexical: LexicalIndex | None = None,
) -> SearchIndex:
    """Build one retrieval index over the train split or a seed's 240 rows.

    Args:
        dataset: The loaded dataset.
        sources: The resolved sources (the vector cache).
        seed: ``None`` for the 45,000-document index; a seed for the 240.
        vectors: Preloaded index vectors (else read from the cache).
        lexical: A prebuilt lexical index (else fit here).

    Returns:
        The search index.
    """
    if seed is None:
        rows = tuple(range(len(dataset.train)))
        regime = REGIME_45K
    else:
        rows = tuple(int(row) for row in seed_sample(seed, n_sources=len(dataset.train)))
        regime = REGIME_240
    docs = tuple(dataset.train[row] for row in rows)
    label_matrix = build_label_matrix(docs, dataset.labels)
    index_vectors = load_document_vectors(sources, docs) if vectors is None else np.asarray(vectors)
    return SearchIndex(
        regime=regime,
        seed=seed,
        rows=rows,
        docs=docs,
        label_matrix=label_matrix,
        vectors=index_vectors,
        lexical=lexical if lexical is not None else build_lexical_index(docs),
    )


@dataclass(frozen=True)
class LabelTag:
    """One gold label, tagged for display."""

    label: str
    title: str
    bucket: str
    support: int
    unseen: bool
    parent_names: tuple[str, ...]
    definition: str | None
    definition_kind: str | None


def label_tag(dataset: Dataset, label: str) -> LabelTag:
    """Build the display tag of one label."""
    entry = dataset.catalog.get(label)
    bucket = dataset.bucket.get(label, "unseen")
    return LabelTag(
        label=label,
        title=label_title(dataset, label),
        bucket=bucket,
        support=int(dataset.support.get(label, 0)),
        unseen=bucket == "unseen",
        parent_names=entry.parent_names if entry else (),
        definition=entry.definition if entry else None,
        definition_kind=entry.definition_kind if entry else None,
    )


@dataclass(frozen=True)
class MethodRow:
    """One label in one method's top-10 for the query document."""

    rank: int
    label: str
    title: str
    score: float | None
    pack_rank: int
    gold: bool
    hit: bool
    bucket: str
    unseen: bool


@dataclass(frozen=True)
class MethodRanking:
    """One method's top-10 for the query document."""

    key: str
    label: str
    rows: tuple[MethodRow, ...]
    hits: int
    unseen_hits: int


@dataclass(frozen=True)
class MatrixRow:
    """One label of the label x source matrix."""

    label: str
    title: str
    gold: bool
    marks: tuple[bool, ...]


@dataclass(frozen=True)
class DocumentView:
    """Everything the One document view shows for one dev document.

    Attributes:
        document_row: The dev row number.
        doc: The document.
        gold: The gold-label tags, in label-column order.
        lexical: The TF-IDF neighbours.
        embedding: The MiniLM neighbours.
        overlap: The neighbours' label overlap.
        methods: The five method rankings.
        matrix: The label x source matrix.
        scores: The per-method score arrays (label-column order).
        orders: The per-method descending label orders.
        pack_rank: The pack's 0-based rank of every label.
        best_cosine: The best neighbour cosine carrying each label, per source.
        index_regime: The retrieval index's regime.
        index_seed: The retrieval index's seed.
        index_size: The number of labelled index documents.
    """

    document_row: int
    doc: Document
    gold: tuple[LabelTag, ...]
    lexical: tuple[NeighbourRow, ...]
    embedding: tuple[NeighbourRow, ...]
    overlap: OverlapSummary
    methods: tuple[MethodRanking, ...]
    matrix: tuple[MatrixRow, ...]
    scores: Mapping[str, np.ndarray]
    orders: Mapping[str, np.ndarray]
    pack_rank: np.ndarray
    best_cosine: Mapping[str, Mapping[str, float]]
    index_regime: str
    index_seed: int | None
    index_size: int


def _top_order(scores: np.ndarray) -> np.ndarray:
    """The descending label order (stable ties, earlier column wins)."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def _method(
    key: str,
    label: str,
    order: np.ndarray,
    scores: np.ndarray | None,
    *,
    labels: Sequence[str],
    titles: Sequence[str],
    gold: set[str],
    bucket: Mapping[str, str],
    pack_rank: np.ndarray,
    k: int,
) -> MethodRanking:
    """Build one method's top-k rows with gold, hit and unseen marks."""
    rows: list[MethodRow] = []
    for rank, column in enumerate(order[:k], start=1):
        label_id = labels[int(column)]
        is_gold = label_id in gold
        rows.append(
            MethodRow(
                rank=rank,
                label=label_id,
                title=titles[int(column)],
                score=None if scores is None else float(scores[int(column)]),
                pack_rank=int(pack_rank[int(column)]) + 1,
                gold=is_gold,
                hit=is_gold,
                bucket=bucket.get(label_id, ""),
                unseen=bucket.get(label_id) == "unseen",
            )
        )
    return MethodRanking(
        key=key,
        label=label,
        rows=tuple(rows),
        hits=sum(1 for row in rows if row.hit),
        unseen_hits=sum(1 for row in rows if row.hit and row.unseen),
    )


def _best_cosines(
    similarity: np.ndarray,
    docs: Sequence[Document],
    *,
    k: int = KNN_K,
) -> dict[str, float]:
    """Each label's best cosine among the top-k neighbour documents."""
    order = np.argsort(-np.asarray(similarity, dtype=np.float64), kind="stable")[:k]
    best: dict[str, float] = {}
    for column in order:
        score = float(similarity[int(column)])
        for label in docs[int(column)].labels:
            if score > best.get(label, -np.inf):
                best[label] = score
    return best


def label_text_scores(query_vector: np.ndarray, label_vectors: np.ndarray) -> np.ndarray:
    """The query's cosine to every label text (a dot product of unit vectors)."""
    return np.asarray(
        np.asarray(label_vectors, dtype=np.float32) @ np.asarray(query_vector, dtype=np.float32),
        dtype=np.float64,
    )


def _label_matrix_row(
    labels: Sequence[str],
    titles: Sequence[str],
    gold: set[str],
    source_sets: Mapping[str, set[str]],
) -> tuple[MatrixRow, ...]:
    """The label x source matrix: gold first, then the sources' labels."""
    ordered = set(gold)
    for values in source_sets.values():
        ordered.update(values)
    order = {label: column for column, label in enumerate(labels)}
    rows = sorted(ordered, key=lambda label: (label not in gold, order.get(label, len(labels))))
    return tuple(
        MatrixRow(
            label=label,
            title=titles[order[label]],
            gold=label in gold,
            marks=tuple(label in source_sets[key] for key, _ in MATRIX_SOURCES),
        )
        for label in rows
    )


def build_document_view(
    dataset: Dataset,
    ranking: Ranking,
    index: SearchIndex,
    *,
    document_row: int,
    query_vector: np.ndarray,
    label_vectors_title: np.ndarray,
    label_vectors_parents: np.ndarray,
    top_k: int = TOP_K,
) -> DocumentView:
    """Compute the full One document view for one dev row.

    The scores are the track's own: TF-IDF cosine votes
    (``vote_scores_from_similarity``), MiniLM cosine votes
    (``knn_vote_scores``) and the label-text cosine (a dot product of unit
    vectors). The TC-17 pack contributes its ranks, not a score — the npz
    carries no score column. The label vectors are loaded by the caller (once
    per run, not once per document).

    Args:
        dataset: The loaded dataset.
        ranking: The TC-17 ranking.
        index: The chosen retrieval index.
        document_row: The dev row number.
        query_vector: The dev document's cached unit-norm vector.
        label_vectors_title: The title-only label cache (v1).
        label_vectors_parents: The title+parents label cache (v2).
        top_k: The depth of the displayed rankings.

    Returns:
        The view.

    Raises:
        ValueError: If the row is out of range or a matrix's width is wrong.
    """
    if not 0 <= document_row < len(dataset.dev):
        raise ValueError(f"document row {document_row} is outside 0..{len(dataset.dev) - 1}")
    if ranking.ranked.shape[1] != len(dataset.labels):
        raise ValueError(
            f"the ranking has {ranking.ranked.shape[1]} columns, "
            f"the label space {len(dataset.labels)}"
        )
    if label_vectors_title.shape[0] != len(dataset.labels):
        raise ValueError("the title label cache does not have one row per label")
    if label_vectors_parents.shape[0] != len(dataset.labels):
        raise ValueError("the enriched label cache does not have one row per label")
    doc = dataset.dev[document_row]
    labels = dataset.labels
    titles = tuple(label_title(dataset, label) for label in labels)
    gold_columns = sorted(dataset.gold[document_row])
    gold = tuple(label_tag(dataset, labels[column]) for column in gold_columns)
    gold_names = tuple(tag.label for tag in gold)
    gold_set = set(gold_names)

    lexical_scores = lexical_similarity(index.lexical, doc.text)
    embedding_scores = np.asarray(index.vectors, dtype=np.float32) @ np.asarray(
        query_vector, dtype=np.float32
    )
    lexical_rows = _neighbour_rows(lexical_scores, index.lexical.docs, gold_names, k=top_k)
    embedding_rows = _neighbour_rows(embedding_scores, index.docs, gold_names, k=top_k)
    neighbour_sets = [row.labels for row in (*lexical_rows, *embedding_rows)]
    overlap = overlap_summary(gold_names, neighbour_sets)

    lexical_votes = vote_scores_from_similarity(
        lexical_scores[None, :],
        index.label_matrix,
        k=KNN_K,
        power=VOTE_POWER,
    )[0]
    embedding_votes = knn_vote_scores(
        index.vectors,
        index.label_matrix,
        np.asarray(query_vector, dtype=np.float32)[None, :],
        k=KNN_K,
        power=VOTE_POWER,
    )[0]
    title_scores = label_text_scores(query_vector, label_vectors_title)
    parents_scores = label_text_scores(query_vector, label_vectors_parents)

    pack_row = np.asarray(ranking.ranked[document_row], dtype=np.int64)
    pack_rank = np.empty(len(labels), dtype=np.int32)
    pack_rank[pack_row] = np.arange(len(labels), dtype=np.int32)

    scores = {
        "lexical_knn": np.asarray(lexical_votes, dtype=np.float64),
        "embedding_knn": np.asarray(embedding_votes, dtype=np.float64),
        "label_text_title": title_scores,
        "label_text_title_parents": parents_scores,
    }
    orders = {name: _top_order(values) for name, values in scores.items()}
    bucket = dataset.bucket
    method_specs = (
        ("pack", "TC-17 pack", pack_row, None),
        ("lexical_knn", "lexical k-NN", orders["lexical_knn"], scores["lexical_knn"]),
        (
            "embedding_knn",
            "embedding k-NN",
            orders["embedding_knn"],
            scores["embedding_knn"],
        ),
        (
            "label_text_title",
            "label text (title)",
            orders["label_text_title"],
            scores["label_text_title"],
        ),
        (
            "label_text_title_parents",
            "label text (title+parents)",
            orders["label_text_title_parents"],
            scores["label_text_title_parents"],
        ),
    )
    methods = tuple(
        _method(
            key,
            label,
            order,
            values,
            labels=labels,
            titles=titles,
            gold=gold_set,
            bucket=bucket,
            pack_rank=pack_rank,
            k=top_k,
        )
        for key, label, order, values in method_specs
    )

    by_key = {method.key: {row.label for row in method.rows} for method in methods}
    source_sets = {
        "tfidf_nn": {label for row in lexical_rows for label in row.labels},
        "embedding_nn": {label for row in embedding_rows for label in row.labels},
        **by_key,
    }
    return DocumentView(
        document_row=document_row,
        doc=doc,
        gold=gold,
        lexical=lexical_rows,
        embedding=embedding_rows,
        overlap=overlap,
        methods=methods,
        matrix=_label_matrix_row(labels, titles, gold_set, source_sets),
        scores=scores,
        orders=orders,
        pack_rank=pack_rank,
        best_cosine={
            "lexical": _best_cosines(lexical_scores, index.lexical.docs),
            "embedding": _best_cosines(embedding_scores, index.docs),
        },
        index_regime=index.regime,
        index_seed=index.seed,
        index_size=index.size,
    )


@dataclass(frozen=True)
class LabelSignals:
    """The signals behind one candidate label for one document."""

    label: str
    title: str
    gold: bool
    bucket: str
    support: int
    unseen: bool
    parent_names: tuple[str, ...]
    definition: str | None
    definition_kind: str | None
    pack_rank: int
    ranks: Mapping[str, int]
    scores: Mapping[str, float | None]
    best_cosine: Mapping[str, float]
    index_support: int
    index_bucket: str


def label_signals(
    dataset: Dataset,
    view: DocumentView,
    label: str,
    *,
    index_support: int | None = None,
) -> LabelSignals:
    """Collect the stored/computed signals behind one candidate label.

    Every number is either read from the dataset/catalogue or computed by
    :func:`build_document_view` with the track's own rule; nothing is invented.

    Raises:
        KeyError: If the label is not in the 4,197-label space.
    """
    if label not in dataset.label_index:
        raise KeyError(f"label {label!r} is not in the label space")
    column = dataset.label_index[label]
    tag = label_tag(dataset, label)
    ranks = {key: int(np.flatnonzero(order == column)[0]) + 1 for key, order in view.orders.items()}
    support = int(index_support or 0)
    return LabelSignals(
        label=label,
        title=tag.title,
        gold=label in {row.label for row in view.gold},
        bucket=tag.bucket,
        support=tag.support,
        unseen=tag.unseen,
        parent_names=tag.parent_names,
        definition=tag.definition,
        definition_kind=tag.definition_kind,
        pack_rank=int(view.pack_rank[column]) + 1,
        ranks=ranks,
        scores={key: float(values[column]) for key, values in view.scores.items()},
        best_cosine={
            source: float(values.get(label, 0.0)) for source, values in view.best_cosine.items()
        },
        index_support=support,
        index_bucket=sparse_bucket(support),
    )


def sparse_bucket(support: int) -> str:
    """TC-21's support bucket from a 240-document corpus."""
    if support <= 0:
        return "unseen"
    if support == 1:
        return "1"
    if support <= 4:
        return "2-4"
    return "5+"


# --- the Methods page's curve and the Errors page's misses ------------------


@dataclass(frozen=True)
class CurveRow:
    """One point of the label-sparse curve.

    Attributes:
        labelled_documents: ``240`` (a seed) or ``45000`` (a full-train run).
        family: The canonical method family (lexical / embedding / pack / text).
        run_id: The run the point came from.
        method: The artifact's arm key.
        slice: The evaluation slice.
        recall50: R@50.
        unseen_recall50: The unseen micro recall@50, when carried.
        regime: ``45k`` or ``240``.
    """

    labelled_documents: int
    family: str
    run_id: str
    method: str
    slice: str
    recall50: float
    unseen_recall50: float | None
    regime: str


def method_family(method: str) -> str:
    """A method key's canonical family (``other`` when the curve does not use it)."""
    key = method.lower()
    if "label_text" in key:
        return "label text"
    if "lexical" in key:
        return "lexical k-NN"
    if "embedding" in key:
        return "embedding k-NN"
    if "pack" in key:
        return "pack"
    return "other"


def label_sparse_curve(rows: Sequence[MetricRow]) -> tuple[CurveRow, ...]:
    """The R@50-vs-corpus-size points the Methods page plots.

    The 240-document points are the five seeds (each pooled mean is one point);
    the 45,000-document points are the full-train runs. They are **not** one
    continuous curve, and the app says so.
    """
    points: list[CurveRow] = []
    for row in rows:
        if row.recall50 is None:
            continue
        family = method_family(row.method)
        if family == "other":
            continue
        points.append(
            CurveRow(
                labelled_documents=(
                    LABELLED_DOCUMENTS if row.regime == REGIME_240 else SOURCE_DOCUMENTS
                ),
                family=family,
                run_id=row.run_id,
                method=row.method,
                slice=row.slice,
                recall50=row.recall50,
                unseen_recall50=row.unseen_recall50,
                regime=row.regime,
            )
        )
    return tuple(points)


@dataclass(frozen=True)
class DocumentMiss:
    """One dev document's misses under the TC-17 pack's top-50."""

    document_row: int
    doc_id: str
    title: str
    n_gold: int
    n_missed: int
    missed: tuple[str, ...]
    missed_titles: tuple[str, ...]
    missed_unseen: int


@dataclass(frozen=True)
class LabelMiss:
    """One label's miss count under the TC-17 pack's top-50."""

    label: str
    title: str
    bucket: str
    support: int
    n_missed: int
    n_gold: int


@dataclass(frozen=True)
class MissTable:
    """The Errors page's tidy tables."""

    documents: tuple[DocumentMiss, ...]
    labels: tuple[LabelMiss, ...]
    depth: int
    n_documents: int
    n_gold_assignments: int
    n_missed: int


def pack_misses(dataset: Dataset, ranking: Ranking, *, depth: int = 50) -> MissTable:
    """The TC-17 pack's misses, per document and per label (computed, not read).

    A miss is a gold label outside the pack's top-``depth`` for that document.

    Raises:
        ValueError: If ``depth`` is not positive or the ranking width is wrong.
    """
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    if ranking.ranked.shape[1] != len(dataset.labels):
        raise ValueError(
            f"the ranking has {ranking.ranked.shape[1]} columns, "
            f"the label space {len(dataset.labels)}"
        )
    missed_by_label: Counter[str] = Counter()
    gold_by_label: Counter[str] = Counter()
    documents: list[DocumentMiss] = []
    total_gold = 0
    total_missed = 0
    for row, doc in enumerate(dataset.dev):
        gold = dataset.gold[row]
        top = {int(label) for label in ranking.ranked[row, :depth]}
        missed_columns = sorted(gold - top)
        missed = tuple(dataset.labels[column] for column in missed_columns)
        for column in gold:
            gold_by_label[dataset.labels[int(column)]] += 1
        for label in missed:
            missed_by_label[label] += 1
        total_gold += len(gold)
        total_missed += len(missed)
        documents.append(
            DocumentMiss(
                document_row=row,
                doc_id=doc.celex_id,
                title=doc.title.strip(),
                n_gold=len(gold),
                n_missed=len(missed),
                missed=missed,
                missed_titles=tuple(label_title(dataset, label) for label in missed),
                missed_unseen=sum(1 for label in missed if dataset.bucket.get(label) == "unseen"),
            )
        )
    documents.sort(key=lambda item: (-item.n_missed, item.document_row))
    labels = [
        LabelMiss(
            label=label,
            title=label_title(dataset, label),
            bucket=dataset.bucket.get(label, "unseen"),
            support=int(dataset.support.get(label, 0)),
            n_missed=count,
            n_gold=gold_by_label[label],
        )
        for label, count in missed_by_label.items()
    ]
    order = dataset.label_index
    labels.sort(key=lambda item: (-item.n_missed, order.get(item.label, len(order))))
    return MissTable(
        documents=tuple(documents),
        labels=tuple(labels),
        depth=depth,
        n_documents=len(dataset.dev),
        n_gold_assignments=total_gold,
        n_missed=total_missed,
    )


# --- the label map ----------------------------------------------------------


@dataclass(frozen=True)
class LabelNeighbourRow:
    """One label's nearest label in the label-text embedding space.

    Attributes:
        rank: 1-based rank among the other labels.
        label: The neighbour's label id.
        title: The neighbour's title.
        cosine: The label-text cosine to the selected label.
        bucket: The neighbour's full-train support bucket.
        support: The neighbour's full-train support.
        n_dev_gold: How many dev documents carry the neighbour.
        seen: Whether the neighbour is in the seed's 240-document corpus
            (``None`` when the caller did not pass the seed's seen set).
    """

    rank: int
    label: str
    title: str
    cosine: float
    bucket: str
    support: int
    n_dev_gold: int
    seen: bool | None


def label_neighbours(
    dataset: Dataset,
    label_vectors: np.ndarray,
    label: str,
    *,
    k: int = TOP_K,
    seen: Collection[str] | None = None,
    eligible: Collection[str] | None = None,
) -> tuple[LabelNeighbourRow, ...]:
    """The nearest labels to one label in a label-vector space.

    The cosine is the same dot product the track uses; the projection on the
    map is only an approximation of these neighbours.

    Args:
        dataset: The loaded dataset (for titles, buckets, dev support).
        label_vectors: The ``(n_labels, d)`` unit-norm label vectors.
        label: The selected label id.
        k: The number of neighbours.
        seen: The seed's seen label ids (for the ``seen`` flag).
        eligible: The labels that have a vector in this representation (the
            evidence maps skip the labels with no train document).

    Raises:
        KeyError: If the label is not in the label space.
        ValueError: If the vectors do not have one row per label, or if the
            selected label is not eligible.
    """
    if label not in dataset.label_index:
        raise KeyError(f"label {label!r} is not in the label space")
    vectors = np.asarray(label_vectors, dtype=np.float32)
    if vectors.shape[0] != len(dataset.labels):
        raise ValueError(
            f"the label vectors have {vectors.shape[0]} rows, the label space {len(dataset.labels)}"
        )
    eligible_set = None if eligible is None else set(eligible)
    if eligible_set is not None and label not in eligible_set:
        raise ValueError(f"label {label!r} has no vector in this representation")
    column = dataset.label_index[label]
    scores = vectors @ vectors[column]
    order = np.argsort(-scores, kind="stable")
    dev_counts: Counter[str] = Counter()
    for gold in dataset.gold:
        for gold_column in gold:
            dev_counts[dataset.labels[int(gold_column)]] += 1
    seen_set = None if seen is None else set(seen)
    rows: list[LabelNeighbourRow] = []
    for candidate in order:
        if int(candidate) == column:
            continue
        neighbour = dataset.labels[int(candidate)]
        if eligible_set is not None and neighbour not in eligible_set:
            continue
        rows.append(
            LabelNeighbourRow(
                rank=len(rows) + 1,
                label=neighbour,
                title=label_title(dataset, neighbour),
                cosine=float(scores[int(candidate)]),
                bucket=dataset.bucket.get(neighbour, "unseen"),
                support=int(dataset.support.get(neighbour, 0)),
                n_dev_gold=dev_counts.get(neighbour, 0),
                seen=None if seen_set is None else neighbour in seen_set,
            )
        )
        if len(rows) == k:
            break
    return tuple(rows)


@dataclass(frozen=True)
class LabelMap:
    """A 2-D projection of the label-title embeddings."""

    coordinates: np.ndarray
    method: str
    note: str


def label_map(
    label_vectors: np.ndarray,
    *,
    random_state: int = DEFAULT_SEED,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> LabelMap:
    """Project the label-title embeddings to 2-D (UMAP, PCA fallback).

    ``umap-learn`` is the intended reducer; when it is not installed the
    viewer falls back to PCA and **says so** — it never downloads a package.

    Args:
        label_vectors: The ``(n_labels, d)`` unit-norm label vectors.
        random_state: The reducer's seed.
        n_neighbors: UMAP's neighbourhood size.
        min_dist: UMAP's minimum distance.

    Returns:
        The 2-D coordinates and the reducer's name and note.
    """
    matrix = np.asarray(label_vectors, dtype=np.float32)
    try:
        import umap  # noqa: PLC0415 - optional dependency, probed at use
    except ImportError:
        coordinates = PCA(n_components=2, random_state=random_state).fit_transform(matrix)
        return LabelMap(
            coordinates=np.asarray(coordinates, dtype=np.float64),
            method="PCA",
            note="umap-learn is not installed in this environment; PCA is shown instead.",
        )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coordinates = reducer.fit_transform(matrix)
    return LabelMap(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        method="UMAP",
        note="UMAP of the cached MiniLM label vectors (cosine metric).",
    )


# --- the label-evidence map -------------------------------------------------


@dataclass(frozen=True)
class LabelEvidence:
    """The document-evidence centroid of every label.

    Attributes:
        vectors: The ``(n_labels, d)`` centroids; a label with no document in
            the representation has a zero row.
        support: The number of documents carrying each label.
        has_evidence: Whether a label has at least one document.
    """

    vectors: np.ndarray
    support: np.ndarray
    has_evidence: np.ndarray


def label_evidence_centroids(
    dataset: Dataset,
    sources: Sources | None = None,
    *,
    seed: int | None = None,
    vectors: np.ndarray | None = None,
) -> LabelEvidence:
    """Average the cached document vectors carrying each label.

    The labels are used only to group the train documents; no model is fitted.
    With ``seed=None`` the full 45,000-document train split is read; with a
    seed, only that seed's 240 rows are read (TC-21's rule).

    Args:
        dataset: The loaded dataset.
        sources: The resolved sources (needed when ``vectors`` is not given).
        seed: ``None`` for the full train split; a TC-21 seed for the 240.
        vectors: Preloaded document vectors, row-aligned with the chosen rows.

    Returns:
        The unit-norm centroids, their support counts and the evidence mask.

    Raises:
        ValueError: On a vector shape mismatch or a zero-length centroid.
    """
    if seed is None:
        rows = tuple(range(len(dataset.train)))
    else:
        rows = tuple(int(row) for row in seed_sample(seed, n_sources=len(dataset.train)))
    if vectors is None:
        if sources is None:
            raise ValueError("label_evidence_centroids needs sources or vectors")
        docs = tuple(dataset.train[row] for row in rows)
        vectors = load_document_vectors(sources, docs)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape[0] != len(rows):
        raise ValueError(
            f"the document vectors have {matrix.shape[0]} rows, the selected corpus {len(rows)}"
        )
    n_labels = len(dataset.labels)
    sums = np.zeros((n_labels, matrix.shape[1]), dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.int64)
    for offset, row in enumerate(rows):
        for column in dataset.train_gold[row]:
            sums[int(column)] += matrix[offset]
            counts[int(column)] += 1
    has_evidence = counts > 0
    centroids = np.zeros((n_labels, matrix.shape[1]), dtype=np.float32)
    if has_evidence.any():
        centroids[has_evidence] = (sums[has_evidence] / counts[has_evidence, None]).astype(
            np.float32
        )
        norms = np.linalg.norm(centroids[has_evidence], axis=1)
        if np.any(norms == 0.0):
            raise ValueError("a label's evidence centroid is a zero vector")
        centroids[has_evidence] = centroids[has_evidence] / norms[:, None]
    return LabelEvidence(
        vectors=centroids,
        support=counts,
        has_evidence=has_evidence,
    )


@dataclass(frozen=True)
class EvidenceLabelMap:
    """A 2-D projection of the label-evidence centroids.

    Attributes:
        coordinates: ``(n_labels, 2)`` coordinates; NaN for a label with no
            document in the representation.
        vectors: The ``(n_labels, d)`` unit-norm centroids (zero rows for labels
            with no document).
        method: ``UMAP`` or ``PCA``.
        note: One line saying exactly what was projected.
        support: The number of documents carrying each label.
        has_evidence: Whether a label has at least one document.
    """

    coordinates: np.ndarray
    vectors: np.ndarray
    method: str
    note: str
    support: np.ndarray
    has_evidence: np.ndarray


def label_evidence_map(
    dataset: Dataset,
    sources: Sources | None = None,
    *,
    seed: int | None = None,
    vectors: np.ndarray | None = None,
    random_state: int = DEFAULT_SEED,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> EvidenceLabelMap:
    """Project the label-evidence centroids to 2-D (UMAP, PCA fallback)."""
    evidence = label_evidence_centroids(
        dataset,
        sources,
        seed=seed,
        vectors=vectors,
    )
    indices = np.flatnonzero(evidence.has_evidence)
    if len(indices) == 0:
        raise ValueError("no label has a document in this representation")
    projection = label_map(
        evidence.vectors[indices],
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    coordinates = np.full((len(dataset.labels), 2), np.nan, dtype=np.float64)
    coordinates[indices] = projection.coordinates
    scope = (
        "the full 45,000-document train split"
        if seed is None
        else f"the seed {seed} 240-document sample"
    )
    if projection.method == "UMAP":
        note = f"UMAP of the mean full-text embeddings of {scope} (cosine metric)."
    else:
        note = (
            "umap-learn is not installed in this environment; PCA of the mean "
            f"full-text embeddings of {scope} is shown instead."
        )
    return EvidenceLabelMap(
        coordinates=coordinates,
        vectors=evidence.vectors,
        method=projection.method,
        note=note,
        support=evidence.support,
        has_evidence=evidence.has_evidence,
    )


# --- the document map -------------------------------------------------------


@dataclass(frozen=True)
class DocumentMap:
    """A 2-D projection of the dev documents' ``full_text``.

    Attributes:
        coordinates: ``(n_dev, 2)`` coordinates.
        representation: ``tfidf`` or ``minilm``.
        method: ``UMAP`` or ``PCA``.
        note: One line saying exactly what was projected.
    """

    coordinates: np.ndarray
    representation: str
    method: str
    note: str


def document_map(
    dataset: Dataset,
    sources: Sources,
    *,
    representation: str = "tfidf",
    vectors: np.ndarray | None = None,
    n_components: int = 128,
    random_state: int = DEFAULT_SEED,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> DocumentMap:
    """Project the 6,000 dev documents by their ``full_text``.

    Two representations:

    * ``tfidf`` — the track's TF-IDF config fitted on the **train** documents,
      transformed for dev, reduced with ``TruncatedSVD`` and projected;
    * ``minilm`` — the cached MiniLM ``full_text`` vectors, projected.

    No label enters either projection. The map is a view, not a model.

    Raises:
        ValueError: On an unknown representation or a vector shape mismatch.
    """
    if representation not in ("tfidf", "minilm"):
        raise ValueError(f"unknown document-map representation {representation!r}")
    if representation == "minilm":
        matrix = (
            load_document_vectors(sources, dataset.dev)
            if vectors is None
            else np.asarray(vectors, dtype=np.float32)
        )
        if matrix.shape[0] != len(dataset.dev):
            raise ValueError(
                f"the document vectors have {matrix.shape[0]} rows, "
                f"the dev split {len(dataset.dev)}"
            )
        projection = label_map(
            matrix,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        note = (
            "UMAP of the cached MiniLM full-text vectors (cosine metric)."
            if projection.method == "UMAP"
            else "umap-learn is not installed; PCA of the cached MiniLM "
            "full-text vectors is shown instead."
        )
        return DocumentMap(
            coordinates=projection.coordinates,
            representation=representation,
            method=projection.method,
            note=note,
        )
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    train_matrix = vectorizer.fit_transform([doc.text for doc in dataset.train])
    dev_matrix = vectorizer.transform([doc.text for doc in dataset.dev])
    components = max(1, min(n_components, min(train_matrix.shape) - 1))
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    svd.fit(train_matrix)
    dev_reduced = normalize(svd.transform(dev_matrix), norm="l2", axis=1, copy=False)
    del train_matrix, dev_matrix
    projection = label_map(
        dev_reduced,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    explained = float(svd.explained_variance_ratio_.sum())
    note = (
        f"TF-IDF word 1–2 grams fitted on the {len(dataset.train):,} train "
        f"documents, transformed for dev, TruncatedSVD {components} "
        f"({explained:.1%} variance) and UMAP (cosine)."
    )
    if projection.method != "UMAP":
        note = (
            "umap-learn is not installed; PCA of the TF-IDF → TruncatedSVD "
            f"{components} dev vectors is shown instead."
        )
    return DocumentMap(
        coordinates=projection.coordinates,
        representation=representation,
        method=projection.method,
        note=note,
    )


# --- the guard --------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One guard check: what it verified and whether it held."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class GuardReport:
    """The guard's checks and the viewer's provenance."""

    checks: tuple[Check, ...]
    provenance: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether every check held."""
        return all(check.ok for check in self.checks)

    def lines(self) -> list[str]:
        """The guard's own report lines."""
        output = ["PROVENANCE", *[f"  {line}" for line in self.provenance], "", "GUARD"]
        for index, check in enumerate(self.checks, start=1):
            status = "OK" if check.ok else "MISMATCH"
            output.append(f"  {index}. {check.name}: {status}")
            output.append(f"     {check.detail}")
        output.append("")
        output.append(f"RESULT: {'OK' if self.ok else 'MISMATCH'}")
        return output


def title_skeleton(title: str) -> str:
    """The track's title-skeleton family key (digits -> ``#``, punctuation cut)."""
    text = title.lower()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[^a-z# ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def family_new_mask(dataset: Dataset) -> np.ndarray:
    """The TC-3 ``family_new`` mask: dev titles whose family is absent from train."""
    families = {title_skeleton(doc.title) for doc in dataset.train}
    return np.asarray(
        [title_skeleton(doc.title) not in families for doc in dataset.dev],
        dtype=bool,
    )


def macro_recall_at_50(
    ranked: np.ndarray, gold: Sequence[frozenset[int]], rows: np.ndarray
) -> float:
    """The macro R@50 over a set of documents (the track's primary metric)."""
    values = []
    for row in rows:
        wanted = gold[int(row)]
        if not wanted:
            continue
        top = {int(label) for label in ranked[int(row), :50]}
        values.append(len(wanted & top) / len(wanted))
    return float(np.mean(values)) if values else 0.0


def check_ranking(
    ranking: Ranking,
    *,
    label_count: int,
    expected_documents: int,
    expected_labels: int,
) -> Check:
    """Check the TC-17 ranking's shape, dtype, range and permutation property."""
    matrix = np.asarray(ranking.ranked)
    problems: list[str] = []
    if matrix.shape != (expected_documents, expected_labels):
        problems.append(
            f"shape is {matrix.shape}, expected ({expected_documents}, {expected_labels})"
        )
    if matrix.dtype != np.int32:
        problems.append(f"dtype is {matrix.dtype}, expected int32")
    if label_count != expected_labels:
        problems.append(f"the label space has {label_count} labels, expected {expected_labels}")
    if matrix.ndim == 2 and matrix.shape[1] == label_count and matrix.size:
        if int(matrix.min()) < 0 or int(matrix.max()) >= label_count:
            problems.append(
                f"values span {int(matrix.min())}..{int(matrix.max())}, "
                f"expected 0..{label_count - 1}"
            )
        elif not np.array_equal(
            np.sort(matrix, axis=1),
            np.broadcast_to(np.arange(label_count, dtype=matrix.dtype), matrix.shape),
        ):
            problems.append("a row is not a permutation of the label space")
    detail = (
        "; ".join(problems)
        if problems
        else f"{matrix.shape[0]} x {matrix.shape[1]} int32, every row a permutation"
    )
    return Check("TC-17 ranking", not problems, detail)


def check_catalog(dataset: Dataset) -> Check:
    """Check that the catalogue and the vocabulary cover the label space."""
    missing_catalog = [label for label in dataset.labels if label not in dataset.catalog]
    missing_title = [label for label in dataset.labels if not label_title(dataset, label).strip()]
    problems: list[str] = []
    if missing_catalog:
        problems.append(
            f"{len(missing_catalog)} labels have no catalogue row: {missing_catalog[:5]}"
        )
    if missing_title:
        problems.append(f"{len(missing_title)} labels have no title: {missing_title[:5]}")
    detail = (
        "; ".join(problems)
        if problems
        else f"{len(dataset.labels)} labels; catalogue rows {len(dataset.catalog)}; "
        f"{sum(1 for label in dataset.labels if dataset.catalog[label].parents)} with parents"
    )
    return Check("label space + catalogue", not problems, detail)


def check_seed_sample(
    dataset: Dataset,
    *,
    seed: int,
    expected_seen: int | None = None,
    n_sources: int = SOURCE_DOCUMENTS,
    n_labelled: int = LABELLED_DOCUMENTS,
) -> Check:
    """Check the chosen seed's 240 rows against the pre-registered rule."""
    if n_sources != len(dataset.train):
        return Check(
            "TC-21 seed sample",
            False,
            f"the rule samples {n_sources} train rows, the dataset has {len(dataset.train)}",
        )
    sample = seed_sample(seed, n_sources=n_sources, n_labelled=n_labelled)
    repeated = len(sample) != len({int(row) for row in sample})
    in_range = bool(((sample >= 0) & (sample < n_sources)).all())
    sorted_ascending = bool(np.all(sample[:-1] <= sample[1:]))
    seen = {label for row in sample for label in dataset.train[int(row)].labels}
    problems: list[str] = []
    if len(sample) != n_labelled:
        problems.append(f"the sample has {len(sample)} rows, expected {n_labelled}")
    if repeated:
        problems.append("the sample repeats a row")
    if not in_range:
        problems.append("a sampled row is outside the train split")
    if not sorted_ascending:
        problems.append("the sample is not sorted")
    if expected_seen is not None and len(seen) != expected_seen:
        problems.append(f"the sample sees {len(seen)} labels, the artifact says {expected_seen}")
    detail = (
        "; ".join(problems)
        if problems
        else f"seed {seed}: {n_labelled} rows, distinct, sorted, in range; {len(seen)} seen labels"
    )
    return Check("TC-21 seed sample", not problems, detail)


def check_pack_reproduction(dataset: Dataset, ranking: Ranking) -> Check:
    """Check the ranking's own ``family_new`` R@50 against the npz checkpoint."""
    mask = family_new_mask(dataset)
    recomputed = macro_recall_at_50(ranking.ranked, dataset.gold, np.flatnonzero(mask))
    baseline = ranking.baseline.get("recall@50")
    if baseline is None:
        return Check(
            "TC-17 pack reproduction",
            False,
            "the npz carries no baseline.recall@50",
        )
    difference = abs(recomputed - float(baseline))
    detail = (
        f"family_new R@50 recomputed {recomputed:.6f} vs the npz's {float(baseline):.6f} "
        f"({len(np.flatnonzero(mask))} documents, |Δ|={difference:.2e})"
    )
    return Check(
        "TC-17 pack reproduction",
        difference <= 1e-6,
        detail,
    )


def check_registry_coverage(results: Path) -> Check:
    """Check that every ``eurlex-*`` metrics folder is in the registry."""
    unlisted = unregistered_runs(results)
    missing = [
        spec.run_id for spec in RUN_SPECS if not (results / spec.run_id / "metrics.json").is_file()
    ]
    problems: list[str] = []
    if unlisted:
        problems.append(f"{len(unlisted)} unregistered runs: {', '.join(unlisted[:5])}")
    if missing:
        problems.append(f"{len(missing)} registry artifacts missing: {', '.join(missing[:5])}")
    detail = (
        "; ".join(problems)
        if problems
        else f"{len(RUN_SPECS)} registered runs; every eurlex-* metrics artifact is listed"
    )
    return Check("metrics registry", not problems, detail)


def sha256_file(path: Path) -> str:
    """One file's sha256.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(sources: Sources, dataset: Dataset, ranking: Ranking) -> tuple[str, ...]:
    """The viewer's provenance block (every source and cache, named)."""
    return (
        f"data: {sources.data_dir}/train.jsonl sha256 "
        f"{sha256_file(sources.data_dir / 'train.jsonl')}",
        f"data: {sources.data_dir}/dev.jsonl sha256 {sha256_file(sources.data_dir / 'dev.jsonl')}",
        f"data: test.jsonl is never opened; label space {len(dataset.labels)} "
        f"(the sorted train+dev union)",
        f"catalogue: {sources.catalog_path} sha256 {sha256_file(sources.catalog_path)}; "
        f"{len(dataset.catalog)} rows",
        f"vocabulary: {sources.vocabulary_path} sha256 {sha256_file(sources.vocabulary_path)}",
        f"TC-17 ranking: {sources.ranking_path} sha256 "
        f"{sha256_file(sources.ranking_path)}; {ranking.ranked.shape} "
        f"{ranking.ranked.dtype}",
        f"caches: {sources.cache}/v1/<model-slug> for {sources.model_id}; "
        f"documents {sources.document_version}, labels {sources.label_v1_version} / "
        f"{sources.label_v2_version}",
        f"rules: full_text = title+header+recitals+body; TF-IDF {TFIDF_CONFIG}; "
        f"k-NN k={KNN_K}, power={VOTE_POWER}, cosine votes; stable ties by label column",
        "note: read-only; metrics are read from results/, per-document scores are "
        "recomputed on demand and may differ in the last decimal from the runs' means.",
    )


def check_all(
    dataset: Dataset,
    ranking: Ranking,
    sources: Sources,
    *,
    seed: int = DEFAULT_SEED,
    expected_documents: int = EXPECTED_DOCUMENTS,
    expected_labels: int = EXPECTED_LABELS,
) -> GuardReport:
    """Run every guard check and return the report (never raises on a mismatch)."""
    seen = None
    sparse = sources.results / "eurlex-label-sparse" / "metrics.json"
    if sparse.is_file():
        try:
            payload = read_json(sparse)
            samples = (payload.get("protocol") or {}).get("samples") or []
            for entry in samples:
                if int(entry.get("seed", -1)) == seed:
                    seen = int(entry.get("seen_labels", 0))
                    break
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            seen = None
    checks = (
        check_ranking(
            ranking,
            label_count=len(dataset.labels),
            expected_documents=expected_documents,
            expected_labels=expected_labels,
        ),
        check_catalog(dataset),
        check_seed_sample(dataset, seed=seed, expected_seen=seen),
        check_pack_reproduction(dataset, ranking),
        check_registry_coverage(sources.results),
    )
    return GuardReport(
        checks=checks,
        provenance=build_provenance(sources, dataset, ranking),
    )
