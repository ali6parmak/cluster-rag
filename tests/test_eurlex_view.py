"""Tests for the EURLEX-4K viewer's data layer.

Everything here is a **tiny synthetic** example: four labels, three train and
two dev documents, a fake cache of three-dimensional unit vectors written with
``tmp_path``, and a hand-built metrics dict. No real artifact is read, no model
is loaded, no network is touched and Streamlit is never imported (the layer is
pure).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
LABELS = ("A", "B", "C", "D")


def _load_module() -> Any:
    """Load the data layer from the repository's tools folder."""
    name = "eurlex_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / "eurlex_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ev() -> Any:
    """The data layer module."""
    return _load_module()


def _doc(ev: Any, celex: str, title: str, labels: tuple[str, ...]) -> Any:
    """One synthetic document with the track's fields."""
    body = f"{title} body text"
    return ev.Document(
        celex_id=celex,
        document_type="Regulation",
        title=title,
        header=f"HEADER {title}",
        recitals=f"recitals for {title}",
        main_body=(body,),
        labels=labels,
    )


def _dataset(ev: Any) -> Any:
    """A four-label dataset with three train and two dev documents."""
    train = (
        _doc(ev, "train-1", "alpha beta", ("A",)),
        _doc(ev, "train-2", "gamma delta", ("B",)),
        _doc(ev, "train-3", "alpha epsilon", ("A", "C")),
    )
    dev = (
        _doc(ev, "dev-1", "alpha beta", ("A", "B")),
        _doc(ev, "dev-2", "gamma delta", ("C",)),
    )
    label_index = {label: index for index, label in enumerate(LABELS)}
    catalog = {
        label: ev.CatalogEntry(
            label_id=label,
            name=label.lower(),
            parents=("P",) if label == "B" else (),
            ancestors=(),
            parent_names=("parent",) if label == "B" else (),
            definition="a definition" if label == "B" else None,
            definition_kind="definition" if label == "B" else None,
        )
        for label in LABELS
    }
    vocabulary = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
    support = {"A": 2, "B": 1, "C": 1, "D": 0}
    bucket = {label: ev.support_bucket(support[label]) for label in LABELS}
    return ev.Dataset(
        train=train,
        dev=dev,
        labels=LABELS,
        label_index=label_index,
        catalog=catalog,
        vocabulary=vocabulary,
        gold=(frozenset({label_index["A"], label_index["B"]}), frozenset({label_index["C"]})),
        train_gold=(
            frozenset({label_index["A"]}),
            frozenset({label_index["B"]}),
            frozenset({label_index["A"], label_index["C"]}),
        ),
        support=support,
        bucket=bucket,
    )


def _sources(ev: Any, tmp_path: Path) -> Any:
    """The viewer's sources, pointed at a fake cache under ``tmp_path``."""
    return ev.Sources(
        results=tmp_path / "results",
        cache=tmp_path / "cache",
        data_dir=tmp_path / "data",
        catalog_path=tmp_path / "catalog.jsonl",
        vocabulary_path=tmp_path / "vocab.jsonl",
        ranking_path=tmp_path / "ranked.npz",
    )


def _write_vector(ev: Any, sources: Any, text: str, vector: list[float]) -> None:
    """Write one fake cache entry with the viewer's own key rule."""
    path = ev.document_vector_path(sources, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vector, dtype=np.float32))


def _write_label_vectors(ev: Any, sources: Any, dataset: Any, variant: str) -> None:
    """Write one fake vector per label for the given text variant."""
    version = sources.label_v1_version if variant == "title" else sources.label_v2_version
    for index, label in enumerate(dataset.labels):
        vector = np.zeros(3, dtype=np.float32)
        vector[index % 3] = 1.0
        path = ev.label_vector_path(sources, ev.label_text(dataset, label, variant), version)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)


# --- the overlap summariser -------------------------------------------------


def test_overlap_summary_is_the_hand_computed_partition(ev: Any) -> None:
    """Shared = gold ∩ neighbours; the two differences are the rest."""
    summary = ev.overlap_summary(
        ["A", "B", "C"],
        [["A", "B", "D"], ["B", "E"]],
    )
    assert summary.shared == ("A", "B")
    assert summary.document_only == ("C",)
    assert summary.neighbours_only == ("D", "E")
    assert (summary.n_shared, summary.n_document_only, summary.n_neighbours_only) == (2, 1, 2)


def test_overlap_summary_handles_empty_sides(ev: Any) -> None:
    """No neighbours, no overlap; no gold, every neighbour label is only theirs."""
    assert ev.overlap_summary(["A"], []).shared == ()
    assert ev.overlap_summary(["A"], []).document_only == ("A",)
    assert ev.overlap_summary([], [["A"]]).neighbours_only == ("A",)


# --- the nearest neighbours on a fake cache ---------------------------------


def test_embedding_neighbours_on_a_tiny_fake_cache(ev: Any, tmp_path: Path) -> None:
    """The three fake vectors order train-1, train-3, train-2 by cosine."""
    dataset = _dataset(ev)
    sources = _sources(ev, tmp_path)
    vectors = {
        "train-1": [1.0, 0.0, 0.0],
        "train-2": [0.0, 1.0, 0.0],
        "train-3": [0.8, 0.6, 0.0],
        "dev-1": [1.0, 0.0, 0.0],
        "dev-2": [0.0, 0.0, 1.0],
    }
    for doc in (*dataset.train, *dataset.dev):
        _write_vector(ev, sources, doc.text, vectors[doc.celex_id])
    loaded = ev.load_document_vectors(sources, dataset.train)
    assert loaded.shape == (3, 3)
    assert loaded.dtype == np.float32
    neighbours = ev.embedding_neighbours(
        ev.load_document_vectors(sources, (dataset.dev[0],))[0],
        loaded,
        dataset.train,
        k=2,
        gold_labels=("A", "B"),
    )
    assert [row.doc_id for row in neighbours] == ["train-1", "train-3"]
    assert neighbours[0].score == pytest.approx(1.0)
    assert neighbours[1].score == pytest.approx(0.8)
    assert neighbours[0].shared == ("A",)
    assert neighbours[1].shared == ("A",)
    assert neighbours[1].rank == 2
    assert "alpha beta body text" in neighbours[0].text
    assert "alpha epsilon body text" in neighbours[1].text
    assert [section.kind for section in neighbours[0].sections] == [
        "title",
        "header",
        "recitals",
        "main_body",
    ]
    assert neighbours[0].sections[0].text == "alpha beta"
    assert "alpha beta body text" in neighbours[0].sections[3].text


def test_a_missing_cached_vector_says_so_and_never_downloads(ev: Any, tmp_path: Path) -> None:
    """A missing cache entry raises the layer's own clear error."""
    dataset = _dataset(ev)
    sources = _sources(ev, tmp_path)
    _write_vector(ev, sources, dataset.train[0].text, [1.0, 0.0, 0.0])
    with pytest.raises(ev.MissingVectorError, match="no cached vector for document train-2"):
        ev.load_document_vectors(sources, dataset.train)


# --- the metrics registry ---------------------------------------------------


def _candidate_pack_payload() -> dict[str, Any]:
    """A tiny two-slice standard ranking artifact."""
    return {
        "run_id": "eurlex-candidate-pack",
        "change_set": "TC-14 (test)",
        "cost": {"seconds_total": 12.5, "llm_calls": 0},
        "methods": {
            "embedding_knn": {
                "kind": "knn",
                "arm": "embedding k-NN (reference)",
                "slices": {
                    "all_dev": {
                        "primary": {
                            "recall@50": 0.5,
                            "r_precision": 0.25,
                            "ndcg@10": 0.3,
                        },
                        "diagnostics": {"label_recall@50": {"unseen": {"recall": 0.0}}},
                    },
                    "family_new": {
                        "primary": {
                            "recall@50": 0.4,
                            "r_precision": 0.2,
                            "ndcg@10": 0.25,
                        },
                        "diagnostics": {"label_recall@50": {"unseen": {"recall": 0.125}}},
                    },
                },
            }
        },
    }


def test_the_registry_extracts_a_standard_run(ev: Any) -> None:
    """The standard extractor reads the artifact's own arm and slice keys."""
    artifact = Path("results/eurlex-candidate-pack/metrics.json")
    rows = ev.extract_metrics("eurlex-candidate-pack", _candidate_pack_payload(), artifact)
    assert len(rows) == 2
    by_slice = {row.slice: row for row in rows}
    assert by_slice["all_dev"].recall50 == pytest.approx(0.5)
    assert by_slice["family_new"].r_precision == pytest.approx(0.2)
    assert by_slice["family_new"].unseen_recall50 == pytest.approx(0.125)
    assert by_slice["all_dev"].method == "embedding_knn"
    assert by_slice["all_dev"].arm == "embedding k-NN (reference)"
    assert by_slice["all_dev"].cost_seconds == pytest.approx(12.5)
    assert by_slice["all_dev"].artifact == str(artifact)
    assert by_slice["all_dev"].regime == ev.REGIME_45K


def test_the_registry_extracts_the_sparse_pooled_means(ev: Any) -> None:
    """TC-21's pooled ``slice/metric -> {mean, sd}`` document is read as a mean."""
    payload = {
        "run_id": "eurlex-label-sparse",
        "change_set": "TC-21 (test)",
        "cost": {"seconds_total": 3.0},
        "arms": {
            "lexical_knn": {
                "arm": "TF-IDF k-NN on the 240",
                "per_seed": [],
                "pooled": {
                    "all_dev/recall@50": {"mean": 0.52, "sd": 0.01},
                    "all_dev/r_precision": {"mean": 0.24, "sd": 0.01},
                    "all_dev/ndcg@10": {"mean": 0.31, "sd": 0.01},
                    "all_dev/label_recall@50/unseen": {"mean": 0.0, "sd": 0.0},
                    "family_new/recall@50": {"mean": 0.44, "sd": 0.01},
                    "family_new/r_precision": {"mean": 0.17, "sd": 0.01},
                    "family_new/ndcg@10": {"mean": 0.23, "sd": 0.01},
                    "family_new/label_recall@50/unseen": {"mean": 0.0, "sd": 0.0},
                    "family_new_clean/recall@50": {"mean": 0.43, "sd": 0.01},
                    "family_new_clean/r_precision": {"mean": 0.16, "sd": 0.01},
                    "family_new_clean/ndcg@10": {"mean": 0.22, "sd": 0.01},
                    "family_new_clean/label_recall@50/unseen": {"mean": 0.0, "sd": 0.0},
                },
            }
        },
    }
    rows = ev.extract_metrics(
        "eurlex-label-sparse",
        payload,
        Path("results/eurlex-label-sparse/metrics.json"),
    )
    assert len(rows) == 3
    assert {row.regime for row in rows} == {ev.REGIME_240}
    by_slice = {row.slice: row for row in rows}
    assert by_slice["all_dev"].recall50 == pytest.approx(0.52)
    assert by_slice["all_dev"].unseen_recall50 == pytest.approx(0.0)
    assert by_slice["family_new"].r_precision == pytest.approx(0.17)


def test_an_unregistered_run_is_refused(ev: Any) -> None:
    """The registry is explicit: an unknown id never gets a generic walk."""
    with pytest.raises(KeyError, match="not in the registry"):
        ev.extract_metrics("eurlex-not-a-run", {}, Path("x"))


# --- the guard --------------------------------------------------------------


def test_the_guard_refuses_a_wrong_ranking_shape(ev: Any) -> None:
    """A (5, 4) matrix is not the TC-17 pack's (6, 4,197)."""
    ranking = ev.Ranking(
        ranked=np.zeros((5, 4), dtype=np.int32),
        baseline={},
        deltas={},
        seconds=0.0,
    )
    check = ev.check_ranking(
        ranking,
        label_count=4,
        expected_documents=6,
        expected_labels=4,
    )
    assert not check.ok
    assert "shape is (5, 4)" in check.detail


def test_the_guard_accepts_a_permutation(ev: Any) -> None:
    """A tiny permutation matrix passes the ranking check."""
    ranking = ev.Ranking(
        ranked=np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32),
        baseline={},
        deltas={},
        seconds=0.0,
    )
    check = ev.check_ranking(
        ranking,
        label_count=4,
        expected_documents=2,
        expected_labels=4,
    )
    assert check.ok, check.detail


def test_the_guard_refuses_a_repeated_label_in_a_row(ev: Any) -> None:
    """A row that repeats a label is not a ranking of the 4,197 labels."""
    ranking = ev.Ranking(
        ranked=np.asarray([[0, 0, 2, 3]], dtype=np.int32),
        baseline={},
        deltas={},
        seconds=0.0,
    )
    check = ev.check_ranking(
        ranking,
        label_count=4,
        expected_documents=1,
        expected_labels=4,
    )
    assert not check.ok
    assert "not a permutation" in check.detail


def test_the_seed_rule_is_reproducible_and_the_sample_is_sorted(ev: Any) -> None:
    """TC-21's rule: ``default_rng(seed).choice(n, 240, replace=False)``, sorted."""
    sample = ev.seed_sample(42)
    assert sample.shape == (240,)
    assert len({int(row) for row in sample}) == 240
    assert np.all(sample[:-1] <= sample[1:])
    assert sample[0] == 330 and sample[1] == 878 and sample[-1] == 44746
    assert np.array_equal(sample, ev.seed_sample(42))


def test_the_seed_guard_checks_the_sample_against_the_dataset(ev: Any) -> None:
    """The sample check passes on the rule and refuses a wrong seen count."""
    dataset = _dataset(ev)
    sample = ev.seed_sample(42, n_sources=len(dataset.train), n_labelled=2)
    seen = {label for row in sample for label in dataset.train[int(row)].labels}
    ok = ev.check_seed_sample(
        dataset,
        seed=42,
        n_sources=len(dataset.train),
        n_labelled=2,
        expected_seen=len(seen),
    )
    assert ok.ok, ok.detail
    refused = ev.check_seed_sample(
        dataset,
        seed=42,
        n_sources=len(dataset.train),
        n_labelled=2,
        expected_seen=len(seen) + 1,
    )
    assert not refused.ok
    assert "the artifact says" in refused.detail


# --- the label-text builders ------------------------------------------------


def test_the_label_texts_are_the_pre_registered_variants(ev: Any) -> None:
    """v1 is the title; v2 is title + de-duplicated parent names; v3 adds the note."""
    dataset = _dataset(ev)
    assert ev.label_text(dataset, "A", "title") == "alpha"
    assert ev.label_text(dataset, "B", "title_parents") == "beta | parent"
    assert ev.label_text(dataset, "A", "title_parents") == "alpha"
    assert ev.label_text(dataset, "B", "title_parents_definition") == "beta | parent | a definition"
    with pytest.raises(ValueError, match="unknown label-text variant"):
        ev.label_text(dataset, "A", "not-a-variant")


# --- the assembled one-document view ---------------------------------------


def test_the_document_view_marks_gold_and_builds_the_matrix(ev: Any, tmp_path: Path) -> None:
    """The view's rankings, overlap and label x source matrix agree by hand."""
    dataset = _dataset(ev)
    sources = _sources(ev, tmp_path)
    vectors = {
        "train-1": [1.0, 0.0, 0.0],
        "train-2": [0.0, 1.0, 0.0],
        "train-3": [0.8, 0.6, 0.0],
        "dev-1": [1.0, 0.0, 0.0],
        "dev-2": [0.0, 0.0, 1.0],
    }
    for doc in (*dataset.train, *dataset.dev):
        _write_vector(ev, sources, doc.text, vectors[doc.celex_id])
    _write_label_vectors(ev, sources, dataset, "title")
    _write_label_vectors(ev, sources, dataset, "title_parents")
    vectors_train = ev.load_document_vectors(sources, dataset.train)
    index = ev.build_search_index(dataset, sources, vectors=vectors_train)
    ranking = ev.Ranking(
        ranked=np.asarray([[0, 2, 1, 3], [2, 0, 3, 1]], dtype=np.int32),
        baseline={},
        deltas={},
        seconds=0.0,
    )
    view = ev.build_document_view(
        dataset,
        ranking,
        index,
        document_row=0,
        query_vector=ev.load_document_vectors(sources, (dataset.dev[0],))[0],
        label_vectors_title=ev.load_label_vectors(sources, dataset, "title"),
        label_vectors_parents=ev.load_label_vectors(sources, dataset, "title_parents"),
    )
    assert [tag.label for tag in view.gold] == ["A", "B"]
    assert [method.key for method in view.methods] == [
        "pack",
        "lexical_knn",
        "embedding_knn",
        "label_text_title",
        "label_text_title_parents",
    ]
    assert view.methods[0].rows[0].label == "A"
    assert view.methods[0].rows[0].gold
    assert view.methods[0].rows[0].pack_rank == 1
    assert view.overlap.shared  # train-1 carries A
    assert {row.label for row in view.matrix} >= {"A", "B"}
    assert view.matrix[0].gold  # gold labels are sorted first
    signals = ev.label_signals(dataset, view, "A", index_support=2)
    assert signals.pack_rank == 1
    assert signals.bucket == "tail"
    assert signals.ranks["lexical_knn"] >= 1
    assert signals.best_cosine["embedding"] == pytest.approx(1.0)
    assert signals.index_bucket == "2-4"


# --- the errors view and the curve -----------------------------------------


def test_pack_misses_counts_the_gold_labels_outside_the_top_k(ev: Any) -> None:
    """dev-1's B is at rank 4, so a depth-2 rank misses it; dev-2 misses none."""
    dataset = _dataset(ev)
    ranking = ev.Ranking(
        ranked=np.asarray([[0, 2, 3, 1], [2, 3, 0, 1]], dtype=np.int32),
        baseline={},
        deltas={},
        seconds=0.0,
    )
    misses = ev.pack_misses(dataset, ranking, depth=2)
    assert misses.n_gold_assignments == 3
    assert misses.n_missed == 1
    assert misses.documents[0].doc_id == "dev-1"
    assert misses.documents[0].missed == ("B",)
    assert misses.documents[0].missed_titles == ("beta",)
    assert misses.labels[0].label == "B"
    assert misses.labels[0].n_gold == 1


def test_the_curve_keeps_the_four_retrieval_families(ev: Any) -> None:
    """Unknown families are not plotted; the 240/45k sizes are the documented ones."""

    def row(method: str, regime: str) -> Any:
        return ev.MetricRow(
            run_id="run",
            change_set="TC",
            method=method,
            arm=method,
            regime=regime,
            slice="family_new",
            recall50=0.5,
            r_precision=0.2,
            ndcg10=0.3,
            unseen_recall50=None,
            micro_f1=None,
            cost_seconds=1.0,
            artifact="x",
        )

    points = ev.label_sparse_curve(
        [
            row("lexical_knn", ev.REGIME_240),
            row("candidate_pack", ev.REGIME_45K),
            row("dictionary", ev.REGIME_45K),
        ]
    )
    assert [(point.family, point.labelled_documents) for point in points] == [
        ("lexical k-NN", 240),
        ("pack", 45000),
    ]


def test_label_neighbours_are_the_exact_cosines(ev: Any) -> None:
    """A's nearest labels by label-text cosine are B, C, D; self is excluded."""
    dataset = _dataset(ev)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rows = ev.label_neighbours(dataset, vectors, "A", k=3, seen={"C"})
    assert [row.label for row in rows] == ["B", "C", "D"]
    assert [row.rank for row in rows] == [1, 2, 3]
    assert rows[0].cosine == pytest.approx(0.8)
    assert rows[0].title == "beta"
    assert rows[0].n_dev_gold == 1
    assert rows[0].seen is False
    assert rows[1].seen is True
    with pytest.raises(KeyError, match="not in the label space"):
        ev.label_neighbours(dataset, vectors, "Z")


def test_label_evidence_centroids_average_the_documents(ev: Any) -> None:
    """A's centroid is the normalized mean of train-1 and train-3; D has none."""
    dataset = _dataset(ev)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
        dtype=np.float32,
    )
    evidence = ev.label_evidence_centroids(dataset, vectors=vectors)
    assert evidence.support.tolist() == [2, 1, 1, 0]
    assert evidence.has_evidence.tolist() == [True, True, True, False]
    assert evidence.vectors[0].tolist() == pytest.approx([0.948683, 0.316228, 0.0])
    assert np.linalg.norm(evidence.vectors[0]) == pytest.approx(1.0)
    assert evidence.vectors[3].tolist() == [0.0, 0.0, 0.0]


def test_label_neighbours_can_be_restricted_to_eligible_labels(ev: Any) -> None:
    """The evidence map only ranks labels that have a centroid."""
    dataset = _dataset(ev)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rows = ev.label_neighbours(dataset, vectors, "A", k=3, seen={"A"}, eligible={"A", "B"})
    assert [row.label for row in rows] == ["B"]
    with pytest.raises(ValueError, match="no vector in this representation"):
        ev.label_neighbours(dataset, vectors, "C", eligible={"A", "B"})
