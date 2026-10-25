"""Tests for the text-processing track's Stage-0 analysis (`analyze-omct`).

Contracts under test (TEXT_PROCESSING_PLAN.md section 3, Stage 0.1):

- the same-document grid: the cell counts, the judged cells and the unjudged
  pool the control is drawn from;
- the control set: a seeded, **proportional**, per-document sample of the
  unjudged cells that never contains a judged cell and that the same seed
  reproduces;
- the rank percentile: the position from the top of the paragraph's own
  same-document candidate list (0 = best);
- the artifact's shape: the sections Stage 0's claims may cite;
- the CLI end-to-end with a fake embedder: the run folder, the frozen
  config, the artifact and the track's table row.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from cluster_rag import cli
from cluster_rag.cli import main
from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    EmbedderConfig,
    PreprocessConfig,
)
from cluster_rag.omct_analysis import (
    SideEncoding,
    build_dataset_view,
    compare_arms,
    document_code,
    score_model_arms,
    side_similarities,
)
from cluster_rag.pairs import load_pairs
from cluster_rag.preprocess import build_preprocess_spec
from cluster_rag.text_processors.aggregation import parse_aggregation
from cluster_rag.text_processors.chunkers import FullTextChunker
from cluster_rag.text_processors.control import (
    SAMPLING_RULE_PROPORTIONAL,
    DocumentGrid,
    allocate_quotas,
    rank_percentiles,
    sample_control,
)
from fake_embedder import FakeEmbedder

ANALYSIS_CONFIG_EXTRA = """
analysis:
  embedders:
    - name: fake-model
      embed:
        backend: fake
        model_id: fake-embedder
        batch_size: 4
        prefixes:
          query: "query: "
          document: "passage: "
  control_size: 3
  sampling_rule: proportional
  question_transform: strip_block
  chunker: full_text
"""

ANALYSIS_ARMS_CONFIG_EXTRA = """
analysis:
  embedders:
    - name: fake-model
      embed:
        backend: fake
        model_id: fake-embedder
        batch_size: 4
        prefixes:
          query: "query: "
          document: "passage: "
  control_size: 3
  sampling_rule: proportional
  question_transform: strip_block
  chunker:
    name: atoms
    params:
      split_clauses: false
  aggregations: [max, "top_k_mean:2", mean]
  question_transforms: [keep]
  bootstrap:
    resamples: 50
    seed: 7
"""


@pytest.fixture()
def view(sample_pairs: Path):
    """The model-independent view of the pairwise fixture."""
    records = load_pairs(sample_pairs).records
    return build_dataset_view(records, control_size=3, seed=42)


def test_grid_math_counts_cells_judged_and_unjudged(view) -> None:
    """The grid holds every same-document cell; the judged ones are marked."""
    grids = {grid.document: grid for grid in view.grids}
    assert set(grids) == {"Doc A", "Doc B", "Doc C"}
    # Doc A: 2 paragraphs x 1 question, both judged
    assert grids["Doc A"].n_cells == 2
    assert grids["Doc A"].n_judged == 2
    assert grids["Doc A"].n_unjudged == 0
    # Doc B: 3 paragraphs x 2 questions, 3 judged
    assert grids["Doc B"].n_cells == 6
    assert grids["Doc B"].n_judged == 3
    assert grids["Doc B"].n_unjudged == 3
    # Doc C: 1 paragraph x 1 question, judged
    assert grids["Doc C"].n_cells == 1
    assert grids["Doc C"].n_unjudged == 0

    assert view.pooled["grid_cells"] == 9
    assert view.pooled["judged_cells"] == 6
    assert view.pooled["unjudged_cells"] == 3
    assert view.pooled["n_paragraphs"] == 6
    # the fixture carries four distinct question texts (two of them numbered 10)
    assert view.pooled["n_questions"] == 4


def test_strata_report_base_rates_per_document(view) -> None:
    """Every document reports its own base rate next to the pooled one."""
    strata = {stratum["document"]: stratum for stratum in view.strata}
    assert strata["Doc A"]["n_pairs"] == 2
    assert strata["Doc A"]["positive_share"] == 0.5
    assert strata["Doc B"]["n_pairs"] == 3
    assert strata["Doc B"]["positive_share"] == pytest.approx(1 / 3)
    assert strata["Doc B"]["n_paragraphs_without_positive"] == 2  # B2 and B3
    assert view.pooled["positive_share"] == pytest.approx(0.5)
    assert view.pooled["macro_positive_share"] == pytest.approx((0.5 + 1 / 3 + 1.0) / 3)


def test_control_quotas_are_proportional_to_the_unjudged_cells() -> None:
    """The quota split follows the unjudged cells (largest-remainder method)."""
    quotas = allocate_quotas({"a": 10, "b": 30, "c": 60}, 100)
    assert quotas == {"a": 10, "b": 30, "c": 60}
    # 5 slots over 3 strata of 1/1/1 -> the leftover goes to the first name
    assert allocate_quotas({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    # a stratum never gets more slots than it has cells
    assert allocate_quotas({"a": 1, "b": 99}, 10) == {"a": 0, "b": 10}
    with pytest.raises(ValueError, match="cannot draw"):
        allocate_quotas({"a": 1}, 5)


def test_control_sampling_is_deterministic_and_seed_dependent() -> None:
    """The same seed gives the same cells; another seed gives other cells."""
    grid = DocumentGrid(
        document="Doc",
        paragraph_rows=(0, 1, 2, 3, 4),
        question_cols=(0, 1),
        judged=frozenset({(0, 0), (1, 1)}),
    )
    first = sample_control([grid], size=3, seed=42)
    second = sample_control([grid], size=3, seed=42)
    other = sample_control([grid], size=3, seed=7)
    assert first.cells == second.cells
    assert set(first.cells) != set(other.cells)
    assert len(first.cells) == 3
    # the drawn cells are unjudged cells of the grid
    for document, row, col in first.cells:
        assert document == "Doc"
        assert (row, col) not in grid.judged
        assert row in grid.paragraph_rows and col in grid.question_cols


def test_control_sampling_on_the_fixture_uses_the_whole_unjudged_pool(view) -> None:
    """With three unjudged cells and a quota of three, the pool is drawn whole."""
    assert view.control.n_cells == 3
    assert view.control.quotas == {"Doc A": 0, "Doc B": 3, "Doc C": 0}


def test_control_never_contains_a_judged_cell(view) -> None:
    """The control is the *unjudged* part of the grid (subset B, not A)."""
    judged = {(grid.document, row, col) for grid in view.grids for row, col in grid.judged}
    for document, row, col in view.control.cells:
        assert (document, row, col) not in judged


def test_control_records_its_provenance(view) -> None:
    """The artifact's control block names the seed, the rule and the quotas."""
    described = view.control.describe()
    assert described["size_requested"] == 3
    assert described["size_drawn"] == 3
    assert described["seed"] == 42
    assert described["sampling_rule"] == SAMPLING_RULE_PROPORTIONAL
    assert described["per_document"]["Doc B"]["quota"] == 3
    assert described["per_document"]["Doc B"]["n_drawn"] == 3


def test_rank_percentile_is_the_position_from_the_top() -> None:
    """0 = the best candidate of the paragraph's own candidate list."""
    similarities = np.array(
        [
            [0.9, 0.1, 0.5],  # row 0: best candidate is column 0
            [0.2, 0.8, 0.4],  # row 1: best candidate is column 1
        ]
    )
    candidate_columns = [(0, 1, 2), (0, 1, 2)]
    percentiles = rank_percentiles(
        similarities,
        rows=[0, 0, 0, 1],
        columns=[0, 2, 1, 1],
        candidate_columns=candidate_columns,
    )
    assert percentiles[0] == 0.0  # the top candidate of row 0
    assert percentiles[1] == pytest.approx(1 / 3)  # the middle candidate of row 0
    assert percentiles[2] == pytest.approx(2 / 3)  # the worst candidate of row 0
    assert percentiles[3] == 0.0  # the top candidate of row 1


def test_document_code_uses_the_pair_id_convention() -> None:
    """The cell ids follow the file's own convention; a fallback is a slug."""
    assert document_code("Doc A", ["NU-10-Q2", "NU-11-Q3"]) == "NU"
    assert document_code("Doc A", ["P-1", "P-2"]) == "Doc_A"
    assert document_code("Doc A", ["NU-10-Q2", "CIDH-1-Q1"]) == "Doc_A"


def test_score_model_produces_the_artifact_blocks(view) -> None:
    """The per-model block carries the pairs, the control cells and the numbers."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    report = score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
    )
    assert report["name"] == "fake-model"
    assert len(report["judged_pairs"]) == 6
    assert len(report["control_cells"]) == 3
    assert report["sides"]["paragraph"]["kind"] == "document"
    assert report["sides"]["question"]["kind"] == "query"
    assert report["sides"]["paragraph"]["cache_version"] == "raw-v1"
    assert report["sides"]["paragraph"]["chunker"] == "full_text-v1"
    assert set(report["score_distributions"]) >= {"label1", "label0", "control"}
    assert set(report["negative_enrichment"]["rank_percentile"]) == {
        "label1",
        "label0",
        "control",
    }
    assert report["pair_auc"]["overall"] is not None
    assert report["top1_candidate_provenance"]["n_paragraphs"] == 6
    assert report["arms"] == {} and report["comparisons"] == {}
    for pair in report["judged_pairs"]:
        assert "score" in pair and "rank_percentile" in pair
    for cell in report["control_cells"]:
        assert cell["cell_id"].count("-") >= 1


def test_analyze_omct_command_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """The command writes the artifact, the frozen config and the table row."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-test",
        data_external=sample_pairs,
        extra=ANALYSIS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    # the omct- dataset prefix is added by the command itself
    results_dir = tmp_path / "results" / "omct-analyze-test"
    assert (results_dir / "config.yaml").is_file()
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    assert payload["run_id"] == "omct-analyze-test"
    assert payload["command"] == "analyze-omct"
    assert payload["seed"] == 42
    assert payload["analysis"]["control"]["size_drawn"] == 3
    assert payload["analysis"]["control"]["n_judged_cells_excluded"] == 6
    assert len(payload["analysis"]["control"]["excluded_judged_cells"]) == 6
    assert payload["analysis"]["grid"]["n_cells"] == 9
    assert len(payload["models"]) == 1
    model = payload["models"][0]
    assert len(model["judged_pairs"]) == 6
    assert len(model["control_cells"]) == 3
    assert model["sides"]["paragraph"]["cache_misses"] == 6  # cold run: 6 paragraphs
    assert model["sides"]["question"]["cache_misses"] == 4  # 4 question texts

    out = capsys.readouterr().out
    assert "OMCT analysis (Stage 0): omct-analyze-test" in out

    # a warm rerun encodes nothing: every vector comes from the cache
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    model = payload["models"][0]
    assert model["sides"]["paragraph"]["cache_hits"] == 6
    assert model["sides"]["paragraph"]["cache_misses"] == 0
    assert model["sides"]["question"]["cache_hits"] == 4
    assert model["sides"]["question"]["cache_misses"] == 0


def test_analyze_omct_appends_the_track_table_row(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """The row lands in the track's own table file (command-derived)."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="analyze-table",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=ANALYSIS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## OMCT — analyze-omct" in text
    data_rows = [line for line in text.splitlines() if line.startswith("| omct-analyze-table")]
    assert len(data_rows) == 1
    assert "fake-embedder" in data_rows[0]


def test_analyze_omct_requires_the_analysis_section(sample_pairs, write_config, caplog) -> None:
    """Without an analysis: section the command fails with a clear message."""
    config_file = write_config(sample_pairs, run_id="no-analysis", data_external=sample_pairs)
    with caplog.at_level(logging.ERROR):
        assert main(["analyze-omct", "--config", str(config_file)]) == 1
    assert any("analysis: section" in record.message for record in caplog.records)


def test_analysis_config_rejects_an_unknown_chunker() -> None:
    """The chunker name is validated at config time (fail before any run)."""
    with pytest.raises(ValueError, match="Unknown chunker"):
        AnalysisConfig(
            embedders=(
                AnalysisEmbedderConfig(name="m", embed=EmbedderConfig(backend="hf", model_id="m")),
            ),
            chunker="does-not-exist",
        )


def test_analysis_config_accepts_a_bare_chunker_name() -> None:
    """A bare string is read as the chunker's name (the Stage-0 configs load)."""
    config = AnalysisConfig(
        embedders=(
            AnalysisEmbedderConfig(name="m", embed=EmbedderConfig(backend="hf", model_id="m")),
        ),
        chunker="atoms",
    )
    assert config.chunker.name == "atoms"
    assert config.chunker.params == {}


def test_analysis_config_rejects_unknown_arms() -> None:
    """The arms are validated at config time (fail before any run)."""
    embedders = (
        AnalysisEmbedderConfig(name="m", embed=EmbedderConfig(backend="hf", model_id="m")),
    )
    with pytest.raises(ValueError, match="Unknown aggregation"):
        AnalysisConfig(embedders=embedders, aggregations=("median",))
    with pytest.raises(ValueError, match="need a chunker that splits"):
        AnalysisConfig(embedders=embedders, aggregations=("max",))
    with pytest.raises(ValueError, match="must not repeat the baseline transform"):
        AnalysisConfig(embedders=embedders, question_transforms=("strip_block",))
    with pytest.raises(ValueError, match="Unknown parameter"):
        AnalysisConfig(embedders=embedders, chunker={"name": "atoms", "params": {"nope": 1}})


def test_side_similarities_folds_the_paragraph_chunks() -> None:
    """A chunked paragraph side is folded by the aggregation rule."""
    paragraph_side = SideEncoding(
        texts=("p", "q"),
        chunk_lists=(("p1", "p2"), ("q1",)),
        vectors=np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]),
    )
    question_side = SideEncoding(
        texts=("a",),
        chunk_lists=(("a",),),
        vectors=np.array([[1.0, 0.0]]),
    )
    maximum = side_similarities(paragraph_side, question_side, parse_aggregation("max"))
    assert maximum.shape == (2, 1)
    assert maximum[0, 0] == pytest.approx(1.0)
    mean = side_similarities(paragraph_side, question_side, parse_aggregation("mean"))
    assert mean[0, 0] == pytest.approx(0.5)
    top1 = side_similarities(paragraph_side, question_side, parse_aggregation("top_k_mean:1"))
    assert top1[0, 0] == pytest.approx(1.0)


def test_side_similarities_refuses_a_chunked_question_side() -> None:
    """The question-side chunker is out of scope: a chunked side is refused."""
    paragraph_side = SideEncoding(
        texts=("p",), chunk_lists=(("p1",),), vectors=np.array([[1.0, 0.0]])
    )
    question_side = SideEncoding(
        texts=("a",), chunk_lists=(("a1", "a2"),), vectors=np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    with pytest.raises(ValueError, match="one chunk per question"):
        side_similarities(paragraph_side, question_side, parse_aggregation("max"))


def test_compare_arms_is_paired_and_carries_the_control_side(view) -> None:
    """The comparison: the paired statistic, the control shift and the de-shifted AUC."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    report = score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
    )
    # the same arm compared with itself: no difference anywhere
    comparison = compare_arms(
        view,
        arm_name="same",
        reference_similarities=np.zeros((6, 4)),
        arm_similarities=np.zeros((6, 4)),
        resamples=20,
        seed=1,
    )
    assert comparison["paired_auc"]["delta_auc"] == 0.0
    assert comparison["control_shift"]["mean_delta"] == 0.0
    assert comparison["floor_gap"]["delta"] == 0.0
    assert comparison["deshifted_auc"]["overall"] is not None
    assert set(comparison["deshifted_auc"]["per_document_shift"]) == {"Doc A", "Doc B", "Doc C"}
    assert report["pair_auc"]["overall"] is not None


def test_analyze_omct_arms_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """The arms land in the artifact, paired with the baseline arm on the same cells."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-arms",
        data_external=sample_pairs,
        extra=ANALYSIS_ARMS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-arms"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    model = payload["models"][0]
    arms = model["arms"]
    # the three aggregations of the atoms chunker + the question-transform arm
    assert set(arms) == {"atoms_max", "atoms_top_k_mean_2", "atoms_mean", "full_text_keep"}
    assert payload["analysis"]["aggregations"] == ["max", "top_k_mean:2", "mean"]
    assert payload["analysis"]["question_transforms"] == ["keep"]
    assert payload["analysis"]["bootstrap"] == {"resamples": 50, "seed": 7, "confidence": 0.95}

    # the pairing is by construction: every arm measures the same cells
    baseline_cells = [cell["cell_id"] for cell in model["control_cells"]]
    for arm in arms.values():
        assert [cell["cell_id"] for cell in arm["control_cells"]] == baseline_cells
        assert [pair["pair_id"] for pair in arm["judged_pairs"]] == [
            pair["pair_id"] for pair in model["judged_pairs"]
        ]

    # every arm carries its comparison against the baseline
    comparisons = model["comparisons"]
    assert set(comparisons) == set(arms)
    for name, comparison in comparisons.items():
        paired = comparison["paired_auc"]
        assert paired["n_resamples"] == 50 and paired["seed"] == 7
        assert paired["ci_low"] <= paired["delta_auc"] <= paired["ci_high"]
        assert "control_shift" in comparison and "deshifted_auc" in comparison
        assert comparison["reference_arm"] == "full_text"
        assert name in arms

    out = capsys.readouterr().out
    assert "OMCT analysis (Stage 0): omct-analyze-arms" in out
    assert "dAUC" in out


def test_analyze_omct_arms_table_rows_use_the_arms_columns(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """A run with arms writes the Stage-1 columns (one row per model and arm)."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="analyze-arms-table",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=ANALYSIS_ARMS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    assert "## OMCT — analyze-omct" in text
    data_rows = [line for line in text.splitlines() if line.startswith("| omct-analyze-arms-table")]
    assert len(data_rows) == 5  # the baseline arm + four arms
    assert any("| atoms_max |" in row for row in data_rows)
    assert any("| full_text |" in row for row in data_rows)
    # the baseline row carries no comparison (it is the reference)
    baseline_row = next(row for row in data_rows if "| full_text |" in row)
    assert "| n/a |" in baseline_row
