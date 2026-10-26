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
    BaselineArm,
    ChunkedSide,
    SideEncoding,
    build_dataset_view,
    compare_arms,
    document_code,
    score_model_arms,
    score_stage3_model,
    side_similarities,
)
from cluster_rag.pairs import PairRecord, load_pairs
from cluster_rag.preprocess import build_preprocess_spec
from cluster_rag.text_processors.aggregation import parse_aggregation
from cluster_rag.text_processors.chunkers import FullTextChunker, build_chunker
from cluster_rag.text_processors.combining import build_combiner
from cluster_rag.text_processors.control import (
    SAMPLING_RULE_PROPORTIONAL,
    DocumentGrid,
    allocate_quotas,
    rank_percentiles,
    sample_control,
)
from cluster_rag.text_processors.features import FeatureEmbedders, PairFeatureExtractor
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


ANALYSIS_LADDER_CONFIG_EXTRA = """
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
    name: word_windows
    params:
      n_words: 3
      stride: 2
  question_chunker:
    name: word_windows
    params:
      n_words: 3
      stride: 2
  aggregations: [max, "local_global:0.5:max", "decayed_top_k:2:0.5:0.25", "peak:2:0.05"]
  bootstrap:
    resamples: 20
    seed: 3
"""

ANALYSIS_SEMANTIC_CONFIG_EXTRA = """
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
    name: semantic
    params:
      threshold: 0.5
  semantic_thresholds: [0.2, 0.8]
  aggregations: [max]
  bootstrap:
    resamples: 20
    seed: 3
"""

ANALYSIS_STAGE3_CONFIG_EXTRA = """
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
  bootstrap:
    resamples: 20
    seed: 3
  stage3:
    features:
      paragraph_chunker:
        name: word_windows
        params:
          n_words: 3
          stride: 2
      top_k: 2
    combiner:
      name: logistic
      params:
        C: 1.0
    control_weight: 0.3
    permutation_resamples: 3
"""


ANALYSIS_STAGE3_GROUPS_CONFIG_EXTRA = """
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
  bootstrap:
    resamples: 20
    seed: 3
  stage3:
    features:
      paragraph_chunker:
        name: word_windows
        params:
          n_words: 3
          stride: 2
      top_k: 2
      groups: [embedding, lexical]
    combiner:
      name: logistic
      params:
        C: 1.0
    control_weight: 0.3
    permutation_resamples: 3
"""


ANALYSIS_STAGE3_HISTGB_CONFIG_EXTRA = """
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
  bootstrap:
    resamples: 20
    seed: 3
  stage3:
    features:
      paragraph_chunker:
        name: word_windows
        params:
          n_words: 3
          stride: 2
      top_k: 2
    combiner:
      name: histgb
      params:
        max_iter: 10
        min_samples_leaf: 1
    control_weight: 0.3
    permutation_resamples: 3
"""


@pytest.fixture()
def view(sample_pairs: Path):
    """The model-independent view of the pairwise fixture."""
    records = load_pairs(sample_pairs).records
    return build_dataset_view(records, control_size=3, seed=42)


def test_analyze_omct_ladder_run_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """A Stage-2 ladder run: the chunked arms and the symmetric question side."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-ladder",
        data_external=sample_pairs,
        extra=ANALYSIS_LADDER_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-ladder"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    model = payload["models"][0]
    arms = model["arms"]
    # the four aggregation rules of the word-window chunker
    assert set(arms) == {
        "words3_max",
        "words3_local_global_0.5_max",
        "words3_decayed_top_k_2_0.5_0.25",
        "words3_peak_2_0.05",
    }
    assert payload["analysis"]["chunker"] == "word_windows"
    assert payload["analysis"]["question_chunker"] == "word_windows"
    assert payload["analysis"]["aggregations"] == [
        "max",
        "local_global:0.5:max",
        "decayed_top_k:2:0.5:0.25",
        "peak:2:0.05",
    ]
    # the symmetric arm folds the question's chunks: the question side has
    # more chunks than questions
    question_side = arms["words3_max"]["sides"]["question"]
    assert question_side["n_chunks"] > question_side["n_texts"]
    # every arm is paired with the baseline on the same cells
    for arm in arms.values():
        assert [cell["cell_id"] for cell in arm["control_cells"]] == [
            cell["cell_id"] for cell in model["control_cells"]
        ]
    assert "dAUC" in capsys.readouterr().out


def test_analyze_omct_semantic_run_selects_the_threshold_on_the_fit_side(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """A swept run: the threshold is chosen per document on the fit side."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-semantic",
        data_external=sample_pairs,
        extra=ANALYSIS_SEMANTIC_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-semantic"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    model = payload["models"][0]
    selection = model["threshold_selection"]
    assert selection["grid"] == [0.2, 0.8]
    assert selection["criterion"] == "auc"
    assert len(selection["rotations"]) == 3
    for rotation in selection["rotations"]:
        assert rotation["holdout"] not in rotation["fit_documents"]
        assert rotation["chosen"]["max"]["threshold"] in (0.2, 0.8)
    assert payload["analysis"]["semantic_thresholds"] == [0.2, 0.8]
    assert "semantic_max" in model["arms"]


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


def test_analysis_config_validates_the_stage2_knobs() -> None:
    """The Stage-2 knobs: the swept grid, the symmetric variant, the criterion."""
    embedders = (
        AnalysisEmbedderConfig(name="m", embed=EmbedderConfig(backend="hf", model_id="m")),
    )
    # the swept grid belongs to the semantic chunker
    with pytest.raises(ValueError, match="only applies to the 'semantic' chunker"):
        AnalysisConfig(
            embedders=embedders,
            chunker={"name": "atoms"},
            aggregations=("max",),
            semantic_thresholds=(0.3, 0.5),
        )
    with pytest.raises(ValueError, match="must not repeat a value"):
        AnalysisConfig(
            embedders=embedders,
            chunker={"name": "semantic"},
            semantic_thresholds=(0.3, 0.3),
        )
    with pytest.raises(ValueError, match=r"must be in \[-1, 1\]"):
        AnalysisConfig(
            embedders=embedders,
            chunker={"name": "semantic"},
            semantic_thresholds=(1.5,),
        )
    # a chunked question side needs the arms that fold it
    with pytest.raises(ValueError, match="question_chunker needs aggregation rules"):
        AnalysisConfig(
            embedders=embedders,
            chunker={"name": "atoms"},
            question_chunker={"name": "word_windows", "params": {"n_words": 15}},
        )
    # a valid Stage-2 configuration loads
    config = AnalysisConfig(
        embedders=embedders,
        chunker={"name": "semantic"},
        aggregations=("max", "local_global:0.5:max"),
        question_chunker={"name": "sentence_windows", "params": {"n_sentences": 2}},
        semantic_thresholds=(0.3, 0.5),
        semantic_selection="floor_gap",
    )
    assert config.semantic_thresholds == (0.3, 0.5)
    assert config.semantic_selection == "floor_gap"
    assert config.question_chunker is not None
    assert config.question_chunker.name == "sentence_windows"


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


def test_side_similarities_folds_a_chunked_question_side() -> None:
    """A chunked question side is folded with the maximum (the symmetric arm)."""
    paragraph_side = SideEncoding(
        texts=("p",), chunk_lists=(("p1",),), vectors=np.array([[1.0, 0.0]])
    )
    question_side = SideEncoding(
        texts=("a",), chunk_lists=(("a1", "a2"),), vectors=np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    folded = side_similarities(paragraph_side, question_side, parse_aggregation("max"))
    # the paragraph chunk matches the first question chunk perfectly
    assert folded[0, 0] == pytest.approx(1.0)
    # the baseline arm needs one chunk per text on both sides
    with pytest.raises(ValueError, match="one chunk per question"):
        side_similarities(paragraph_side, question_side, None)


def test_side_similarities_blends_the_full_text_score() -> None:
    """The local+global rule needs the full-text matrix as its second input."""
    paragraph_side = SideEncoding(
        texts=("p",), chunk_lists=(("p1", "p2"),), vectors=np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    question_side = SideEncoding(
        texts=("a",), chunk_lists=(("a1",),), vectors=np.array([[1.0, 0.0]])
    )
    global_matrix = np.array([[0.2]])
    blended = side_similarities(
        paragraph_side,
        question_side,
        parse_aggregation("local_global:0.5:max"),
        global_similarities=global_matrix,
    )
    assert blended[0, 0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.2)


def test_swept_variant_chooses_the_threshold_on_the_fit_side(view) -> None:
    """The swept parameter is chosen per document on the fit side only."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    sides = [
        ChunkedSide(
            chunker=build_chunker("semantic", {"threshold": threshold}, embedder=embedder),
            paragraph_embedder=embedder,
            threshold=threshold,
        )
        for threshold in (0.2, 0.8)
    ]
    report = score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
        swept_sides=sides,
        aggregations=(parse_aggregation("max"),),
        semantic_selection="auc",
        bootstrap_resamples=20,
        bootstrap_seed=1,
    )
    selection = report["threshold_selection"]
    assert selection["grid"] == [0.2, 0.8]
    assert selection["criterion"] == "auc"
    assert len(selection["rotations"]) == 3
    for rotation in selection["rotations"]:
        assert rotation["holdout"] in {"Doc A", "Doc B", "Doc C"}
        assert rotation["holdout"] not in rotation["fit_documents"]
        chosen = rotation["chosen"]["max"]
        assert chosen["threshold"] in (0.2, 0.8)
        # the chosen threshold is the one with the best fit-side AUC
        usable = {
            float(threshold): block["max"]["fit"]["auc"]
            for threshold, block in rotation["grid"].items()
            if block["max"]["fit"]["auc"] is not None
        }
        assert chosen["threshold"] == max(
            usable, key=lambda threshold: (usable[threshold], -threshold)
        )
        # the holdout's own numbers are reported beside the choice
        assert chosen["holdout"]["n_pairs"] == rotation["n_holdout_pairs"]
    # every document's threshold is recorded
    assert set(selection["chosen_by_document"]["max"]) == {"Doc A", "Doc B", "Doc C"}
    # the arm exists and carries the swept parameter's provenance
    arm = report["arms"]["semantic_max"]
    assert arm["sides"]["paragraph"]["swept_parameter"]["values"] == [0.2, 0.8]
    assert arm["sides"]["paragraph"]["n_chunks_by_threshold"]
    # the other fit-side criterion is wired too: on this fixture the control
    # cells live in one document only, so the fit side of that document's
    # holdout has no control cells, the floor gap is undefined and the
    # selection fails loudly instead of silently falling back.
    with pytest.raises(ValueError, match="no usable criterion value"):
        score_model_arms(
            view,
            model_name="fake-model",
            paragraph_embedder=embedder,
            question_embedders={"strip_block": embedder},
            baseline_chunker=FullTextChunker(),
            question_transform="strip_block",
            question_block_markers=("Answer choices:", "Notes:"),
            preprocess_spec=spec,
            swept_sides=sides,
            aggregations=(parse_aggregation("max"),),
            semantic_selection="floor_gap",
            bootstrap_resamples=20,
            bootstrap_seed=1,
        )


def test_swept_variant_covers_every_cell_of_the_baseline(view) -> None:
    """The cross-fitted arm measures the same cells as the baseline arm."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    sides = [
        ChunkedSide(
            chunker=build_chunker("semantic", {"threshold": threshold}, embedder=embedder),
            paragraph_embedder=embedder,
            threshold=threshold,
        )
        for threshold in (0.2, 0.8)
    ]
    report = score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
        swept_sides=sides,
        aggregations=(parse_aggregation("max"),),
        bootstrap_resamples=20,
        bootstrap_seed=1,
    )
    arm = report["arms"]["semantic_max"]
    assert [cell["cell_id"] for cell in arm["control_cells"]] == [
        cell["cell_id"] for cell in report["control_cells"]
    ]
    assert [pair["pair_id"] for pair in arm["judged_pairs"]] == [
        pair["pair_id"] for pair in report["judged_pairs"]
    ]
    assert "semantic_max" in report["comparisons"]
    assert report["comparisons"]["semantic_max"]["reference_arm"] == "full_text"


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


def test_analyze_omct_stage3_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """A Stage-3 run: the learned arm, the rotations and the label-permutation floors."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-stage3"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    model = payload["models"][0]
    stage3 = model["stage3"]
    # the config is recorded, so the run is reproducible from the artifact
    assert payload["analysis"]["stage3"]["control_weight"] == 0.3
    assert payload["analysis"]["stage3"]["combiner"]["name"] == "logistic"
    assert payload["analysis"]["stage3"]["features"]["top_k"] == 2
    # the feature pack's provenance
    assert stage3["feature_pack"]["n_features"] == 15
    assert stage3["feature_pack"]["top_k"] == 2
    assert stage3["combiner"]["name"] == "logistic"
    # one rotation per document, and the fit side never holds the holdout out
    assert len(stage3["rotations"]) == 3
    for rotation in stage3["rotations"]:
        assert rotation["holdout"] not in rotation["fit_documents"]
        assert rotation["n_fit_pairs"] + rotation["n_holdout_pairs"] == 6
        assert rotation["control_weight"] == 0.3
        assert rotation["fit_weight_mass"]["control"] == pytest.approx(
            rotation["n_fit_control"] * 0.3
        )
    # the out-of-fold matrix scores the same-document cells only
    assert stage3["grid"]["n_cells"] == 9
    assert stage3["grid"]["n_scored_cells"] == 9
    assert stage3["grid"]["n_matrix_positions"] == 24
    assert stage3["grid"]["n_unscored_positions"] == 15
    # the two label-permutation floors
    floor = stage3["permutation_floor"]
    assert floor["n_resamples"] == 3
    assert floor["chance"]["mean"] is not None
    assert floor["control_role"]["mean"] is not None
    for rotation in stage3["rotations"]:
        assert len(rotation["permutation_floor"]["chance_auc"]) == 3
        assert len(rotation["permutation_floor"]["control_role_auc"]) == 3
    # the comparison blocks of the Stage-2 runs are computed on the arm
    comparison = stage3["comparison"]
    assert comparison["reference_arm"] == "full_text"
    assert set(comparison) >= {
        "paired_auc",
        "pair_auc",
        "auc_decomposition",
        "control_side",
        "floor_gap",
        "deshifted_auc",
        "hit_at_k",
    }
    assert comparison["auc_decomposition"]["within_group_paired"]["n_resamples"] == 20
    # the paragraph lengths are recorded (the length-vs-shift question)
    lengths = payload["analysis"]["paragraph_lengths"]
    assert len(lengths["per_paragraph"]) == 6
    assert set(lengths["by_document"]) == {"Doc A", "Doc B", "Doc C"}
    assert model["length_correlation"]["overall"]["n"] == 3

    out = capsys.readouterr().out
    assert "OMCT analysis (Stage 0): omct-analyze-stage3" in out
    assert "chance floor" in out


def test_analyze_omct_stage3_group_selection_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """A Stage-3.1b run: the pack keeps only the selected feature groups."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-groups",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_GROUPS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-stage3-groups"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    stage3 = payload["models"][0]["stage3"]
    # the config is recorded, so the run is reproducible from the artifact
    assert payload["analysis"]["stage3"]["features"]["groups"] == ["embedding", "lexical"]
    # the pack computes every feature but the combiner sees the groups only
    assert stage3["feature_pack"]["groups"] == ["embedding", "lexical"]
    assert stage3["feature_pack"]["n_features"] == 9
    assert "cosine_full" in stage3["feature_pack"]["feature_names"]
    assert "lex_char_ngram_cosine" in stage3["feature_pack"]["feature_names"]
    assert "qtype_option_count" not in stage3["feature_pack"]["feature_names"]
    assert "qtype_option_count" in stage3["feature_pack"]["dropped_features"]
    # the coefficients are named after the selected features
    coefficients = stage3["rotations"][0]["coefficients"]["values"]
    assert set(coefficients) == set(stage3["feature_pack"]["feature_names"])
    # the question overlap of the two sides is measured (the leakage rule (i))
    for rotation in stage3["rotations"]:
        overlap = rotation["question_overlap"]
        assert overlap["n_holdout_questions"] > 0
        assert (
            0 <= overlap["n_holdout_questions_also_judged_in_fit"] <= overlap["n_holdout_questions"]
        )
        assert overlap["n_holdout_pairs_with_a_shared_question"] <= rotation["n_holdout_pairs"]


def test_analyze_omct_stage3_histgb_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """A Stage-3.2 run: the gradient-boosted combiner and its feature usage."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-histgb",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_HISTGB_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-stage3-histgb"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    stage3 = payload["models"][0]["stage3"]
    # the config is recorded, so the run is reproducible from the artifact
    assert payload["analysis"]["stage3"]["combiner"]["name"] == "histgb"
    assert stage3["combiner"]["name"] == "histgb"
    assert stage3["combiner"]["params"]["max_iter"] == 10
    # the same harness as the Stage-3.1 runs: three rotations, two floors
    assert len(stage3["rotations"]) == 3
    assert stage3["permutation_floor"]["n_resamples"] == 3
    # the importances capability: the trees' split usage, named after the
    # pack's features (a tree ensemble has no coefficients)
    feature_names = set(stage3["feature_pack"]["feature_names"])
    for rotation in stage3["rotations"]:
        assert rotation["coefficients"] == {}
        usage = rotation["feature_usage"]
        assert set(usage["per_feature"]) == feature_names
        for entry in usage["per_feature"].values():
            assert set(entry) == {
                "n_splits",
                "total_gain",
                "max_depth",
                "split_share",
                "gain_share",
            }
        total_splits = sum(entry["n_splits"] for entry in usage["per_feature"].values())
        if total_splits:
            assert sum(
                entry["split_share"] for entry in usage["per_feature"].values()
            ) == pytest.approx(1.0)

    out = capsys.readouterr().out
    assert "OMCT analysis (Stage 0): omct-analyze-stage3-histgb" in out


def test_stage3_rotations_hold_out_one_document_at_a_time(view) -> None:
    """The document-level holdout: the fit side never contains the holdout."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    baseline = BaselineArm()
    score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
        baseline=baseline,
    )
    assert baseline.similarities is not None
    stage3 = score_stage3_model(
        view,
        model_name="fake-model",
        extractor=PairFeatureExtractor(
            FeatureEmbedders(
                paragraph=embedder,
                question=embedder,
                chunks=embedder,
            ),
            paragraph_chunker=FullTextChunker(),
            top_k=2,
        ),
        combiner_factory=lambda: build_combiner("logistic", {"C": 1.0}),
        reference_similarities=baseline.similarities,
        control_weight=0.3,
        permutation_resamples=2,
        seed=42,
        bootstrap_resamples=20,
        bootstrap_seed=1,
    )
    assert {rotation["holdout"] for rotation in stage3["rotations"]} == {
        "Doc A",
        "Doc B",
        "Doc C",
    }
    for rotation in stage3["rotations"]:
        assert set(rotation["fit_documents"]) == {"Doc A", "Doc B", "Doc C"} - {rotation["holdout"]}
    # the out-of-fold matrix holds every same-document cell
    assert stage3["grid"]["n_cells"] == 9
    assert stage3["grid"]["n_scored_cells"] == 9
    # the per-rotation summary pairs the arm with the baseline and the floors
    summary = stage3["rotations_summary"]
    assert len(summary) == 3
    for entry in summary:
        assert "chance_floor_mean" in entry
        assert "control_role_floor_mean" in entry
    # "Doc C" holds a single positive pair, so its AUC is undefined
    by_holdout = {entry["holdout"]: entry for entry in summary}
    assert by_holdout["Doc C"]["arm_pair_auc"] is None
    assert by_holdout["Doc C"]["reference_pair_auc"] is None
    assert by_holdout["Doc C"]["chance_floor_mean"] is None
    assert by_holdout["Doc A"]["arm_pair_auc"] is not None
    assert by_holdout["Doc A"]["reference_pair_auc"] is not None
    assert by_holdout["Doc A"]["chance_floor_mean"] is not None


def test_stage3_learns_a_lexical_signal_the_cosine_cannot_see() -> None:
    """A controlled signal: the learned arm finds it, the chance floor does not.

    The signal is lexical (the question's topic word), so the hash-based
    fake cosine carries none of it: the arm's gain over the baseline is the
    feature pack's doing, and the label-permutation floor stays at chance.
    """
    records = []
    counter = 0
    for document in ("Doc A", "Doc B", "Doc C"):
        for index in range(4):
            counter += 1
            paragraph = f"The committee discussed matter {counter} in its report."
            # the first question is the positive one, the second the negative
            for question_index, question_text in enumerate(
                (
                    "Question: Does the report discuss zorblax?",
                    "Question: Does the report discuss quillmarr?",
                )
            ):
                if question_index == 1 and index >= 2:
                    continue  # the unjudged cells (the control's pool)
                records.append(
                    PairRecord(
                        pair_id=f"{document}-{index}-{question_index}",
                        source_csv="src.csv",
                        document=document,
                        paragraph_number=str(index),
                        paragraph_text=paragraph,
                        question_number=question_index,
                        question_text=question_text,
                        label=1 if question_index == 0 else 0,
                        label_origin="merged",
                    )
                )
    view = build_dataset_view(records, control_size=4, seed=42)
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    baseline = BaselineArm()
    score_model_arms(
        view,
        model_name="fake-model",
        paragraph_embedder=embedder,
        question_embedders={"strip_block": embedder},
        baseline_chunker=FullTextChunker(),
        question_transform="strip_block",
        question_block_markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
        baseline=baseline,
    )
    stage3 = score_stage3_model(
        view,
        model_name="fake-model",
        extractor=PairFeatureExtractor(
            FeatureEmbedders(paragraph=embedder, question=embedder, chunks=embedder),
            paragraph_chunker=FullTextChunker(),
            top_k=2,
        ),
        combiner_factory=lambda: build_combiner("logistic", {"C": 1.0}),
        reference_similarities=baseline.similarities,
        control_weight=0.3,
        permutation_resamples=5,
        seed=42,
        bootstrap_resamples=50,
        bootstrap_seed=1,
    )
    # the lexical signal is learnable: the arm is well above chance
    assert stage3["permutation_floor"]["chance"]["mean"] == pytest.approx(0.5, abs=0.2)
    arm_macro = stage3["comparison"]["pair_auc"]["macro"]["arm"]
    assert arm_macro > 0.8
    # the baseline (a hash-based cosine) carries no signal to learn from
    assert arm_macro > stage3["comparison"]["pair_auc"]["macro"]["reference"]


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


def test_analyze_omct_stage3_table_row_carries_the_groups_and_the_hit1(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """A Stage-3 run writes the learned-arm columns (the groups, the hit@1)."""
    monkeypatch.setattr(
        cli,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-table",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=ANALYSIS_STAGE3_GROUPS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    text = table.read_text("utf-8")
    data_rows = [
        line for line in text.splitlines() if line.startswith("| omct-analyze-stage3-table")
    ]
    assert len(data_rows) == 1
    row = data_rows[0]
    # the feature groups and the pack's size
    assert "| embedding+lexical |" in row
    assert "| 9 |" in row
    # the hit@1 of the arm and of the cosine
    hit_cells = row.split("|")
    assert hit_cells[-4].strip() != "n/a"  # hit1_pos
    assert hit_cells[-3].strip() != "n/a"  # hit1_pos_ref
    # the two floor columns carry the within-document weighted floors
    header = [line for line in text.splitlines() if line.startswith("| run_id")][0]
    assert "perm_auc_within" in header
    assert "ctrl_role_auc_within" in header
