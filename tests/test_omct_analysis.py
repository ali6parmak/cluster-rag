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

import ast
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from cluster_rag.cli import main
from cluster_rag.commands import omct as omct_commands
from cluster_rag.config import (
    AnalysisConfig,
    AnalysisEmbedderConfig,
    DictionaryConfig,
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
from cluster_rag.text_processors.features import (
    FeatureEmbedders,
    PairFeatureExtractor,
    ProfileUnits,
)
from fake_embedder import FakeEmbedder

ANALYSIS_ROOT = Path(__file__).resolve().parents[1] / "src" / "cluster_rag" / "omct_analysis"
FORBIDDEN_IMPORTS = ("torch", "sentence_transformers", "httpx", "sqlite3", "requests")

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

# CS-48: the dictionary section needs no embedder (the map is a description).
DICTIONARY_CONFIG_EXTRA = """
analysis:
  embedders: []
  control_size: 3
  sampling_rule: proportional
  question_transform: strip_block
  chunker: full_text
  dictionary:
    include_map: true
    folds: 5
    seed: null
"""

# The empty embedder list alone is not enough (the guard's message names it).
ANALYSIS_CONFIG_NO_EMBEDDERS_EXTRA = """
analysis:
  embedders: []
  control_size: 3
  sampling_rule: proportional
  question_transform: strip_block
  chunker: full_text
"""

# CS-51: the class-restricted arms (the hard filter against the soft prior).
DICTIONARY_CLASSES_CONFIG_EXTRA = """
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
  dictionary:
    include_map: true
    folds: 5
    scorings:
      - sum_logodds
      - sum_positive
      - max_logodds
      - support_count
      - sum_zscore
      - sum_zscore_positive
    primary: sum_zscore_positive
    class_restrictions: [nogeneric, distinctive]
  bootstrap:
    resamples: 200
"""

# CS-52: the grammatical arms (the tagger is faked in the test: no nltk data).
DICTIONARY_POS_CONFIG_EXTRA = """
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
  dictionary:
    include_map: true
    folds: 5
    scorings:
      - sum_zscore_positive
    primary: sum_zscore_positive
    pos_restrictions: [noun, verb]
  bootstrap:
    resamples: 200
"""

# CS-54: the stability readout and the stability-filtered arms.
DICTIONARY_STABILITY_CONFIG_EXTRA = """
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
  dictionary:
    include_map: true
    folds: 5
    scorings: [sum_zscore_positive, support_count]
    primary: sum_zscore_positive
    blind: false
    stability: true
    stability_resamples: 5
    stability_fold_resamples: 3
    stability_thresholds: [0.75, 0.9]
  bootstrap:
    resamples: 200
"""

# CS-53: the phrase table and the union arm (the word z arm + the phrase z arm).
DICTIONARY_PHRASES_CONFIG_EXTRA = """
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
  dictionary:
    include_map: true
    folds: 5
    units: phrases
    scorings: [sum_zscore_positive, support_count]
    primary: sum_zscore_positive
    blind: false
    union: true
  bootstrap:
    resamples: 200
"""

# CS-49: the dictionary arm needs exactly one reference model (its cosine).
DICTIONARY_ARM_CONFIG_EXTRA = """analysis:
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
  dictionary:
    include_map: true
    folds: 5
    scorings: [sum_logodds, sum_positive, max_logodds, support_count]
    blind: true
  bootstrap:
    resamples: 200
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


ANALYSIS_STAGE3_REPORTING_CONFIG_EXTRA = """
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
    reporting:
      enabled: true
      bins: 4
      threshold_criterion: floor_gap
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


# The Stage-3 random-vector floor (CS-6d): the feature pack's embedding sides
# are random unit vectors instead of the model's.
ANALYSIS_STAGE3_RANDOM_CONFIG_EXTRA = """
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
    embeddings: random
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
        omct_commands,
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
        omct_commands,
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
        omct_commands,
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
        omct_commands,
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


def test_analyze_omct_dictionary_run_is_model_free(
    sample_pairs, write_config, tmp_path, capsys
) -> None:
    """CS-48: the question map is a description — no embedder is built."""
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-dict-test"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    # the run carries no model: the map is model-independent
    assert payload["models"] == []
    block = payload["dictionary"]
    assert block is not None
    assert block["corpus"] == {
        "n_reviewed_pairs": 6,
        "n_positive": 3,
        "n_negative": 3,
        "n_questions": 4,
        "n_paragraphs": 6,
        "n_documents": 3,
    }
    shape = block["map"]
    assert shape["n_questions_with_table"] == 3
    assert shape["n_questions_empty"] == 1
    assert shape["n_contrastable_questions"] == 2
    assert any("media freedom" in question for question in shape["empty_questions"])
    # the map's tables carry the positive pair's words with their counts
    defenders = next(q for q in block["tables"] if "defenders detained" in q)
    rows = {row["word"]: row for row in block["tables"][defenders]}
    assert rows["defenders"]["n_with"] == 1
    assert rows["defenders"]["df"] == 1
    # the availability readout reads every pair against the other folds
    availability = block["availability"]
    assert availability["folds"] == 5
    assert availability["overall"]["n_pairs"] == 6
    assert len(availability["per_pair"]) == 6
    assert 0.0 <= (availability["overall"]["with_table_share"] or 0.0) <= 1.0
    # the map is deterministic: a re-run reproduces the same folds
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    again = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    assert again["dictionary"]["availability"]["per_pair"] == availability["per_pair"]

    out = capsys.readouterr().out
    assert "dictionary map (CS-48): 4 questions" in out
    # the row lands in the dictionary section's shape
    text = table.read_text("utf-8")
    assert "## OMCT — analyze-omct" in text
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-test"))
    assert "dictionary_map" in row
    assert "| 4 | 3 | 1 | 2 |" in row  # questions, tables, empty, contrastable


def test_analyze_omct_dictionary_arm_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """CS-49: the map as a pair scorer — the arm, the comparison and the row."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-arm-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_ARM_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    payload = json.loads(
        (tmp_path / "results" / "omct-dict-arm-test" / "analysis.json").read_text("utf-8")
    )
    assert [model.get("kind") for model in payload["models"]] == [None, "dictionary"]
    entry = payload["models"][1]
    assert entry["name"] == "dictionary"
    assert entry["model"]["reference_model"] == "fake-model"
    assert entry["model"]["primary_scoring"] == "sum_positive"
    readout = entry["readout"]
    assert set(readout["scorings"]) == {
        "sum_logodds",
        "sum_positive",
        "max_logodds",
        "support_count",
    }
    assert readout["primary"] == "sum_positive"
    assert readout["blind"] is not None
    assert set(readout["blind_gap_by_arm"]) == set(readout["scorings"])
    assert set(readout["spread_comparisons"]) == set(readout["scorings"])
    # the arm's comparison is against the run's baseline, paired
    comparison = entry["comparisons"]["dictionary"]
    assert comparison["reference_arm"] == "full_text"
    assert "within_group_paired" in comparison["auc_decomposition"]
    assert readout["floor"]["n_floor_pairs"] >= 0
    # the map block is still written beside the arm (CS-48's block)
    assert payload["dictionary"]["map"]["n_questions"] == 4

    out = capsys.readouterr().out
    assert "[dictionary] the question map (sum_positive)" in out
    text = table.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-arm-test"))
    assert "sum_positive" in row


def test_analyze_omct_dictionary_classes_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """CS-51: the class-restricted arms land in their own section with a delta."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-classes-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_CLASSES_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    payload = json.loads(
        (tmp_path / "results" / "omct-dict-classes-test" / "analysis.json").read_text("utf-8")
    )
    entry = next(model for model in payload["models"] if model.get("kind") == "dictionary")
    readout = entry["readout"]
    assert readout["primary"] == "sum_zscore_positive"
    assert set(readout["class_arms"]) == {"nogeneric", "distinctive"}
    for name in ("nogeneric", "distinctive"):
        arm = readout["class_arms"][name]
        assert 0.0 <= (arm["within_auc"] or 0.0) <= 1.0
        assert arm["reference_arm"] == "sum_zscore_positive"
        assert arm["floor_pairs"] >= 0
    assert entry["class_readout"]["thresholds"]["n_labelled_tables"] == 3
    assert entry["class_readout"]["thresholds"]["generic_min_tables"] == 3

    text = table.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-classes-test"))
    assert "sum_zscore_positive" in row


def test_analyze_omct_dictionary_pos_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """CS-52: the grammatical classes and the POS arms (the tagger is faked)."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    monkeypatch.setattr(
        omct_commands,
        "pos_word_classes",
        lambda config, texts: (
            {"defenders": "noun", "detained": "verb", "forced": "adjective"},
            {},
            {"model_id": "fake-tagger", "n_texts_tagged": len(texts)},
        ),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-pos-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_POS_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    payload = json.loads(
        (tmp_path / "results" / "omct-dict-pos-test" / "analysis.json").read_text("utf-8")
    )
    entry = next(model for model in payload["models"] if model.get("kind") == "dictionary")
    readout = entry["readout"]
    assert set(readout["pos_arms"]) == {"noun", "verb"}
    assert readout["pos_arms"]["noun"]["reference_arm"] == "sum_zscore_positive"
    assert entry["pos_readout"]["available"] is True
    assert entry["pos_readout"]["classes"]["noun"]["words"] == 1
    assert entry["pos_readout"]["classes"]["verb"]["words"] == 1
    assert payload["dictionary"]["pos"]["tagger"]["model_id"] == "fake-tagger"
    text = table.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-pos-test"))
    assert "sum_zscore_positive" in row


def test_analyze_omct_dictionary_phrases_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """CS-53: the phrase table, the two halves, the union and its own section row."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-phrases-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_PHRASES_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    payload = json.loads(
        (tmp_path / "results" / "omct-dict-phrases-test" / "analysis.json").read_text("utf-8")
    )
    # the map block now describes the PHRASE table, and says so
    block = payload["dictionary"]
    assert block["map"]["units"] == "phrases"
    assert block["unit_profile"]["name"] == "phrases"
    assert block["unit_profile"]["n_units"] == block["map"]["n_words"]
    entry = next(model for model in payload["models"] if model.get("kind") == "dictionary")
    assert entry["model"]["map_units"] == "phrases"
    assert entry["model"]["primary_scoring"] == "sum_zscore_positive"
    readout = entry["readout"]
    # no blind counterpart in this run (the pre-registration's reason)
    assert readout["blind"] is None
    union = readout["union"]
    assert union["arm"] == "sum_zscore_positive_phrase"
    assert union["scoring"] == "sum_zscore_positive"
    for key in ("within_auc", "pooled_auc"):
        assert union[key] is not None
    # the two halves and the increment are all measured
    assert union["word"]["within_auc"] is not None
    assert union["phrase"]["within_auc"] is not None
    assert union["vs_word"]["within_auc_word"] == union["word"]["within_auc"]
    assert union["vs_word"]["bar"] == 0.05
    assert union["vs_word"]["null_band"] == 0.02
    assert union["vs_word"]["ci_low"] <= union["vs_word"]["delta"] <= union["vs_word"]["ci_high"]
    # the count arms are carried for both halves (CS-49's instrument)
    assert union["word"]["count_within_auc"] is not None
    assert union["phrase"]["count_within_auc"] is not None
    # the floor is the two halves' shared set
    assert union["floor"]["n_floor_pairs"] >= 0
    assert union["floor"]["word_floor_pairs"] == union["floor"]["n_floor_pairs"]
    assert isinstance(union["floor"]["same_floor_set"], bool)
    # the phrase arm's own cosine comparison is on the entry (test 2)
    assert readout["reference"]["model"] == "fake-model"
    assert "sum_zscore_positive" in readout["scorings"]

    text = table.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-phrases-test"))
    assert "sum_zscore_positive_phrase" in row


def test_analyze_omct_dictionary_stability_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """CS-54: the bootstrap readout, the filtered arms and their own section row."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    table = tmp_path / "experiments_table_text_processing_omct.md"
    config_file = write_config(
        sample_pairs,
        run_id="dict-stability-test",
        data_external=sample_pairs,
        experiment_table_text_processing_omct=table,
        extra=DICTIONARY_STABILITY_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    payload = json.loads(
        (tmp_path / "results" / "omct-dict-stability-test" / "analysis.json").read_text("utf-8")
    )
    block = payload["dictionary"]["stability"]
    assert block["resamples"] == 5
    assert block["n_pairs"] == 6
    assert block["n_links"] == payload["dictionary"]["map"]["n_stored_pairs"]
    assert len(block["survival_curve"]) == 21
    assert block["single_pair"]["n_with"] == 1
    assert block["kept_share"]["0.75"] is not None
    entry = next(model for model in payload["models"] if model.get("kind") == "dictionary")
    readout = entry["readout"]
    assert readout["stability_thresholds"] == ["0.75", "0.90"]
    assert set(readout["stability_arms"]) == {"0.75", "0.90"}
    for arm in readout["stability_arms"].values():
        assert arm["within_auc"] is None or 0.0 <= arm["within_auc"] <= 1.0
        assert arm["delta_vs_full"] is not None
        assert arm["ci_low"] <= arm["ci_high"]
        assert arm["floor_pairs"] >= 0
        assert arm["reference_arm"] == "sum_zscore_positive"
    assert entry["stability_readout"]["resamples"] == 5

    text = table.read_text("utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| omct-dict-stability-test"))
    assert "sum_zscore_positive" in row


def test_analyze_omct_without_embedders_needs_the_dictionary_section(
    sample_pairs, write_config, caplog
) -> None:
    """An empty embedder list is only allowed for a dictionary-only run."""
    config_file = write_config(
        sample_pairs,
        run_id="no-embedders",
        data_external=sample_pairs,
        extra=ANALYSIS_CONFIG_NO_EMBEDDERS_EXTRA,
    )
    with caplog.at_level(logging.ERROR):
        assert main(["analyze-omct", "--config", str(config_file)]) == 1
    assert any("analysis.embedders" in record.message for record in caplog.records)


def test_dictionary_config_is_validated_and_defaults_are_the_plan() -> None:
    """The CS-48 knobs: five folds, the run's seed, the map on."""
    config = AnalysisConfig(embedders=(), dictionary=DictionaryConfig())
    assert config.dictionary is not None
    assert config.dictionary.include_map is True
    assert config.dictionary.folds == 5
    assert config.dictionary.seed is None
    with pytest.raises(ValueError):
        DictionaryConfig(folds=1)
    with pytest.raises(ValueError):
        DictionaryConfig(unknown=True)
    with pytest.raises(ValueError, match="unknown arm"):
        DictionaryConfig(scorings=("sum_g2",))
    with pytest.raises(ValueError, match="must not repeat"):
        DictionaryConfig(scorings=("sum_positive", "sum_positive"))
    # the arm needs exactly one reference embedder
    with pytest.raises(ValueError, match="exactly one reference embedder"):
        AnalysisConfig(
            embedders=(),
            dictionary=DictionaryConfig(scorings=("sum_positive",)),
        )
    # CS-50: the pre-registered primary arm is a config value, its name validated
    assert DictionaryConfig(primary="sum_zscore_positive").primary == "sum_zscore_positive"
    assert DictionaryConfig().primary is None
    with pytest.raises(ValueError, match="not a known arm"):
        DictionaryConfig(primary="sum_g2")
    # CS-51: the class restrictions are validated too
    assert DictionaryConfig(class_restrictions=("nogeneric", "distinctive")).class_restrictions == (
        "nogeneric",
        "distinctive",
    )
    with pytest.raises(ValueError, match="unknown restriction"):
        DictionaryConfig(class_restrictions=("generic",))
    # CS-52: the POS restrictions are validated too
    assert DictionaryConfig(pos_restrictions=("noun", "verb")).pos_restrictions == (
        "noun",
        "verb",
    )
    with pytest.raises(ValueError, match="unknown restriction"):
        DictionaryConfig(pos_restrictions=("adjective",))
    # CS-53: the map's unit and the union arm are validated too
    assert DictionaryConfig().units == "words"
    assert DictionaryConfig(units="phrases").units == "phrases"
    assert DictionaryConfig(units="phrases", union=True).union is True
    with pytest.raises(ValueError, match="analysis.dictionary.units must be one of"):
        DictionaryConfig(units="characters")
    # the union needs the phrase unit (it sums the word and the phrase halves)
    with pytest.raises(ValueError, match="needs analysis.dictionary.units"):
        DictionaryConfig(union=True)
    # CS-54: the stability knobs and the filtered-arm guard
    assert DictionaryConfig().stability is False
    assert DictionaryConfig().stability_resamples == 500
    assert DictionaryConfig().stability_fold_resamples == 100
    assert DictionaryConfig(stability=True, scorings=("sum_zscore_positive",)).stability is True
    assert DictionaryConfig(stability_thresholds=(0.9, 0.75)).stability_thresholds == (0.75, 0.9)
    with pytest.raises(ValueError, match="must not be empty"):
        DictionaryConfig(stability_thresholds=())
    with pytest.raises(ValueError, match="shares in"):
        DictionaryConfig(stability_thresholds=(1.5,))
    with pytest.raises(ValueError, match="needs analysis.dictionary.scorings"):
        DictionaryConfig(stability=True)


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
        omct_commands,
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
        omct_commands,
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
        omct_commands,
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
        omct_commands,
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


def test_analyze_omct_stage3_reporting_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path, capsys
) -> None:
    """The Stage-3 reporting: the calibration, the thresholds and the margin rule."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-reporting",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_REPORTING_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0

    results_dir = tmp_path / "results" / "omct-analyze-stage3-reporting"
    payload = json.loads((results_dir / "analysis.json").read_text("utf-8"))
    stage3 = payload["models"][0]["stage3"]
    # the run's config records the reporting's settings
    reporting_config = payload["analysis"]["stage3"]["reporting"]
    assert reporting_config["enabled"] is True
    assert reporting_config["bins"] == 4
    assert reporting_config["threshold_criterion"] == "floor_gap"

    reporting = stage3["reporting"]
    assert set(reporting) == {"calibration", "thresholds", "safe_negative"}

    # the calibration: the arm's probability and the cosine's fit-side map,
    # pooled and per document, with the bin counts
    calibration = reporting["calibration"]
    for key in ("arm", "cosine"):
        block = calibration[key]
        assert block["pooled"]["n"] == 6
        assert block["pooled"]["n_bins"] == 4
        assert len(block["pooled"]["bins"]) == 4
        assert sum(bin_["n"] for bin_ in block["pooled"]["bins"]) == 6
        assert set(block["by_document"]) == {"Doc A", "Doc B", "Doc C"}
        for document, diagram in block["by_document"].items():
            # the fixture's judged pairs per document: 2, 3 and 1
            assert diagram["n"] == {"Doc A": 2, "Doc B": 3, "Doc C": 1}[document]
            assert sum(bin_["n"] for bin_ in diagram["bins"]) == diagram["n"]
        assert block["macro_ece"] is not None
    # the cosine's map is a fit-side logistic on the cosine as its single feature
    assert calibration["cosine"]["map"]["feature"] == "cosine_full"

    # the thresholds: one per document, chosen on the fit side, applied to the
    # holdout, with the pooled out-of-fold operating point beside them
    thresholds = reporting["thresholds"]
    assert thresholds["criterion"] == "floor_gap"
    assert set(thresholds["per_document"]) == {"Doc A", "Doc B", "Doc C"}
    for block in thresholds["per_document"].values():
        assert "threshold" in block["fit_side"]
        assert block["fit_side"]["n_candidates"] > 0
        operating = block["holdout"]["operating_point"]
        if operating is not None:
            assert operating["n"] == (
                operating["tp"] + operating["fn"] + operating["fp"] + operating["tn"]
            )
    assert thresholds["out_of_fold"]["median_threshold"] is not None
    assert thresholds["out_of_fold"]["operating_point_at_median"]["n"] == 6

    # the safe-negative rule: the three variants, the coverage over the
    # unjudged grid and the harm on the verified positives
    safe_negative = reporting["safe_negative"]
    assert set(safe_negative["variants"]) == {"cosine_margin", "arm_margin", "both_margins"}
    assert "UNMEASURED" in safe_negative["unmeasured"]
    for _name, block in safe_negative["variants"].items():
        pooled = block["pooled"]
        assert pooled["n_unjudged_cells"] == 3
        assert pooled["n_unjudged_cleared"] <= 3
        assert 0.0 <= (pooled["coverage"] or 0.0) <= 1.0
        assert 0.0 <= (pooled["harm"] or 0.0) <= 1.0
        assert set(block["per_document"]) == {"Doc A", "Doc B", "Doc C"}
    # the control-role floor's counterpart of the arm-margin rule: the draws
    # of the two rotations whose fit side has unjudged cells to reach for
    # (the fixture's third document has none, so its rotation has no rule)
    floor = safe_negative["variants"]["arm_margin"]["floor_arm"]
    assert floor["n_draws"] == 6
    assert floor["harm_at_matched_coverage"]["coverage"] is not None

    out = capsys.readouterr().out
    assert "calibration" in out
    assert "safe-negative rule" in out


def test_analyze_omct_stage3_reporting_is_off_by_default(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """The reporting is opt-in, so the earlier runs' artifacts keep their content."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-no-reporting",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "omct-analyze-stage3-no-reporting" / "analysis.json").read_text(
            "utf-8"
        )
    )
    stage3 = payload["models"][0]["stage3"]
    assert payload["analysis"]["stage3"]["reporting"]["enabled"] is False
    # no reporting block: the earlier runs' artifacts keep their exact content
    assert "reporting" not in stage3
    for rotation in stage3["rotations"]:
        assert "reporting" not in rotation


def test_analyze_omct_stage3_reporting_is_reproducible(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """The same seed gives the same reporting numbers."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-reporting",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_REPORTING_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    first = json.loads(
        (tmp_path / "results" / "omct-analyze-stage3-reporting" / "analysis.json").read_text(
            "utf-8"
        )
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    second = json.loads(
        (tmp_path / "results" / "omct-analyze-stage3-reporting" / "analysis.json").read_text(
            "utf-8"
        )
    )
    assert first["models"][0]["stage3"]["reporting"] == second["models"][0]["stage3"]["reporting"]


def test_analyze_omct_stage3_random_floor_end_to_end(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """The Stage-3 random-vector floor: the pack's embedding sides become noise."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    # the same run with the model's embeddings (the Stage-3.1 behaviour)
    model_config = write_config(
        sample_pairs,
        run_id="analyze-stage3-model",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(model_config)]) == 0
    model_payload = json.loads(
        (tmp_path / "results" / "omct-analyze-stage3-model" / "analysis.json").read_text("utf-8")
    )
    model_stage3 = model_payload["models"][0]["stage3"]
    assert model_payload["analysis"]["stage3"]["embeddings"] == "model"
    assert model_stage3["feature_embeddings"]["mode"] == "model"

    # the same run with the random-vector floor
    random_config = write_config(
        sample_pairs,
        run_id="analyze-stage3-random",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_RANDOM_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(random_config)]) == 0
    payload = json.loads(
        (tmp_path / "results" / "omct-analyze-stage3-random" / "analysis.json").read_text("utf-8")
    )
    model = payload["models"][0]
    stage3 = model["stage3"]
    # the run's config records the mode, so the artifact says which space the
    # arm's embedding features live in
    assert payload["analysis"]["stage3"]["embeddings"] == "random"
    provenance = stage3["feature_embeddings"]
    assert provenance["mode"] == "random"
    assert provenance["seed"] == payload["seed"]
    assert provenance["dimension"] == 8
    assert provenance["cached"] is False
    # the random vectors are never written to the embedding cache
    sides = stage3["feature_pack"]["embedding_sides"]
    assert sides
    assert all(side["cache_hits"] is None for side in sides.values())
    # the baseline arm keeps the model's embeddings: the guard still holds and
    # the arm-vs-cosine comparison stays meaningful
    assert [pair["score"] for pair in model["judged_pairs"]] == [
        pair["score"] for pair in model_payload["models"][0]["judged_pairs"]
    ]
    # ... and the learned arm's own scores do change: its embedding features
    # are noise now
    assert (
        stage3["out_of_fold_scores"]["judged_pairs"]
        != (model_stage3["out_of_fold_scores"]["judged_pairs"])
    )
    # the same harness as every other Stage-3 run: three rotations, two floors
    assert len(stage3["rotations"]) == 3
    assert stage3["permutation_floor"]["n_resamples"] == 3


def test_analyze_omct_stage3_random_floor_is_reproducible(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """The floor is a function of the seed: the same seed gives the same run."""
    monkeypatch.setattr(
        omct_commands,
        "build_embedder",
        lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8),
    )
    config_file = write_config(
        sample_pairs,
        run_id="analyze-stage3-random",
        data_external=sample_pairs,
        extra=ANALYSIS_STAGE3_RANDOM_CONFIG_EXTRA,
    )
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    path = tmp_path / "results" / "omct-analyze-stage3-random" / "analysis.json"
    first = json.loads(path.read_text("utf-8"))["models"][0]["stage3"]
    assert main(["analyze-omct", "--config", str(config_file)]) == 0
    second = json.loads(path.read_text("utf-8"))["models"][0]["stage3"]
    assert first["out_of_fold_scores"] == second["out_of_fold_scores"]
    assert first["rotations_summary"] == second["rotations_summary"]


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


def test_stage3_reports_the_feature_diagnostics_and_the_length_check(view) -> None:
    """The Stage-3.3 reporting: every feature's own numbers and the length check."""
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
    units = ProfileUnits(
        sentences=build_chunker("units", {"kind": "sentence"}),
        ngrams=build_chunker("units", {"kind": "ngram", "n_words": 5}),
    )
    stage3 = score_stage3_model(
        view,
        model_name="fake-model",
        extractor=PairFeatureExtractor(
            FeatureEmbedders(
                paragraph=embedder,
                question=embedder,
                chunks=embedder,
                paragraph_sentences=embedder,
                question_sentences=embedder,
                paragraph_ngrams=embedder,
                question_ngrams=embedder,
            ),
            paragraph_chunker=FullTextChunker(),
            top_k=2,
            groups=("embedding", "lexical", "chunk_profile"),
            profile_units=units,
        ),
        combiner_factory=lambda: build_combiner("logistic", {"C": 1.0}),
        reference_similarities=baseline.similarities,
        control_weight=0.3,
        permutation_resamples=2,
        seed=42,
        bootstrap_resamples=20,
        bootstrap_seed=1,
    )
    # every feature the pack computes gets its own block: the discrimination
    # number, the control-side counterpart and the length correlation
    diagnostics = stage3["feature_diagnostics"]["per_feature"]
    assert set(diagnostics) == set(stage3["feature_pack"]["all_feature_names"])
    for entry in diagnostics.values():
        assert "within_auc" in entry
        assert "pooled_auc" in entry
        assert "macro_auc" in entry
        assert set(entry["control"]) == {"mean", "p50", "rank_p50", "top_fifth_share"}
        assert "overall" in entry["length_correlation"]
    # the length check: the arm, the cosine and the scale-free rank shift
    length_check = stage3["length_check"]
    assert set(length_check) == {"arm", "reference", "rank_shift", "note"}
    for block in ("arm", "reference", "rank_shift"):
        assert "overall" in length_check[block]
    # the out-of-fold scores: one per judged pair and per control cell, in the
    # order of the model's blocks (they are what a paired per-pair comparison
    # between two arms of different runs needs)
    scores = stage3["out_of_fold_scores"]
    assert len(scores["judged_pairs"]) == len(view.pairs)
    assert len(scores["control_cells"]) == len(view.control.cells)
    assert all(0.0 <= value <= 1.0 for value in scores["judged_pairs"])


def test_analyze_omct_arms_table_rows_use_the_arms_columns(
    sample_pairs, write_config, monkeypatch, tmp_path
) -> None:
    """A run with arms writes the Stage-1 columns (one row per model and arm)."""
    monkeypatch.setattr(
        omct_commands,
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
        omct_commands,
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


def test_the_analysis_driver_is_a_pure_core() -> None:
    """The analysis driver never imports a model client.

    It is the track's **measurement** code: a model client here would put a
    non-reproducible dependency between a run and its numbers. The embedder
    reaches it as a port (``cluster_rag.embed.base``), injected by the CLI.
    """
    pattern = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(FORBIDDEN_IMPORTS) + r")\b")
    for module_path in sorted(ANALYSIS_ROOT.rglob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for line in source.splitlines():
            assert not pattern.match(line), f"{module_path.name} imports a model client: {line}"


def test_the_analysis_driver_imports_only_numpy_and_cluster_rag() -> None:
    """Beyond numpy, the analysis driver may not import a third-party library.

    The track's pure package has one documented exception (scikit-learn, for
    the combiner); the driver has none — it orchestrates the pure core and the
    harness's own metrics.
    """
    allowed = {"numpy"}
    stdlib = set(sys.stdlib_module_names)
    for module_path in sorted(ANALYSIS_ROOT.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        for module in sorted(imported):
            if module in stdlib or module in allowed or module.startswith("cluster_rag"):
                continue
            raise AssertionError(
                f"{module_path.name} imports {module}: the analysis driver may only "
                "import numpy and cluster_rag"
            )
