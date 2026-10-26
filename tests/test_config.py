"""Tests for the YAML config system."""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster_rag.config import EvalConfig, load_config

CONFIG_YAML = """
paths:
  data_raw: data/raw/upr.jsonl
  data_processed: data/processed
  data_splits: data/splits
  cache: cache
  results: results
"""


def test_load_config_parses_values(tmp_path) -> None:
    """A valid config validates and keeps relative paths untouched."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nseed: 7\nrun_id: test-run\n", encoding="utf-8")

    config = load_config(config_file)
    assert config.seed == 7
    assert config.run_id == "test-run"
    assert config.paths.data_raw == Path("data/raw/upr.jsonl")
    assert config.paths.label_catalog is None


def test_load_config_defaults(tmp_path) -> None:
    """run_id and seed have defaults; only paths are required."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")

    config = load_config(config_file)
    assert config.seed == 42
    assert config.run_id is None


def test_load_config_rejects_missing_paths(tmp_path) -> None:
    """A required path missing from the YAML is a validation error."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("paths:\n  data_raw: data/raw/upr.jsonl\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_file)


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    """Typos in a config file (unknown keys) fail immediately."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nembeddr: typo-section\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_file)


def test_load_config_rejects_non_mapping(tmp_path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_file)


def test_load_config_stats_section(tmp_path) -> None:
    """The stats section validates, overrides the schema defaults and rejects typos."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML + "\nstats:\n  frequency_head_min: 77\n", encoding="utf-8")

    config = load_config(config_file)
    assert config.stats.frequency_head_min == 77
    assert config.stats.frequency_torso_min == 40  # untouched default

    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG_YAML + "\nstats:\n  frequency_head_minm: 77\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_embed_section(tmp_path) -> None:
    """The embed section validates backends/prefixes for both backends."""
    hf = CONFIG_YAML + (
        "\nembed:"
        "\n  backend: hf"
        "\n  model_id: sentence-transformers/all-MiniLM-L6-v2"
        "\n  revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        "\n  device: cuda"
        "\n  prefixes:"
        "\n    query: 'query: '"
        "\n    document: 'passage: '"
    )
    config = load_config_str(hf, tmp_path)
    assert config.embed.backend == "hf"
    assert config.embed.prefixes.query == "query: " and config.embed.prefixes.label == ""
    assert config.embed.batch_size == 64  # untouched default
    assert config.preprocess.variant == "raw" and config.cache.enabled

    ollama = hf.replace("backend: hf", "backend: ollama").replace(
        "  revision: c9745ed1d9f207416be6d2e6f8de32d1f16199bf", "  host: http://h:1234"
    )
    config = load_config_str(ollama, tmp_path)
    assert config.embed.host == "http://h:1234" and config.embed.timeout == 120.0

    bad = hf + "\n  batch_size: 0\n"
    with pytest.raises(ValueError):
        load_config_str(bad, tmp_path)


def test_load_config_probe_transform_defaults_and_validation(tmp_path) -> None:
    """strip_block is the default question transform; unknown ones fail at load time."""
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(CONFIG_YAML, encoding="utf-8")
    config = load_config(config_file)
    assert config.probe.question_transforms == ("strip_block",)

    both = tmp_path / "both.yaml"
    both.write_text(
        CONFIG_YAML + "\nprobe:\n  question_transforms: [strip_block, keep]\n",
        encoding="utf-8",
    )
    assert load_config(both).probe.question_transforms == ("strip_block", "keep")

    bad = tmp_path / "bad.yaml"
    bad.write_text(CONFIG_YAML + "\nprobe:\n  question_transforms: [squeeze]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown question transform"):
        load_config(bad)


def test_load_config_extends_merges_parent(tmp_path) -> None:
    """extends merges a parent tree underneath; child wins in deep merge."""
    parent = tmp_path / "base.yaml"
    parent.write_text(CONFIG_YAML + "\nseed: 7\nrun_id: parent\n", encoding="utf-8")
    child = tmp_path / "exp.yaml"
    child.write_text(
        "extends: base.yaml\nseed: 99\npaths:\n  cache: other-cache\n", encoding="utf-8"
    )
    config = load_config(child)
    assert config.seed == 99  # child override
    assert config.run_id == "parent"  # parent value kept
    assert config.paths.cache == Path("other-cache")  # deep merge into paths
    assert config.paths.data_raw == Path("data/raw/upr.jsonl")  # parent paths kept


def test_load_config_extends_supports_chains_and_rejects_cycles(tmp_path) -> None:
    """Grandparents merge too; a extends cycle fails instead of hanging."""
    grand = tmp_path / "grand.yaml"
    grand.write_text(CONFIG_YAML + "\nrun_id: grand\n", encoding="utf-8")
    parent = tmp_path / "parent.yaml"
    parent.write_text("extends: grand.yaml\nseed: 7\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\n", encoding="utf-8")
    config = load_config(child)
    assert (config.seed, config.run_id) == (7, "grand")


def test_extends_cycle_direct(tmp_path) -> None:
    """A config extending itself is rejected."""
    child = tmp_path / "selfref.yaml"
    child.write_text(CONFIG_YAML + "\nextends: selfref.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle|deep"):
        load_config(child)


def load_config_str(text: str, directory: Path):
    """Helper: write a yaml string under directory and load it."""
    config_file = directory / "cfg.yaml"
    config_file.write_text(text, encoding="utf-8")
    return load_config(config_file)


EVAL_SECTION = """\

eval:
  predictor:
    name: most_frequent
"""


def test_load_config_eval_defaults(tmp_path) -> None:
    """eval defaults: top_k k=2 decision, k/m grids, val split, empty params."""
    config = load_config_str(CONFIG_YAML + EVAL_SECTION, tmp_path)
    assert config.eval is not None
    assert config.eval.predictor.name == "most_frequent"
    assert config.eval.predictor.params == {}
    assert config.eval.decision.rule == "top_k" and config.eval.decision.k == 2
    assert config.eval.decision.threshold is None and config.eval.decision.min_score is None
    assert config.eval.metrics.ks == (1, 3, 5, 10)
    assert config.eval.metrics.shortlist_ms == (20, 50, 100)
    assert config.eval.split == "val"


def test_load_config_eval_overrides(tmp_path) -> None:
    """Decision rules carry their own params; grids and split are configurable."""
    text = (
        CONFIG_YAML
        + """\

eval:
  predictor:
    name: random
    params: {}
  decision:
    rule: top_k_min_score
    k: 3
    min_score: 0.25
  metrics:
    ks: [1, 2, 5]
    shortlist_ms: [10, 20]
  split: test
"""
    )
    config = load_config_str(text, tmp_path)
    eval_config = config.eval
    assert eval_config is not None
    assert eval_config.predictor.name == "random"
    assert eval_config.decision.rule == "top_k_min_score" and eval_config.decision.k == 3
    assert eval_config.decision.min_score == 0.25
    assert eval_config.metrics.ks == (1, 2, 5)
    assert eval_config.metrics.shortlist_ms == (10, 20)
    assert eval_config.split == "test"


def test_load_config_eval_rejects_unknown_rule_and_missing_params(tmp_path) -> None:
    """Unknown rules and rule-params mismatches fail at load time."""
    for bad_section, match in (
        ("    name: x\n  decision:\n    rule: magic\n", "Unknown decision rule"),
        (
            "    name: x\n  decision:\n    rule: score_threshold\n",
            "requires decision.threshold",
        ),
        (
            "    name: x\n  decision:\n    rule: top_k_min_score\n    k: 3\n",
            "requires decision.min_score",
        ),
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(CONFIG_YAML + "\neval:\n  predictor:\n" + bad_section, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(bad)


def test_load_config_eval_split_restricted_and_grids_positive(tmp_path) -> None:
    """Only val/test are evaluable splits; metric grids need positive values."""
    bad_split = tmp_path / "split.yaml"
    bad_split.write_text(CONFIG_YAML + EVAL_SECTION + "  split: train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be 'val' or 'test'"):
        load_config(bad_split)

    bad_grid = tmp_path / "grid.yaml"
    bad_grid.write_text(
        CONFIG_YAML + EVAL_SECTION + "  metrics:\n    ks: [0, 1]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="positive"):
        load_config(bad_grid)


def test_load_config_embed_kinds(tmp_path) -> None:
    """embed_kinds validates (known kinds, non-empty) and defaults to document."""
    config = load_config_str(CONFIG_YAML, tmp_path)
    assert config.embed_kinds == ("document",)

    kinds = load_config_str(CONFIG_YAML + "\nembed_kinds: [query, document]\n", tmp_path)
    assert kinds.embed_kinds == ("query", "document")

    unknown = tmp_path / "kinds.yaml"
    unknown.write_text(CONFIG_YAML + "\nembed_kinds: [vectors]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown encode kind"):
        load_config(unknown)

    empty = tmp_path / "empty.yaml"
    empty.write_text(CONFIG_YAML + "\nembed_kinds: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one encode kind"):
        load_config(empty)


def test_eval_embedding_run_id_validation() -> None:
    """eval.embedding_run_id: optional at load, but only upr-embed-* ids validate."""
    filled = EvalConfig.model_validate(
        {"predictor": {"name": "knn"}, "embedding_run_id": "upr-embed-e5-base-v2"}
    )
    assert filled.embedding_run_id == "upr-embed-e5-base-v2"
    assert EvalConfig.model_validate({"predictor": {"name": "random"}}).embedding_run_id is None
    # omct- ids would mix the auxiliary line into a UPR metric: rejected
    with pytest.raises(ValueError, match="upr-embed-"):
        EvalConfig.model_validate(
            {"predictor": {"name": "knn"}, "embedding_run_id": "omct-probe2-x"}
        )
    # a path is not a run id (the loader resolves against paths.results itself)
    with pytest.raises(ValueError, match="filesystem-safe"):
        EvalConfig.model_validate({"predictor": {"name": "knn"}, "embedding_run_id": "results/x"})


def test_eval_label_embedding_run_id_validation() -> None:
    """eval.label_embedding_run_id: only upr-labelembed-* ids validate."""
    filled = EvalConfig.model_validate(
        {"predictor": {"name": "zeroshot"}, "label_embedding_run_id": "upr-labelembed-x"}
    )
    assert filled.label_embedding_run_id == "upr-labelembed-x"
    assert EvalConfig.model_validate({"predictor": {"name": "knn"}}).label_embedding_run_id is None
    with pytest.raises(ValueError, match="upr-labelembed-"):
        EvalConfig.model_validate(
            {"predictor": {"name": "zeroshot"}, "label_embedding_run_id": "upr-embed-e5-base-v2"}
        )


def test_label_texts_variants() -> None:
    """variant 'name' vs 'template': the {label} placeholder rules are enforced."""
    from cluster_rag.config import LabelTextsConfig

    name = LabelTextsConfig.model_validate({"variant": "name"})
    assert name.render_text("Discrimination") == "Discrimination"

    templated = LabelTextsConfig.model_validate(
        {"variant": "template", "template": "This recommendation is about {label}."}
    )
    assert templated.render_text("Discrimination") == (
        "This recommendation is about Discrimination."
    )

    with pytest.raises(ValueError, match="requires template"):
        LabelTextsConfig.model_validate({"variant": "template"})
    with pytest.raises(ValueError, match="\\{label\\} placeholder"):
        LabelTextsConfig.model_validate({"variant": "template", "template": "About things."})
    with pytest.raises(ValueError, match="must not set template"):
        LabelTextsConfig.model_validate({"variant": "name", "template": "oops"})


def test_upr_analysis_defaults_and_validation() -> None:
    """The UPR analysis section: the defaults and the config-time checks."""
    from cluster_rag.config import UprAnalysisConfig

    section = UprAnalysisConfig()
    assert section.length_buckets == (100, 200, 300)
    assert section.reference_eval_run is None
    assert section.bootstrap.resamples == 2000

    # the length edges are sorted, positive and unique
    assert UprAnalysisConfig(length_buckets=(300, 100)).length_buckets == (100, 300)
    with pytest.raises(ValueError, match="length_buckets"):
        UprAnalysisConfig(length_buckets=(0, 100))
    with pytest.raises(ValueError, match="length_buckets"):
        UprAnalysisConfig(length_buckets=(100, 100))

    # unknown keys are rejected (a typo fails before the run)
    with pytest.raises(ValueError, match="Extra inputs"):
        UprAnalysisConfig.model_validate({"nope": 1})


def test_analysis_stage3_defaults_and_validation() -> None:
    """The Stage-3 section: the settled defaults and the config-time checks."""
    from cluster_rag.config import (
        AnalysisConfig,
        CombinerSpec,
        FeaturePackConfig,
        Stage3Config,
    )

    stage3 = Stage3Config()
    assert stage3.control_weight == 0.3
    assert stage3.permutation_resamples == 5
    assert stage3.combiner.name == "logistic"
    # the feature pack's default chunker is the Stage-2 verdict's window
    assert stage3.features.paragraph_chunker.name == "word_windows"
    assert stage3.features.paragraph_chunker.params == {"n_words": 25, "stride": 12}
    assert stage3.features.question_chunker is None
    assert stage3.features.top_k == 3
    # the feature-group selection defaults to the whole v0 pack
    assert stage3.features.groups == (
        "embedding",
        "lexical",
        "question_part",
        "question_type",
    )
    # the Stage-3.3 chunk profile's own knobs
    assert stage3.features.profile_top_k == 3
    assert stage3.features.profile_ngram_size == 5
    # the embedding sides come from the run's model by default (the Stage-3.1
    # behaviour); the random-vector floor is opt-in
    assert stage3.embeddings == "model"

    # the section is optional: without it the Stage-0/1/2 behaviour is untouched
    assert AnalysisConfig().stage3 is None

    # the control weight is a weight in [0, 1]
    with pytest.raises(ValueError, match="control_weight"):
        Stage3Config.model_validate({"control_weight": 1.5})
    with pytest.raises(ValueError, match="control_weight"):
        Stage3Config.model_validate({"control_weight": -0.1})
    with pytest.raises(ValueError, match="permutation_resamples"):
        Stage3Config.model_validate({"permutation_resamples": -1})
    with pytest.raises(ValueError, match="top_k"):
        FeaturePackConfig.model_validate({"top_k": 0})

    # the combiner is validated against the registry
    with pytest.raises(ValueError, match="Unknown combiner"):
        CombinerSpec.model_validate({"name": "does-not-exist"})
    with pytest.raises(ValueError, match="Unknown parameter"):
        CombinerSpec.model_validate({"name": "logistic", "params": {"nope": 1}})
    assert CombinerSpec.model_validate({"name": "logistic", "params": {"C": 0.5}}).params == {
        "C": 0.5
    }


def test_analysis_stage3_feature_groups_are_validated() -> None:
    """The feature-group knob fails at config time, before any run."""
    from cluster_rag.config import FeaturePackConfig

    # a shorter selection is the ablation's arm
    selected = FeaturePackConfig.model_validate({"groups": ["embedding", "lexical"]})
    assert selected.groups == ("embedding", "lexical")
    # the Stage-3.3 chunk profile is a group of its own (opt-in)
    profile = FeaturePackConfig.model_validate({"groups": ["chunk_profile"]})
    assert profile.groups == ("chunk_profile",)
    # an unknown group name is a configuration error
    with pytest.raises(ValueError, match="Unknown feature group"):
        FeaturePackConfig.model_validate({"groups": ["embedding", "not_a_group"]})
    # an empty selection has no features to fit on
    with pytest.raises(ValueError, match="at least one feature group"):
        FeaturePackConfig.model_validate({"groups": []})
    # a repeated group is a configuration error
    with pytest.raises(ValueError, match="must not repeat"):
        FeaturePackConfig.model_validate({"groups": ["lexical", "lexical"]})
    # the profile's own knobs are validated too
    with pytest.raises(ValueError, match="profile_top_k"):
        FeaturePackConfig.model_validate({"profile_top_k": 0})
    with pytest.raises(ValueError, match="profile_ngram_size"):
        FeaturePackConfig.model_validate({"profile_ngram_size": 1})


def test_analysis_stage3_embedding_mode_is_validated() -> None:
    """The random-vector floor's knob fails at config time, before any run."""
    from cluster_rag.config import Stage3Config

    # the two modes of the feature pack's embedding sides
    assert Stage3Config.model_validate({"embeddings": "model"}).embeddings == "model"
    assert Stage3Config.model_validate({"embeddings": "random"}).embeddings == "random"
    # an unknown mode is a configuration error (it would silently fall back)
    with pytest.raises(ValueError, match="embeddings"):
        Stage3Config.model_validate({"embeddings": "noise"})


def test_analysis_stage3_chunker_params_are_validated() -> None:
    """The feature pack's chunker spec is checked like every other chunker spec."""
    from cluster_rag.config import FeaturePackConfig

    with pytest.raises(ValueError, match="Unknown chunker"):
        FeaturePackConfig.model_validate({"paragraph_chunker": {"name": "does-not-exist"}})
    with pytest.raises(ValueError, match="Unknown parameter"):
        FeaturePackConfig.model_validate(
            {"paragraph_chunker": {"name": "word_windows", "params": {"nope": 1}}}
        )
