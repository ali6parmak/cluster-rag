"""Tests for the track's Stage-3.1 pure core: the feature pack and the combiner.

Contracts under test (TEXT_PROCESSING_PLAN.md section 3, Stage 3.1):

- the feature pack: the v0 feature list, the question's decomposition into
  stem and options, the question-type one-hots, the K-dependent names, and
  the pack's **pair-local, label-free** character (no corpus statistic, no
  document identity, no length feature);
- the logistic combiner: the fit/score contract behind the ``Combiner`` port,
  the sample weights (the control cells are down-weighted), the scaler
  (fitted on the fit side only) and the registry's parameter validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.text_processors.base import Combiner, FeatureExtractor, TextPair
from cluster_rag.text_processors.chunkers import (
    FullTextChunker,
    UnitSplitterConfig,
    build_chunker,
)
from cluster_rag.text_processors.combining import (
    COMBINERS,
    HistGbCombinerConfig,
    LogisticCombinerConfig,
    build_combiner,
    combiner_config_class,
)
from cluster_rag.text_processors.features import (
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_STOPWORDS,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    PROFILE_FEATURE_NAMES,
    V0_FEATURE_NAMES,
    FeatureEmbedders,
    PairFeatureExtractor,
    ProfileUnits,
    feature_group,
    split_question_parts,
)
from fake_embedder import FakeEmbedder

PARAGRAPH = (
    "The Committee notes with concern that forced displacement rose sharply "
    "last year, affecting thousands of families in the region."
)
QUESTION = (
    "Question: How often did forced evictions occur?\n\n"
    "Answer choices:\n- Very often\n- Often\n- Sometimes\n- No data available\n"
)


def _profile_units() -> ProfileUnits:
    """The chunk profile's unit splitters (the sentences and the word pieces)."""
    return ProfileUnits(
        sentences=build_chunker("units", {"kind": "sentence"}),
        ngrams=build_chunker("units", {"kind": "ngram", "n_words": 5}),
    )


def _extractor(
    *,
    dimension: int = 8,
    top_k: int = 3,
    question_embedding_texts: dict[str, str] | None = None,
    chunker=None,
    groups: tuple[str, ...] | None = None,
    profile_units: ProfileUnits | None = None,
    profile_top_k: int = 3,
) -> PairFeatureExtractor:
    """A feature pack over the fake embedder (no network, no model download)."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=dimension)
    return PairFeatureExtractor(
        FeatureEmbedders(
            paragraph=embedder,
            question=embedder,
            chunks=embedder,
            paragraph_sentences=embedder,
            question_sentences=embedder,
            paragraph_ngrams=embedder,
            question_ngrams=embedder,
        ),
        paragraph_chunker=chunker or FullTextChunker(),
        top_k=top_k,
        question_embedding_texts=question_embedding_texts,
        groups=groups,
        profile_units=profile_units,
        profile_top_k=profile_top_k,
    )


def test_question_parts_split_the_stem_from_the_options() -> None:
    """The stem is the topical part; the options are the typed part."""
    parts = split_question_parts(QUESTION)
    assert parts.stem == "How often did forced evictions occur?"
    assert parts.options == ("Very often", "Often", "Sometimes", "No data available")
    assert parts.has_options


def test_question_parts_without_an_options_block() -> None:
    """A question without options has an empty option list."""
    parts = split_question_parts("Question: Were defenders detained?")
    assert parts.stem == "Were defenders detained?"
    assert parts.options == ()
    assert not parts.has_options


def test_question_parts_cut_a_trailing_notes_block() -> None:
    """A 'Notes:' section ends the options block."""
    parts = split_question_parts(
        "Question: How often?\n\nAnswer choices:\n- Often\n- Rarely\n\nNotes: see annex"
    )
    assert parts.options == ("Often", "Rarely")


def test_feature_pack_produces_the_v0_feature_list() -> None:
    """The default selection is the v0 list; the K-dependent name follows the K."""
    extractor = _extractor(top_k=3)
    assert extractor.feature_names == V0_FEATURE_NAMES
    assert len(extractor.feature_names) == 15
    assert "chunk_cosine_top3_mean" in extractor.feature_names
    assert _extractor(top_k=5).feature_names[3] == "chunk_cosine_top5_mean"


def test_feature_pack_computes_the_whole_pack() -> None:
    """The pack computes every feature; the selection is the combiner's view."""
    extractor = _extractor(profile_units=_profile_units())
    assert extractor.all_feature_names == FEATURE_NAMES
    assert len(extractor.all_feature_names) == 25
    # the whole pack: the v0's 15 features plus the Stage-3.3 profile's 10
    whole = _extractor(profile_units=_profile_units(), groups=FEATURE_GROUPS)
    assert whole.feature_names == FEATURE_NAMES
    assert whole.feature_names[: len(V0_FEATURE_NAMES)] == V0_FEATURE_NAMES
    assert whole.feature_names[len(V0_FEATURE_NAMES) :] == PROFILE_FEATURE_NAMES
    # the default selection leaves the profile out (the Stage-3.1 behaviour)
    assert extractor.feature_names == V0_FEATURE_NAMES
    assert set(PROFILE_FEATURE_NAMES) <= set(extractor.dropped_feature_names)
    # the pack's default selection is the config's default selection
    from cluster_rag.config import FeaturePackConfig

    assert FeaturePackConfig().groups == DEFAULT_FEATURE_GROUPS
    assert extractor.groups == DEFAULT_FEATURE_GROUPS


def test_feature_pack_has_no_length_feature() -> None:
    """The reviewer's decision: length is a document proxy, so it is not a feature."""
    names = _extractor().feature_names
    assert not any("length" in name or "chars" in name or "words" in name for name in names)


def test_feature_pack_is_pair_local_and_label_free() -> None:
    """The pack sees the text only: no label, no document, no corpus statistic."""
    extractor = _extractor()
    assert isinstance(extractor, FeatureExtractor)
    pair = TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)
    assert not hasattr(pair, "label")
    matrix = extractor.extract([pair])
    assert matrix.shape == (1, len(extractor.feature_names))
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()


def test_feature_pack_is_deterministic_and_order_independent() -> None:
    """The same pairs give the same rows, whatever their order."""
    extractor = _extractor()
    first = TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)
    second = TextPair(paragraph_text="The parliament adopted the budget.", question_text=QUESTION)
    matrix = extractor.extract([first, second])
    again = extractor.extract([second, first])
    assert matrix[0].tolist() == again[1].tolist()
    assert matrix[1].tolist() == again[0].tolist()


def test_feature_pack_detects_the_question_type() -> None:
    """The type one-hots: the options, the 'no data' option, the frequency."""
    extractor = _extractor()
    features = extractor.extract([TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)])
    row = features[0]
    names = extractor.feature_names
    assert row[names.index("qtype_has_options")] == 1.0
    assert row[names.index("qtype_has_no_data_option")] == 1.0
    assert row[names.index("qtype_frequency")] == 1.0
    assert row[names.index("qtype_option_count")] == 4.0
    assert row[names.index("qtype_yes_no")] == 0.0


def test_feature_pack_uses_the_question_transform_for_the_embedding() -> None:
    """The embedding text is the transformed one; the parts still see the options."""
    stripped = "How often did forced evictions occur?"
    extractor = _extractor(question_embedding_texts={QUESTION: stripped})
    features = extractor.extract([TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)])
    names = extractor.feature_names
    # the options are still found in the raw question
    assert features[0][names.index("qtype_option_count")] == 4.0
    # the embedded question is the stripped one
    reference = _extractor()
    plain = reference.extract([TextPair(paragraph_text=PARAGRAPH, question_text=stripped)])
    assert features[0][names.index("cosine_full")] == pytest.approx(
        plain[0][names.index("cosine_full")]
    )


def test_feature_pack_measures_the_chunk_cosines() -> None:
    """A chunked paragraph gives more than one chunk cosine."""
    chunker = build_chunker("word_windows", {"n_words": 3, "stride": 2})
    extractor = _extractor(chunker=chunker)
    names = extractor.feature_names
    row = extractor.extract([TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)])[0]
    assert row[names.index("chunk_cosine_best")] != row[names.index("cosine_full")]
    assert row[names.index("chunk_cosine_spread")] >= 0.0
    assert row[names.index("chunk_peak_margin")] == pytest.approx(
        row[names.index("chunk_cosine_best")] - row[names.index("chunk_cosine_top3_mean")]
    )


def test_feature_pack_rejects_a_degenerate_k() -> None:
    """A non-positive K is a configuration error."""
    with pytest.raises(ValueError, match="top_k"):
        _extractor(top_k=0)


def test_feature_pack_describes_itself() -> None:
    """The description is the artifact's record of the pack."""
    described = _extractor(profile_units=_profile_units()).describe()
    assert described["name"] == "v0+chunk_profile"
    assert described["n_features"] == len(V0_FEATURE_NAMES)
    assert described["top_k"] == 3
    assert described["profile_top_k"] == 3
    assert described["stopwords"] == sorted(DEFAULT_STOPWORDS)
    assert "pair-local" in described["note"]


# -- the feature groups (the Stage-3.1b ablation's unit) ------------------------


def test_feature_groups_partition_the_pack() -> None:
    """Every feature belongs to exactly one group, and the five cover the pack."""
    assert set(FEATURE_GROUPS) == {
        "embedding",
        "lexical",
        "question_part",
        "question_type",
        "chunk_profile",
    }
    grouped: dict[str, list[str]] = {group: [] for group in FEATURE_GROUPS}
    for name in FEATURE_NAMES:
        group = feature_group(name)
        assert group is not None, name
        grouped[group].append(name)
    # the five groups partition the pack: every feature is in exactly one of
    # them and none is left over
    assert sorted(name for names in grouped.values() for name in names) == sorted(FEATURE_NAMES)
    assert len(grouped["embedding"]) == 6
    assert len(grouped["lexical"]) == 3
    assert len(grouped["question_part"]) == 1
    assert len(grouped["question_type"]) == 5
    assert len(grouped["chunk_profile"]) == 10
    # the groups hold the features the plan's list names
    assert feature_group("cosine_full") == "embedding"
    assert feature_group("chunk_cosine_spread") == "embedding"
    assert feature_group("lex_char_ngram_cosine") == "lexical"
    assert feature_group("qpart_options_coverage") == "question_part"
    assert feature_group("qtype_option_count") == "question_type"
    assert feature_group("cprof_sent_top1") == "chunk_profile"
    assert feature_group("cprof_align_lexical") == "chunk_profile"


def test_feature_pack_keeps_only_the_requested_groups() -> None:
    """The selection keeps the canonical order and drops the other groups' features."""
    extractor = _extractor(groups=("lexical",))
    assert extractor.feature_names == (
        "lex_question_coverage",
        "lex_paragraph_coverage",
        "lex_char_ngram_cosine",
    )
    assert len(extractor.dropped_feature_names) == 22
    assert "cosine_full" in extractor.dropped_feature_names
    assert "qtype_option_count" in extractor.dropped_feature_names
    # the default keeps the v0 pack (the Stage-3.1 behaviour)
    assert _extractor().feature_names == V0_FEATURE_NAMES
    assert "cprof_sent_top1" in _extractor().dropped_feature_names


def test_feature_pack_selection_only_changes_the_columns() -> None:
    """The selected columns are the full pack's columns, value for value."""
    pair = TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)
    full = _extractor(profile_units=_profile_units()).extract_all([pair])[0]
    for groups in (
        ("embedding",),
        ("lexical",),
        ("question_type",),
        ("embedding", "lexical"),
        ("chunk_profile",),
        FEATURE_GROUPS,
    ):
        extractor = _extractor(profile_units=_profile_units(), groups=groups)
        selected = extractor.extract([pair])[0]
        assert selected.shape == (len(extractor.feature_names),)
        for position, name in enumerate(extractor.feature_names):
            assert selected[position] == pytest.approx(full[FEATURE_NAMES.index(name)])


def test_feature_pack_rejects_a_bad_group_selection() -> None:
    """An unknown, an empty or a repeated group is a configuration error."""
    with pytest.raises(ValueError, match="Unknown feature group"):
        _extractor(groups=("embedding", "not_a_group"))
    with pytest.raises(ValueError, match="at least one group"):
        _extractor(groups=())
    with pytest.raises(ValueError, match="must not repeat"):
        _extractor(groups=("lexical", "lexical"))


def test_feature_pack_describes_the_group_selection() -> None:
    """The description records the groups and the dropped features."""
    described = _extractor(groups=("embedding", "question_type")).describe()
    assert described["groups"] == ["embedding", "question_type"]
    assert described["n_features"] == 11
    assert "lex_question_coverage" in described["dropped_features"]
    assert "cosine_full" not in described["dropped_features"]


# -- the chunk profile (the Stage-3.3 block) ------------------------------------


def test_unit_splitter_cuts_sentences_and_word_pieces() -> None:
    """The two granularities: the sentences and the word pieces inside them."""
    sentences = build_chunker("units", {"kind": "sentence"})
    pieces = build_chunker("units", {"kind": "ngram", "n_words": 5})
    text = "One sentence here. And another one, a bit longer than the first."
    assert sentences.units(text) == [
        "One sentence here.",
        "And another one, a bit longer than the first.",
    ]
    # the pieces tile the sentence, so the last one may be shorter
    assert pieces.units("One two three four five six.") == ["One two three four five", "six."]
    # a piece never spans a sentence boundary
    assert all("here. And" not in piece for piece in pieces.units(text))
    assert sentences.units("") == []
    assert pieces.units("") == []


def test_unit_splitter_rejects_a_degenerate_setting() -> None:
    """The unit splitter's parameters are validated."""
    with pytest.raises(ValueError, match="unit kind"):
        UnitSplitterConfig(kind="paragraph")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_words"):
        UnitSplitterConfig(kind="ngram", n_words=1)
    with pytest.raises(ValueError, match="min_unit_chars"):
        UnitSplitterConfig(min_unit_chars=0)


def test_unit_splitter_versions_its_parameters() -> None:
    """Two differently configured splitters never share cached units."""
    sentences = build_chunker("units", {"kind": "sentence"})
    pieces = build_chunker("units", {"kind": "ngram", "n_words": 5})
    other = build_chunker("units", {"kind": "ngram", "n_words": 4})
    assert sentences.version != pieces.version
    assert pieces.version != other.version
    assert sentences.cache_version("raw-v1") != pieces.cache_version("raw-v1")
    assert sentences.slug == "units-sent"
    assert pieces.slug == "units-5w"


def test_profile_features_are_the_ranked_cross_chunk_cosines() -> None:
    """The profile's ranks are the sorted cross-chunk cosines, computed by hand."""
    embedder = FakeEmbedder(model_id="fake-embedder", dimension=8)
    units = _profile_units()
    extractor = PairFeatureExtractor(
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
        groups=("chunk_profile",),
        profile_units=units,
    )
    row = extractor.extract([TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)])[0]
    names = extractor.feature_names
    for granularity, splitter in (("sent", units.sentences), ("ngram", units.ngrams)):
        paragraph_units = splitter.units(PARAGRAPH)
        question_units = splitter.units(QUESTION)
        paragraph_vectors = embedder.encode(paragraph_units, "document")
        question_vectors = embedder.encode(question_units, "query")
        expected = sorted(
            (paragraph_vectors @ question_vectors.T).reshape(-1).tolist(), reverse=True
        )
        for rank in range(3):
            value = row[names.index(f"cprof_{granularity}_top{rank + 1}")]
            if rank < len(expected):
                assert value == pytest.approx(expected[rank])
            else:
                assert value == 0.0  # the zero padding


def test_profile_pads_a_short_block_with_zeros() -> None:
    """A pair with fewer units than ranks pads the missing ranks with zeros."""
    extractor = _extractor(
        profile_units=_profile_units(), groups=("chunk_profile",), profile_top_k=5
    )
    row = extractor.extract([TextPair(paragraph_text="One short line.", question_text="A?")])[0]
    names = extractor.feature_names
    assert row[names.index("cprof_sent_top2")] == 0.0
    assert row[names.index("cprof_ngram_top3")] == 0.0


def test_profile_needs_the_unit_splitters() -> None:
    """The chunk_profile group without the unit splitters is a configuration error."""
    with pytest.raises(ValueError, match="unit splitters"):
        _extractor(groups=("chunk_profile",))


def test_profile_columns_are_zero_without_the_units() -> None:
    """Without the unit splitters the pack computes the other features only."""
    extractor = _extractor()
    row = extractor.extract_all([TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)])[0]
    for name in PROFILE_FEATURE_NAMES:
        assert row[FEATURE_NAMES.index(name)] == 0.0


def test_profile_is_pair_local_and_deterministic() -> None:
    """The profile sees one pair at a time and gives the same row every time."""
    extractor = _extractor(profile_units=_profile_units(), groups=("chunk_profile",))
    first = TextPair(paragraph_text=PARAGRAPH, question_text=QUESTION)
    second = TextPair(paragraph_text="The parliament adopted the budget.", question_text=QUESTION)
    matrix = extractor.extract([first, second])
    again = extractor.extract([second, first])
    assert matrix[0].tolist() == again[1].tolist()
    assert matrix[1].tolist() == again[0].tolist()
    # the two paragraphs differ, so the profile is not a constant
    assert matrix[0].tolist() != matrix[1].tolist()


def test_profile_ranks_follow_the_configured_k() -> None:
    """The profile's number of ranks is the pack's parameter."""
    extractor = _extractor(
        profile_units=_profile_units(), groups=("chunk_profile",), profile_top_k=2
    )
    assert extractor.feature_names == (
        "cprof_sent_top1",
        "cprof_sent_top2",
        "cprof_ngram_top1",
        "cprof_ngram_top2",
        "cprof_align_sent",
        "cprof_align_ngram",
        "cprof_align_lexical",
        "cprof_align_semantic_only",
    )
    with pytest.raises(ValueError, match="profile_top_k"):
        _extractor(profile_top_k=0)


def test_logistic_combiner_fits_and_scores() -> None:
    """The combiner scores a row higher when the first feature is higher."""
    combiner = build_combiner("logistic", {"C": 1.0})
    assert isinstance(combiner, Combiner)
    features = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    combiner.fit(features, labels)
    scores = combiner.score(features)
    assert scores.shape == (4,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[0] < scores[-1]


def test_logistic_combiner_honours_the_sample_weights() -> None:
    """A down-weighted row moves the fit less than a full-weight row."""
    features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    heavy = build_combiner("logistic", {"C": 1.0})
    heavy.fit(features, labels, [1.0, 1.0, 0.01, 0.01])
    light = build_combiner("logistic", {"C": 1.0})
    light.fit(features, labels, [1.0, 1.0, 1.0, 1.0])
    # the down-weighted positives make the model less confident about "high"
    assert heavy.score(features)[-1] < light.score(features)[-1]


def test_logistic_combiner_standardizes_inside_fit() -> None:
    """The scaler is fitted on the fit side's rows, not on the scored rows."""
    fit_features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    combiner = build_combiner("logistic", {"C": 1.0, "standardize": True})
    combiner.fit(fit_features, labels)
    # the scaler is fixed at fit time: a shifted copy maps to a huge z-score,
    # so the scores saturate instead of matching the unshifted ones
    shifted = fit_features + 1000.0
    assert combiner.score(shifted).max() == pytest.approx(1.0)
    assert combiner.score(fit_features).max() < 1.0
    # a combiner fitted on the shifted rows standardizes them away
    other = build_combiner("logistic", {"C": 1.0, "standardize": True})
    other.fit(shifted, labels)
    assert other.score(shifted).max() < 1.0


def test_logistic_combiner_rejects_bad_input() -> None:
    """The combiner fails loudly on a shape or label mismatch."""
    combiner = build_combiner("logistic", {"C": 1.0})
    with pytest.raises(ValueError, match="not fitted"):
        combiner.score(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="rows but"):
        combiner.fit(np.zeros((3, 1), dtype=np.float32), [0, 1])
    with pytest.raises(ValueError, match="0/1"):
        combiner.fit(np.zeros((2, 1), dtype=np.float32), [0, 2])
    with pytest.raises(ValueError, match="zero rows"):
        combiner.fit(np.zeros((0, 1), dtype=np.float32), [])


def test_combiner_registry_validates_its_parameters() -> None:
    """The registry maps a name to its class and rejects unknown parameters."""
    assert set(COMBINERS) == {"logistic", "histgb"}
    assert combiner_config_class("logistic") is LogisticCombinerConfig
    assert combiner_config_class("histgb") is HistGbCombinerConfig
    with pytest.raises(ValueError, match="Unknown combiner"):
        combiner_config_class("lightgbm")
    with pytest.raises(ValueError, match="Unknown parameter"):
        build_combiner("logistic", {"not_a_parameter": 1})
    with pytest.raises(ValueError, match="Unknown parameter"):
        build_combiner("histgb", {"not_a_parameter": 1})
    with pytest.raises(ValueError, match="Unknown combiner"):
        build_combiner("does-not-exist")


def test_logistic_combiner_config_rejects_a_degenerate_parameter() -> None:
    """The parameters are validated at config time."""
    with pytest.raises(ValueError, match="C must be"):
        LogisticCombinerConfig(C=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        LogisticCombinerConfig(max_iter=0)
    with pytest.raises(ValueError, match="class_weight"):
        LogisticCombinerConfig(class_weight="weird")


# -- the gradient-boosted combiner (the Stage-3.2 learner) -------------------


def test_histgb_combiner_fits_and_scores() -> None:
    """The combiner scores a row higher when the informative feature is higher."""
    combiner = build_combiner("histgb", {"min_samples_leaf": 1, "max_iter": 50})
    assert isinstance(combiner, Combiner)
    features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    combiner.fit(features, labels)
    scores = combiner.score(features)
    assert scores.shape == (4,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert scores[0] < scores[-1]


def test_histgb_combiner_honours_the_sample_weights() -> None:
    """A down-weighted row moves the fit less than a full-weight row."""
    features = np.array([[0.0], [0.2], [0.8], [1.0]], dtype=np.float32)
    labels = [0, 0, 1, 1]
    heavy = build_combiner("histgb", {"min_samples_leaf": 1, "max_iter": 50})
    heavy.fit(features, labels, [1.0, 1.0, 0.01, 0.01])
    light = build_combiner("histgb", {"min_samples_leaf": 1, "max_iter": 50})
    light.fit(features, labels, [1.0, 1.0, 1.0, 1.0])
    # the down-weighted positives make the model less confident about "high"
    assert heavy.score(features)[-1] < light.score(features)[-1]


def test_histgb_combiner_reports_the_trees_feature_usage() -> None:
    """The usage diagnostic says which features the trees split on."""
    rng = np.random.default_rng(0)
    informative = rng.normal(size=200)
    features = np.column_stack([informative, np.zeros(200)]).astype(np.float32)
    labels = (informative > 0).astype(int).tolist()
    combiner = build_combiner("histgb", {"min_samples_leaf": 5})
    combiner.fit(features, labels)
    usage = combiner.feature_usage()
    # the constant column can never be split on
    assert usage["1"]["n_splits"] == 0
    assert usage["1"]["total_gain"] == 0.0
    # the informative column is the one the trees use
    assert usage["0"]["n_splits"] > 0
    assert usage["0"]["total_gain"] > 0
    assert usage["0"]["split_share"] == pytest.approx(1.0)
    assert usage["0"]["gain_share"] == pytest.approx(1.0)
    # every feature of the matrix gets an entry (a feature the trees never
    # used is information, not a missing value)
    assert set(usage) == {"0", "1"}


def test_histgb_combiner_rejects_bad_input() -> None:
    """The combiner fails loudly on a shape or label mismatch."""
    combiner = build_combiner("histgb", {})
    with pytest.raises(ValueError, match="not fitted"):
        combiner.score(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="rows but"):
        combiner.fit(np.zeros((3, 1), dtype=np.float32), [0, 1])
    with pytest.raises(ValueError, match="0/1"):
        combiner.fit(np.zeros((2, 1), dtype=np.float32), [0, 2])
    with pytest.raises(ValueError, match="zero rows"):
        combiner.fit(np.zeros((0, 1), dtype=np.float32), [])
    with pytest.raises(ValueError, match="sample weights"):
        combiner.fit(np.zeros((2, 1), dtype=np.float32), [0, 1], [1.0])


def test_logistic_combiner_survives_a_one_class_fit() -> None:
    """A degenerate fit (one class only) scores every row the same.

    The Stage-3 reporting's inner cross-fit needs it: one document's rows
    can carry a single class, and the gradient-boosted combiner behaves the
    same way.
    """
    combiner = build_combiner("logistic", {})
    combiner.fit(np.zeros((4, 1), dtype=np.float32), [1, 1, 1, 1])
    scores = combiner.score(np.zeros((3, 1), dtype=np.float32))
    assert scores.shape == (3,)
    assert set(scores.tolist()) == {0.0}
    # a degenerate fit has no coefficients, and says so
    assert combiner.coefficients() == {}
    assert combiner.describe()["fitted"] is True


def test_histgb_combiner_survives_a_one_class_fit() -> None:
    """A degenerate fit (one class) scores every row the same."""
    combiner = build_combiner("histgb", {})
    combiner.fit(np.zeros((4, 2), dtype=np.float32), [1, 1, 1, 1])
    scores = combiner.score(np.zeros((3, 2), dtype=np.float32))
    assert scores.shape == (3,)
    assert set(scores.tolist()) == {0.0}


def test_histgb_combiner_config_rejects_a_degenerate_parameter() -> None:
    """The parameters are validated at config time."""
    with pytest.raises(ValueError, match="max_iter must be"):
        HistGbCombinerConfig(max_iter=0)
    with pytest.raises(ValueError, match="learning_rate must be"):
        HistGbCombinerConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_leaf_nodes must be"):
        HistGbCombinerConfig(max_leaf_nodes=1)
    with pytest.raises(ValueError, match="min_samples_leaf must be"):
        HistGbCombinerConfig(min_samples_leaf=0)
    with pytest.raises(ValueError, match="l2_regularization must be"):
        HistGbCombinerConfig(l2_regularization=-1.0)
    with pytest.raises(ValueError, match="max_depth must be"):
        HistGbCombinerConfig(max_depth=0)
    with pytest.raises(ValueError, match="max_bins must be"):
        HistGbCombinerConfig(max_bins=1)


def test_histgb_combiner_describes_itself() -> None:
    """The description is the artifact's record of the learner."""
    described = build_combiner("histgb", {}).describe()
    assert described["name"] == "histgb"
    assert described["params"]["max_iter"] == 100
    assert described["params"]["early_stopping"] is False
    assert described["fitted"] is False
    assert "early stopping is off" in described["note"]
