"""The OMCT-side commands: ``probe-omct``, ``analyze-omct``, ``decide-omct``.

``probe-omct`` is the Phase-2 embedding probe on the auxiliary pairwise
data; ``analyze-omct`` is the text-processing track's Stage-0 pair-score
artifact; ``decide-omct`` is the track's Stage-3.5 decision probe. All three
run on the **auxiliary** data: their findings annotate method choice and are
never a UPR metric.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from datetime import date
from typing import Any

from cluster_rag.commands._common import (
    _append_command_rows,
    _cached_embedder,
    _fmt,
    _omct_run_id,
    _require_run_id,
    _write_resolved_config,
)
from cluster_rag.config import AnalysisConfig, AppConfig, ChunkerSpec
from cluster_rag.dataset_stats import file_sha256
from cluster_rag.decision import CachedDecider, SystemOneDecider
from cluster_rag.embed import build_embedder
from cluster_rag.embed.base import token_capability
from cluster_rag.embed.random import RandomEmbedder
from cluster_rag.experiment_table import (
    ANALYZE_OMCT_ARMS_COLUMNS,
    ANALYZE_OMCT_ARMS_KEY_COLUMNS,
    ANALYZE_OMCT_COLUMNS,
    ANALYZE_OMCT_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_COLUMNS,
    ANALYZE_OMCT_STAGE3_KEY_COLUMNS,
    ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS,
    DECIDE_OMCT_COLUMNS,
    DECIDE_OMCT_KEY_COLUMNS,
    PROBE_OMCT_COLUMNS,
    PROBE_OMCT_GROUPS_COLUMNS,
    PROBE_OMCT_GROUPS_KEY_COLUMNS,
    PROBE_OMCT_KEY_COLUMNS,
)
from cluster_rag.omct_analysis import (
    BASELINE_ARM,
    BASELINE_CHUNKER,
    DECISION_ARM,
    DECISION_RULE_FIXED_CUT,
    BaselineArm,
    ChunkedSide,
    LateSide,
    ReportingSpec,
    build_analysis_report,
    build_dataset_view,
    build_stage3_grid,
    compare_arms,
    decision_rule_block,
    score_decision_arm,
    score_model_arms,
    score_stage3_model,
    shortlist_rerank_block,
    side_cache_versions,
)
from cluster_rag.omct_decision import DecisionArm, run_decision_arm, score_decision_grid
from cluster_rag.omct_probe import ProbeCacheOptions, run_probe
from cluster_rag.pairs import load_pairs
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    build_preprocess_spec,
    question_transform_version,
)
from cluster_rag.text_processors.aggregation import parse_aggregations
from cluster_rag.text_processors.chunkers import build_chunker
from cluster_rag.text_processors.combining import COMBINER_LOGISTIC, build_combiner
from cluster_rag.text_processors.features import (
    FeatureEmbedders,
    PairFeatureExtractor,
    ProfileUnits,
)

logger = logging.getLogger(__name__)


def _cmd_probe_omct(config: AppConfig) -> int:
    """Run probe-omct: the Phase 2 auxiliary evidence run (results/omct-<run_id>/)."""
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.embed is None:
        raise ValueError(
            "the embed: section (backend, model_id, prefixes) is required for probe-omct"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run the omct probe")

    pair_report = load_pairs(config.paths.data_external)
    spec = build_preprocess_spec(config.preprocess)
    embedder = build_embedder(config.embed)

    probe_report = run_probe(
        pair_report.records,
        embedder,
        preprocess_spec=spec,
        probe_config=config.probe,
        cache=ProbeCacheOptions(root=str(config.paths.cache)) if config.cache.enabled else None,
    )
    embedder_info = embedder.info
    omct_run_id = _omct_run_id(run_id)

    payload = {
        "run_id": omct_run_id,
        "command": "probe-omct",
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": pair_report.n_valid,
        "n_malformed_pairs": pair_report.n_invalid,
        "embedder": probe_report["embedder"],
        "cache": (
            {"enabled": True, "path": str(config.paths.cache), **(probe_report["cache"] or {})}
            if config.cache.enabled
            else {"enabled": False}
        ),
        "preprocess": {
            "variant": spec.variant,
            "version": spec.version,
            "openers": list(spec.openers),
        },
        "probe": probe_report,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "probe_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    variant_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for variant in payload["probe"]["variants"]:
        same_doc = variant["retrieval"]["same_document"]
        share_over = variant["truncation"].get("share_paragraphs_over_context")
        variant_rows.append(
            {
                "run_id": omct_run_id,
                "date": date.today().isoformat(),
                "backend": embedder_info.backend,
                "model": embedder_info.model_id,
                "model_version": (embedder_info.revision_or_digest or "")[:12],
                "dim": embedder_info.dimension,
                "ctx": embedder_info.context_tokens,
                "transform": variant["transform"],
                "pair_auc": _fmt(variant["pair_auc"], 4),
                "same-doc hit@1": _fmt(same_doc.get("hit@1"), 3),
                "same-doc hit@10": _fmt(same_doc.get("hit@10"), 3),
                "same-doc recall@10": _fmt(same_doc.get("recall@10"), 3),
                "paras>ctx": None if share_over is None else f"{share_over:.1%}",
                "n_paras": variant["retrieval"]["all"]["n_rows"],
                "n_pairs(pos)": (
                    f"{payload['probe']['corpus']['n_pairs']}"
                    f" ({payload['probe']['corpus']['n_positive_pairs']})"
                ),
                "seconds": variant["seconds"],
            }
        )
        for group in variant["group_breakdown"]["groups"]:
            scope = group["same_document"]
            group_rows.append(
                {
                    "run_id": omct_run_id,
                    "transform": variant["transform"],
                    "document": group["document"],
                    "source_csv": group["source_csv"],
                    "n_pairs": group["n_pairs"],
                    "pos": group["n_positive"],
                    "neg": group["n_negative"],
                    "pair_auc": _fmt(group["pair_auc"], 4),
                    "same-doc hit@1": None if scope is None else _fmt(scope.get("hit@1"), 3),
                    "same-doc hit@10": None if scope is None else _fmt(scope.get("hit@10"), 3),
                    "n_rows_with_gold": group["n_rows_with_gold"],
                }
            )
    _append_command_rows(
        config,
        "probe-omct",
        columns=PROBE_OMCT_COLUMNS,
        key_columns=PROBE_OMCT_KEY_COLUMNS,
        rows=variant_rows,
    )
    _append_command_rows(
        config,
        "probe-omct",
        columns=PROBE_OMCT_GROUPS_COLUMNS,
        key_columns=PROBE_OMCT_GROUPS_KEY_COLUMNS,
        rows=group_rows,
        groups=True,
    )

    print(f"OMCT probe: {omct_run_id} with {embedder_info.backend}:{embedder_info.model_id}")
    print(
        f"  corpus: {pair_report.n_valid} judged pairs "
        f"({payload['probe']['corpus']['n_positive_pairs']} positive), "
        f"{payload['probe']['corpus']['n_distinct_paragraph_texts']} distinct paragraphs, "
        f"{payload['probe']['corpus']['n_distinct_question_texts']} distinct questions"
    )
    for variant in payload["probe"]["variants"]:
        same_doc_hit1 = variant["retrieval"]["same_document"]["hit@1"]
        trunc_share = variant["truncation"].get("share_paragraphs_over_context")
        trunc_text = "n/a" if trunc_share is None else f"{trunc_share:.1%}"
        print(
            f"  [{variant['transform']:>11s}] pair-AUC {variant['pair_auc']:.4f}; "
            f"hit@1 {variant['retrieval']['all']['hit@1']:.3f} (all) / "
            f"{same_doc_hit1:.3f} (same-doc); hit@10 "
            f"{variant['retrieval']['same_document']['hit@10']:.3f} (same-doc); "
            f"paragraphs over context: {trunc_text}"
        )
        for group in variant["group_breakdown"]["groups"]:
            auc_text = "n/a" if group["pair_auc"] is None else f"{group['pair_auc']:.4f}"
            if group["same_document"] is None:
                same_doc_text = "no gold rows"
            else:
                scope = group["same_document"]
                same_doc_text = (
                    f"hit@1 {scope['hit@1']:.3f} / hit@10 {scope['hit@10']:.3f} "
                    f"over {group['n_rows_with_gold']} gold row(s)"
                )
            print(
                f"      {group['document']} (csv {group['source_csv']}): "
                f"pairs {group['n_pairs']} ({group['n_positive']} pos, "
                f"{group['n_negative']} neg), rows {group['n_rows']} | "
                f"pair-AUC {auc_text} | same-doc {same_doc_text}"
            )
    print(f"Report written to {results_dir / 'probe_report.json'}")
    return 0


def _within_ci(comparison: dict[str, Any] | None) -> str | None:
    """The CI of the within-document AUC difference (the transferable part)."""
    if not comparison:
        return None
    paired = comparison.get("auc_decomposition", {}).get("within_group_paired")
    if not paired:
        return None
    return f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]"


def _analyze_omct_row(
    model: dict[str, Any],
    *,
    run_id: str,
    arm_name: str,
    arm: dict[str, Any],
    comparison: dict[str, Any] | None,
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the analysis: the model, the arm and its numbers."""
    info = model["embedder"]
    distributions = arm["score_distributions"]
    enrichment = arm["negative_enrichment"]["rank_percentile"]
    paired = comparison["paired_auc"] if comparison else None
    paragraph_side = arm["sides"]["paragraph"]
    n_texts = paragraph_side.get("n_texts") or 0
    n_chunks = paragraph_side.get("n_chunks") or 0
    row: dict[str, Any] = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info["backend"],
        "model": info["model_id"],
        "model_version": (info["revision_or_digest"] or "")[:12],
        "dim": info["dimension"],
        "ctx": info["context_tokens"],
        "arm": arm_name,
        "chunker": (
            paragraph_side.get("side_slug")
            or paragraph_side.get("chunker_slug")
            or paragraph_side.get("chunker_name")
        ),
        "aggregation": (arm.get("aggregation") or {}).get("spec"),
        "transform": arm["sides"]["question"].get("transform"),
        "pair_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "macro_auc": _fmt(arm["pair_auc"].get("macro"), 4),
        "within_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("within_group_auc"), 4),
        "cross_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("cross_group_auc"), 4),
        "within_d_auc": _fmt(
            (comparison or {}).get("auc_decomposition", {}).get("within_group_delta"), 4
        ),
        "within_d_auc_ci": _within_ci(comparison),
        "d_auc": _fmt(paired["delta_auc"], 4) if paired else None,
        "d_auc_ci": (f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None),
        "deshift_auc": (_fmt(comparison["deshifted_auc"]["overall"], 4) if comparison else None),
        "floor_gap_delta": (_fmt(comparison["floor_gap"]["delta"], 4) if comparison else None),
        "ctrl_mean": _fmt(distributions["control"].get("mean"), 4),
        "ctrl_mean_delta": (
            _fmt(comparison["control_shift"]["mean_delta"], 4) if comparison else None
        ),
        "ctrl_p50": _fmt(distributions["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50": _fmt(enrichment["control"].get("quantiles", {}).get("0.5"), 4),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "chunks_per_text": (round(n_chunks / n_texts, 2) if n_texts else None),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    return row


def _chunker_with_embedder(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
    *,
    threshold: float | None = None,
    base_version: str | None = None,
    cache: bool = True,
) -> tuple[Any, Any]:
    """Build a chunker and the embedder of its chunks.

    The semantic chunker finds its boundaries from the **atom** embeddings, so
    it takes the ``Embedder`` port: the CLI wires the cached backend in here
    (the package itself never imports a model client). The atoms go through
    the atoms chunker's cache version, the chunks through the chunker's own
    version — which carries the swept threshold, so two thresholds never
    share cached chunks.

    ``base_version`` overrides the version slot the chunks are cached under
    (the paragraph side's preprocessing version by default; the question side
    uses the question transform's version).

    ``cache=False`` returns the embedder as it is: the Stage-3 random-vector
    floor passes its own (uncached) embedder here, so the random vectors never
    enter the model's embedding cache.
    """
    params = dict(chunker_spec.params)
    if threshold is not None:
        params["threshold"] = threshold

    def _side(version: str) -> Any:
        """One side's embedder: the cached backend, or the embedder as it is."""
        return _cached_embedder(embedder, config, version) if cache else embedder

    if chunker_spec.name == "semantic":
        atom_params = {"split_clauses": bool(params.get("atom_split_clauses", False))}
        atom_chunker = build_chunker("atoms", atom_params)
        atom_version, _ = side_cache_versions(
            atom_chunker, spec, question_transform, question_block_markers
        )
        chunker = build_chunker(
            "semantic",
            params,
            embedder=_side(atom_version),
        )
    else:
        chunker = build_chunker(chunker_spec.name, params)
    if base_version is None:
        paragraph_version, _ = side_cache_versions(
            chunker, spec, question_transform, question_block_markers
        )
    else:
        paragraph_version = chunker.cache_version(base_version)
    return chunker, _side(paragraph_version)


def _chunked_side(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    question_chunker_spec: ChunkerSpec | None,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
) -> ChunkedSide:
    """The chunked paragraph side and, for the symmetric variants, the question side."""
    chunker, paragraph_embedder = _chunker_with_embedder(
        config, embedder, chunker_spec, spec, question_transform, question_block_markers
    )
    question_chunker = None
    question_embedder = None
    if question_chunker_spec is not None:
        question_chunker, question_embedder = _chunker_with_embedder(
            config,
            embedder,
            question_chunker_spec,
            spec,
            question_transform,
            question_block_markers,
        )
    return ChunkedSide(
        chunker=chunker,
        paragraph_embedder=paragraph_embedder,
        question_chunker=question_chunker,
        question_embedder=question_embedder,
    )


def _late_side(
    config: AppConfig,
    embedder: Any,
    analysis: AnalysisConfig,
) -> LateSide:
    """The late-chunked side (Stage 2.7): the chunker, the embedder, the poolings.

    The late side embeds the **whole document once** and pools the
    token-level embeddings per chunk, so it needs an embedder that can
    return token-level embeddings. The capability is checked here, before any
    work: a backend without it (Ollama's ``/api/embed`` returns one vector
    per text and nothing else) gets a clear message instead of a failure
    deep inside the run.

    The side uses the **raw** backend, not the cached one: the token-level
    pass is not cached (the cache keys on the chunk's text, and a
    late-chunked vector is a function of the whole document), so the cache
    wrapper would only add a layer without a hit.
    """
    if analysis.late_chunker is None:
        raise ValueError("the late-chunked side needs analysis.late_chunker")
    if token_capability(embedder) is None:
        backend = getattr(getattr(embedder, "info", None), "backend", "unknown")
        raise ValueError(
            "the late-chunked arm needs token-level embeddings; the backend "
            f"{backend!r} cannot return them (Ollama's /api/embed returns one "
            "vector per text and nothing else), so the arm is HF-only"
        )
    chunker = build_chunker(analysis.late_chunker.name, analysis.late_chunker.params)
    question_chunker = None
    if analysis.late_question_chunker is not None:
        question_chunker = build_chunker(
            analysis.late_question_chunker.name, analysis.late_question_chunker.params
        )
    return LateSide(
        chunker=chunker,
        paragraph_embedder=embedder,
        question_chunker=question_chunker,
        question_embedder=embedder if question_chunker is not None else None,
        poolings=tuple(analysis.late_pooling),
    )


def _swept_sides(
    config: AppConfig,
    embedder: Any,
    chunker_spec: ChunkerSpec,
    spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
    thresholds: Sequence[float],
) -> list[ChunkedSide]:
    """One chunked side per point of the swept grid (the semantic threshold)."""
    sides: list[ChunkedSide] = []
    for threshold in thresholds:
        chunker, paragraph_embedder = _chunker_with_embedder(
            config,
            embedder,
            chunker_spec,
            spec,
            question_transform,
            question_block_markers,
            threshold=threshold,
        )
        sides.append(
            ChunkedSide(
                chunker=chunker,
                paragraph_embedder=paragraph_embedder,
                threshold=threshold,
            )
        )
    return sides


def _stage3_extractor(
    config: AppConfig,
    embedder: Any,
    analysis: AnalysisConfig,
    spec: PreprocessSpec,
    view: Any,
    *,
    paragraph_embedder: Any,
    question_embedder: Any,
    feature_embedder: Any | None = None,
) -> PairFeatureExtractor:
    """The Stage-3 feature pack, with the cached backends wired in.

    The pack takes the ``Embedder`` **port** (the same precedent as the
    semantic chunker): the package never imports a model client, the CLI
    wires the cached backends in here. Every side has its own cache-version
    slot, so they are separate ports: the whole paragraph (the baseline arm's
    paragraph slot), the whole question (the baseline arm's question slot),
    the chunks (the chunker's slot) and the Stage-3.3 chunk profile's four
    unit sides (the sentences and the word pieces, on each side).

    ``feature_embedder`` is the **Stage-3 random-vector floor** (CS-6d): when
    it is given, every embedding side of the pack is that embedder instead of
    the model's (cached) backend, so the floor replaces the embedding
    geometry and nothing else. It is never cached, so the random vectors never
    enter the model's embedding cache.
    """
    features = analysis.stage3.features
    random_mode = feature_embedder is not None
    paragraph_chunker, chunk_embedder = _chunker_with_embedder(
        config,
        feature_embedder if random_mode else embedder,
        features.paragraph_chunker,
        spec,
        analysis.question_transform,
        analysis.question_block_markers,
        cache=not random_mode,
    )
    question_chunks_embedder = None
    if features.question_chunker is not None:
        question_version = question_transform_version(
            analysis.question_transform, analysis.question_block_markers
        )
        _, question_chunks_embedder = _chunker_with_embedder(
            config,
            feature_embedder if random_mode else embedder,
            features.question_chunker,
            spec,
            analysis.question_transform,
            analysis.question_block_markers,
            base_version=question_version,
            cache=not random_mode,
        )
    # The chunk profile's units (the sentences and the word pieces). The
    # paragraph side is cached under the preprocessing version, the question
    # side under the question transform's version (the profile's question
    # units are cut from the transformed question).
    question_version = question_transform_version(
        analysis.question_transform, analysis.question_block_markers
    )
    sentence_chunker = build_chunker("units", {"kind": "sentence"})
    ngram_chunker = build_chunker(
        "units", {"kind": "ngram", "n_words": features.profile_ngram_size}
    )
    profile_units = ProfileUnits(sentences=sentence_chunker, ngrams=ngram_chunker)
    question_embedding_texts = dict(
        zip(
            view.questions,
            apply_question_transform(
                list(view.questions),
                analysis.question_transform,
                analysis.question_block_markers,
            ),
            strict=True,
        )
    )

    def _side(version: str) -> Any:
        """One side's embedder: the cached model backend, or the random one."""
        if random_mode:
            return feature_embedder
        return _cached_embedder(embedder, config, version)

    return PairFeatureExtractor(
        FeatureEmbedders(
            paragraph=feature_embedder if random_mode else paragraph_embedder,
            question=feature_embedder if random_mode else question_embedder,
            chunks=chunk_embedder,
            question_chunks=question_chunks_embedder,
            paragraph_sentences=_side(sentence_chunker.cache_version(spec.version)),
            question_sentences=_side(sentence_chunker.cache_version(question_version)),
            paragraph_ngrams=_side(ngram_chunker.cache_version(spec.version)),
            question_ngrams=_side(ngram_chunker.cache_version(question_version)),
        ),
        paragraph_chunker=paragraph_chunker,
        top_k=features.top_k,
        question_embedding_texts=question_embedding_texts,
        groups=features.groups,
        profile_units=profile_units,
        profile_top_k=features.profile_top_k,
    )


def _stage3_reporting_table_values(stage3: dict[str, Any]) -> dict[str, Any]:
    """The reporting's table cells: the calibration, the threshold and the rule.

    The rule's cells are the **two-part** variant (the cosine's margin *and*
    the arm's margin below their paragraph's best candidate — CS-6d's
    "retrieve with the cosine, rank with the learned arm") and the
    control-role floor's harm at the same reach: the rule's own numbers
    against the floor, in the rule's own units.
    """
    reporting = stage3.get("reporting") or {}
    calibration = reporting.get("calibration") or {}
    arm_calibration = calibration.get("arm") or {}
    cosine_calibration = calibration.get("cosine") or {}
    thresholds = reporting.get("thresholds") or {}
    per_document = thresholds.get("per_document") or {}
    recalls = [
        ((block.get("holdout") or {}).get("operating_point") or {}).get("recall")
        for block in per_document.values()
    ]
    usable = [value for value in recalls if value is not None]
    macro_recall = sum(usable) / len(usable) if usable else None
    out_of_fold = thresholds.get("out_of_fold") or {}
    median_operating = out_of_fold.get("operating_point_at_median") or {}
    variants = (reporting.get("safe_negative") or {}).get("variants") or {}
    rule = variants.get("both_margins") or {}
    pooled = rule.get("pooled") or {}
    floor = rule.get("floor_arm") or {}
    matched = floor.get("harm_at_matched_coverage") or {}
    return {
        "ece_arm": _fmt((arm_calibration.get("pooled") or {}).get("ece"), 4),
        "ece_arm_macro": _fmt(arm_calibration.get("macro_ece"), 4),
        "ece_cos": _fmt((cosine_calibration.get("pooled") or {}).get("ece"), 4),
        "brier_arm": _fmt((arm_calibration.get("pooled") or {}).get("brier"), 4),
        "thr_criterion": thresholds.get("criterion"),
        "thr_macro_recall": _fmt(macro_recall, 4),
        "thr_oof_recall": _fmt(median_operating.get("recall"), 4),
        "rule_cov": _fmt(pooled.get("coverage"), 4),
        "rule_harm": _fmt(pooled.get("harm"), 4),
        "rule_floor_harm": _fmt(matched.get("harm"), 4),
    }


def _stage3_table_row(
    model: dict[str, Any],
    *,
    run_id: str,
    stage3: dict[str, Any],
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the Stage-3 arm: the model, the arm and its numbers."""
    info = model["embedder"]
    comparison = stage3["comparison"]
    decomposition = comparison["auc_decomposition"]
    paired = comparison["paired_auc"]
    within_paired = decomposition.get("within_group_paired") or {}
    control = comparison["control_side"]["arm"]
    enrichment = comparison["negative_enrichment"]["rank_percentile"]
    feature_pack = stage3["feature_pack"]
    floors = stage3["permutation_floor"]
    hit_at_k = comparison["hit_at_k"]
    row = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info["backend"],
        "model": info["model_id"],
        "model_version": (info["revision_or_digest"] or "")[:12],
        "dim": info["dimension"],
        "ctx": info["context_tokens"],
        "groups": "+".join(feature_pack.get("groups") or []) or None,
        "features": feature_pack.get("n_features"),
        "combiner": (stage3.get("combiner") or {}).get("name"),
        "within_auc": _fmt(decomposition["arm"].get("within_group_auc"), 4),
        "within_d_auc": _fmt(decomposition.get("within_group_delta"), 4),
        "within_d_auc_ci": (
            f"[{within_paired['ci_low']:+.4f}, {within_paired['ci_high']:+.4f}]"
            if within_paired
            else None
        ),
        "pooled_auc": _fmt(comparison["pair_auc"].get("arm"), 4),
        "pooled_d_auc": _fmt(paired.get("delta_auc"), 4),
        "pooled_d_auc_ci": (
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None
        ),
        "macro_auc": _fmt(comparison["pair_auc"].get("macro", {}).get("arm"), 4),
        "perm_auc_within": _fmt(floors.get("within_document", {}).get("chance"), 4),
        "ctrl_role_auc_within": _fmt(floors.get("within_document", {}).get("control_role"), 4),
        "ctrl_mean": _fmt(control.get("mean"), 4),
        "ctrl_rank_p50": _fmt(control.get("rank_p50"), 4),
        "ctrl_rank_p50_delta": _fmt(comparison["control_side"].get("rank_p50_delta"), 4),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "floor_gap": _fmt(comparison["floor_gap"]["arm"].get("standardized"), 4),
        "hit1_pos": _fmt(hit_at_k["arm"]["label1"].get("hit@1"), 4),
        "hit1_pos_ref": _fmt(hit_at_k["reference"]["label1"].get("hit@1"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    if stage3.get("reporting"):
        row.update(_stage3_reporting_table_values(stage3))
    return row


def _decision_arm_row(
    model: dict[str, Any],
    *,
    run_id: str,
    arm_name: str,
    arm: dict[str, Any],
    comparison: dict[str, Any] | None,
    n_pairs: int,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the decision arm (CS-7b).

    The row fills both table shapes' keys (the arms shape and the Stage-3
    shape), so the decision arm's numbers show up whichever shape the run's
    section uses. The Stage-3-only keys (the feature groups, the combiner, the
    permutation floors) are **n/a** for a decision arm: it is not a learned
    arm, it is a frozen zero-shot scorer.
    """
    info = model["model"]
    distributions = arm["score_distributions"]
    enrichment = arm["negative_enrichment"]["rank_percentile"]
    paired = comparison["paired_auc"] if comparison else None
    decomposition = comparison.get("auc_decomposition", {}) if comparison else {}
    within_paired = decomposition.get("within_group_paired") or {}
    hit_at_k = comparison["hit_at_k"] if comparison else {}
    rule = (model.get("rule") or {}).get("pooled") or {}
    fixed = rule.get(DECISION_RULE_FIXED_CUT) or {}
    counterpart = fixed.get("counterpart") or {}
    matched = counterpart.get("harm_at_matched_coverage") or {}
    row: dict[str, Any] = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "backend": info.get("backend"),
        "model": info.get("model_id"),
        "model_version": (info.get("digest") or "")[:12],
        "dim": None,
        "ctx": None,
        "arm": arm_name,
        "chunker": None,
        "aggregation": None,
        "transform": "strip_block",
        "groups": None,
        "features": None,
        "combiner": None,
        "pair_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "pooled_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "pooled_d_auc": _fmt(paired.get("delta_auc"), 4) if paired else None,
        "pooled_d_auc_ci": (
            f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None
        ),
        "macro_auc": _fmt(arm["pair_auc"].get("macro"), 4),
        "within_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("within_group_auc"), 4),
        "cross_auc": _fmt((arm.get("pair_auc_decomposition") or {}).get("cross_group_auc"), 4),
        "within_d_auc": _fmt(decomposition.get("within_group_delta"), 4),
        "within_d_auc_ci": (
            f"[{within_paired['ci_low']:+.4f}, {within_paired['ci_high']:+.4f}]"
            if within_paired
            else None
        ),
        "d_auc": _fmt(paired.get("delta_auc"), 4) if paired else None,
        "d_auc_ci": (f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]" if paired else None),
        "deshift_auc": (_fmt(comparison["deshifted_auc"]["overall"], 4) if comparison else None),
        "floor_gap_delta": (_fmt(comparison["floor_gap"]["delta"], 4) if comparison else None),
        "floor_gap": _fmt((comparison or {}).get("floor_gap", {}).get("arm"), 4),
        "ctrl_mean": _fmt(distributions["control"].get("mean"), 4),
        "ctrl_mean_delta": (
            _fmt(comparison["control_shift"]["mean_delta"], 4) if comparison else None
        ),
        "ctrl_p50": _fmt(distributions["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50": _fmt(enrichment["control"].get("quantiles", {}).get("0.5"), 4),
        "ctrl_rank_p50_delta": (
            _fmt(comparison["control_shift"].get("rank_p50_delta"), 4) if comparison else None
        ),
        "pos_rank_p50": _fmt(enrichment["label1"].get("quantiles", {}).get("0.5"), 4),
        "neg_rank_p50": _fmt(enrichment["label0"].get("quantiles", {}).get("0.5"), 4),
        "chunks_per_text": None,
        "perm_auc_within": None,
        "ctrl_role_auc_within": None,
        "hit1_pos": _fmt((hit_at_k.get("arm") or {}).get("label1", {}).get("hit@1"), 4),
        "hit1_pos_ref": _fmt((hit_at_k.get("reference") or {}).get("label1", {}).get("hit@1"), 4),
        # the decision rule: the fixed cut's reach and harm, and the
        # label-blind counterpart's harm at the same reach
        "rule_cov": _fmt(fixed.get("coverage"), 4),
        "rule_harm": _fmt(fixed.get("harm"), 4),
        "rule_floor_harm": _fmt(matched.get("harm"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }
    return row


def _cmd_analyze_omct(config: AppConfig) -> int:
    """Run analyze-omct: the text-processing track's Stage-0 artifact.

    Writes ``results/omct-<run_id>/{config.yaml, analysis.json}``: the
    per-pair scores of the judged pairs, the control set (the sampled
    unjudged same-document cells) with their scores and their provenance, the
    per-document strata with their base rates, the score distributions per
    label and the negative-enrichment measurement — per model. The command
    adds the ``omct-`` dataset prefix itself (AGENTS.md run-id grammar), and
    the row lands in the track's own table file (the command-derived
    (dataset, track) mapping).

    When the config asks for structural arms (a chunker with parameters plus
    aggregation rules) or for extra question transforms, the same run also
    scores those arms and compares every one of them with the full-text
    baseline arm — paired (the same pairs) and with the control-side
    counterpart next to the discrimination number.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.analysis is None:
        raise ValueError(
            "the analysis: section (the models to score with and the control design) "
            "is required for the `analyze-omct` command"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run analyze-omct")
    if not config.analysis.embedders:
        raise ValueError("analysis.embedders must name at least one model to score with")

    analysis = config.analysis
    pair_report = load_pairs(config.paths.data_external)
    spec = build_preprocess_spec(config.preprocess)

    baseline_chunker = build_chunker(BASELINE_CHUNKER)
    aggregations = parse_aggregations(analysis.aggregations)
    question_transforms = (analysis.question_transform, *analysis.question_transforms)
    bootstrap_seed = analysis.bootstrap.seed if analysis.bootstrap.seed is not None else config.seed

    view = build_dataset_view(
        pair_report.records,
        control_size=analysis.control_size,
        seed=config.seed,
        sampling_rule=analysis.sampling_rule,
    )

    models: list[dict[str, Any]] = []
    for entry in analysis.embedders:
        embedder = build_embedder(entry.embed)
        paragraph_version, _ = side_cache_versions(
            baseline_chunker, spec, analysis.question_transform, analysis.question_block_markers
        )
        paragraph_embedder = _cached_embedder(embedder, config, paragraph_version)
        question_embedders: dict[str, Any] = {}
        for transform in question_transforms:
            _, question_version = side_cache_versions(
                baseline_chunker, spec, transform, analysis.question_block_markers
            )
            question_embedders[transform] = _cached_embedder(embedder, config, question_version)

        # The chunked arms: one fixed chunker, or one side per point of the
        # swept grid (the semantic threshold of Stage 2.3).
        chunked_side = None
        header_side = None
        late_side = None
        swept_sides: list[ChunkedSide] = []
        if aggregations:
            if analysis.semantic_thresholds:
                swept_sides = _swept_sides(
                    config,
                    embedder,
                    analysis.chunker,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                    analysis.semantic_thresholds,
                )
            else:
                chunked_side = _chunked_side(
                    config,
                    embedder,
                    analysis.chunker,
                    analysis.question_chunker,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                )
            if analysis.header_chunker is not None:
                # The chunk header (Stage 2.6): a second chunked side whose
                # chunks carry the paragraph's own first sentence, so the
                # header is measured against the no-header arm of this run.
                header_side = _chunked_side(
                    config,
                    embedder,
                    analysis.header_chunker,
                    None,
                    spec,
                    analysis.question_transform,
                    analysis.question_block_markers,
                )
        if analysis.late_chunker is not None:
            # The late chunking (Stage 2.7): the document embedded once and
            # the token-level embeddings pooled per chunk. Its arms run
            # beside the run's early-chunked arms, so the late geometry is
            # measured against the early one in the same run.
            late_side = _late_side(config, embedder, analysis)
        baseline = BaselineArm()
        model_block = score_model_arms(
            view,
            model_name=entry.name,
            paragraph_embedder=paragraph_embedder,
            question_embedders=question_embedders,
            baseline_chunker=baseline_chunker,
            question_transform=analysis.question_transform,
            question_block_markers=analysis.question_block_markers,
            preprocess_spec=spec,
            chunked_side=chunked_side,
            header_side=header_side,
            late_side=late_side,
            swept_sides=swept_sides,
            aggregations=aggregations,
            semantic_selection=analysis.semantic_selection,
            bootstrap_resamples=analysis.bootstrap.resamples,
            bootstrap_seed=bootstrap_seed,
            baseline=baseline,
        )
        # The learned arm (Stage 3.1): the feature pack + the combiner behind
        # the document-level holdout. The reference is the baseline arm's
        # full-text cosine on the same held-out documents.
        if analysis.stage3 is not None:
            # The Stage-3 random-vector floor (CS-6d): the feature pack's
            # embedding sides are random unit vectors instead of the model's.
            # The run's baseline arm (the full-text cosine) keeps the model's
            # embeddings, so the guard still holds and the arm-vs-cosine
            # comparison stays meaningful.
            feature_embedder = None
            embeddings_provenance: dict[str, Any] = {
                "mode": analysis.stage3.embeddings,
                "note": (
                    "the feature pack's embedding sides come from the run's "
                    "model (through the disk cache)"
                ),
            }
            if analysis.stage3.embeddings == "random":
                feature_embedder = RandomEmbedder(
                    dimension=int(embedder.info.dimension),
                    seed=config.seed,
                )
                embeddings_provenance = feature_embedder.provenance()
            extractor = _stage3_extractor(
                config,
                embedder,
                analysis,
                spec,
                view,
                paragraph_embedder=paragraph_embedder,
                question_embedder=question_embedders[analysis.question_transform],
                feature_embedder=feature_embedder,
            )
            model_block["stage3"] = score_stage3_model(
                view,
                model_name=entry.name,
                extractor=extractor,
                combiner_factory=lambda: build_combiner(
                    analysis.stage3.combiner.name, analysis.stage3.combiner.params
                ),
                reference_similarities=baseline.similarities,
                control_weight=analysis.stage3.control_weight,
                permutation_resamples=analysis.stage3.permutation_resamples,
                seed=config.seed,
                bootstrap_resamples=analysis.bootstrap.resamples,
                bootstrap_seed=bootstrap_seed,
                feature_embeddings=embeddings_provenance,
                # The cosine's calibration map (CS-6e): a logistic regression
                # on the cosine as its single feature, fitted on the fit side
                # only. It is always the logistic regression, whatever the
                # arm's own combiner is: the map is a calibration, not a model.
                calibration_map_factory=lambda: build_combiner(
                    COMBINER_LOGISTIC, {"C": 1.0, "max_iter": 1000, "standardize": True}
                ),
                reporting=ReportingSpec(
                    enabled=analysis.stage3.reporting.enabled,
                    bins=analysis.stage3.reporting.bins,
                    threshold_criterion=analysis.stage3.reporting.threshold_criterion,
                    threshold_quantiles=analysis.stage3.reporting.threshold_quantiles,
                    margin_quantiles=analysis.stage3.reporting.margin_quantiles,
                ),
            )
        models.append(model_block)

    # The decision arm (CS-7b): the decision model scores the **whole**
    # same-document grid, so it becomes an arm of this run — the same cells,
    # the same candidate lists, the same control, and the paired comparison
    # against the full-text cosine. The decision scorer is not an embedding
    # model, so it enters the artifact as its own entry.
    if analysis.decision is not None:
        reporting = ReportingSpec(
            enabled=True,
            bins=analysis.stage3.reporting.bins if analysis.stage3 is not None else 10,
            threshold_criterion=(
                analysis.stage3.reporting.threshold_criterion
                if analysis.stage3 is not None
                else "floor_gap"
            ),
            threshold_quantiles=(
                analysis.stage3.reporting.threshold_quantiles if analysis.stage3 is not None else 99
            ),
            margin_quantiles=(
                analysis.stage3.reporting.margin_quantiles if analysis.stage3 is not None else 49
            ),
        )
        for model_config in analysis.decision.models:
            decider = build_decider(model_config, config)
            arm = DecisionArm(
                name=model_config.name,
                state=model_config.state,
                question_type=model_config.question_type,
                instructions=model_config.instructions,
                options=model_config.options,
                criteria=model_config.criteria,
                positive_option=model_config.positive_option,
                max_input_tokens=model_config.max_input_tokens,
            )
            decision_grid = score_decision_grid(
                view,
                decider,
                arm,
                question_transform=analysis.question_transform,
                question_block_markers=analysis.question_block_markers,
                max_questions_per_call=analysis.decision.max_questions_per_call,
            )
            mismatched_grid = None
            if analysis.decision.mismatched_control:
                mismatched_grid = score_decision_grid(
                    view,
                    decider,
                    arm,
                    question_transform=analysis.question_transform,
                    question_block_markers=analysis.question_block_markers,
                    max_questions_per_call=analysis.decision.max_questions_per_call,
                    mismatched=True,
                )
            decision_block = score_decision_arm(view, decision=decision_grid)
            comparison = compare_arms(
                view,
                arm_name=DECISION_ARM,
                reference_similarities=baseline.similarities,
                arm_similarities=decision_grid.matrix,
                resamples=analysis.bootstrap.resamples,
                seed=bootstrap_seed,
            )
            rule_block = decision_rule_block(
                view,
                grid=build_stage3_grid(view),
                scores=decision_grid.scores,
                mismatched_scores=(None if mismatched_grid is None else mismatched_grid.scores),
                reporting=reporting,
            )
            info = decider.info
            models.append(
                {
                    "name": f"decision-{model_config.name}",
                    "kind": "decision",
                    "model": {
                        "name": model_config.name,
                        "model_id": info.model_id,
                        "digest": info.digest,
                        "backend": info.backend,
                        "host": info.host,
                        "prompt_version": analysis.decision.prompt_version,
                        "state": model_config.state,
                        "question_type": model_config.question_type,
                        "instructions": model_config.instructions,
                        "options": list(model_config.options),
                        "criteria": dict(model_config.criteria),
                        "positive_option": model_config.positive_option,
                        "max_input_tokens": model_config.max_input_tokens,
                        "extra": info.extra,
                    },
                    "arms": {DECISION_ARM: decision_block},
                    "comparisons": {DECISION_ARM: comparison},
                    "rule": rule_block,
                    "shortlist_rerank": shortlist_rerank_block(
                        view,
                        reference_similarities=baseline.similarities,
                        decision_matrix=decision_grid.matrix,
                        ks=analysis.decision.rerank_ks,
                    ),
                    "truncation": {
                        "n_cells": decision_grid.n_cells,
                        "n_truncated": decision_grid.n_truncated,
                        "share": (
                            decision_grid.n_truncated / decision_grid.n_cells
                            if decision_grid.n_cells
                            else None
                        ),
                        "mismatched": {
                            "n_cells": 0 if mismatched_grid is None else mismatched_grid.n_cells,
                            "n_truncated": (
                                0 if mismatched_grid is None else mismatched_grid.n_truncated
                            ),
                        },
                        "note": (
                            "the cells whose paragraph was cut to fit the model's "
                            "input budget (the question and the instructions are "
                            "never cut): a shortened input is never silent"
                        ),
                    },
                    "cache": {
                        "hits": getattr(decider, "cache_hits", 0),
                        "misses": getattr(decider, "cache_misses", 0),
                        "stored": getattr(decider, "n_stored", 0),
                    },
                    "usage": dict(getattr(decider, "usage", {})),
                    "n_calls_planned": decision_grid.n_calls,
                }
            )

    omct_run_id = _omct_run_id(run_id)
    payload = build_analysis_report(
        view,
        models,
        run_id=omct_run_id,
        source=str(config.paths.data_external),
        source_sha256=file_sha256(config.paths.data_external),
        seed=config.seed,
        control_size=analysis.control_size,
        sampling_rule=analysis.sampling_rule,
        question_transform=analysis.question_transform,
        chunker_name=analysis.chunker.name,
        chunker_params=analysis.chunker.params,
        question_chunker_name=(
            None if analysis.question_chunker is None else analysis.question_chunker.name
        ),
        question_chunker_params=(
            {} if analysis.question_chunker is None else analysis.question_chunker.params
        ),
        semantic_thresholds=analysis.semantic_thresholds,
        semantic_selection=analysis.semantic_selection,
        aggregations=[rule.spec for rule in aggregations],
        question_transforms=list(analysis.question_transforms),
        bootstrap={
            "resamples": analysis.bootstrap.resamples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        stage3=(
            None
            if analysis.stage3 is None
            else {
                "features": {
                    "paragraph_chunker": analysis.stage3.features.paragraph_chunker.model_dump(
                        mode="python"
                    ),
                    "question_chunker": (
                        None
                        if analysis.stage3.features.question_chunker is None
                        else analysis.stage3.features.question_chunker.model_dump(mode="python")
                    ),
                    "top_k": analysis.stage3.features.top_k,
                    "groups": list(analysis.stage3.features.groups),
                    "profile_top_k": analysis.stage3.features.profile_top_k,
                    "profile_ngram_size": analysis.stage3.features.profile_ngram_size,
                },
                "combiner": analysis.stage3.combiner.model_dump(mode="python"),
                "control_weight": analysis.stage3.control_weight,
                "permutation_resamples": analysis.stage3.permutation_resamples,
                "embeddings": analysis.stage3.embeddings,
                "reporting": analysis.stage3.reporting.model_dump(mode="python"),
            }
        ),
        baseline_chunker_name=baseline_chunker.name,
        n_valid_pairs=pair_report.n_valid,
        n_malformed_pairs=pair_report.n_invalid,
        total_seconds=time.perf_counter() - started,
    )
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_rows: list[dict[str, Any]] = []
    has_stage3 = False
    for model in models:
        if model.get("kind") == "decision":
            # The decision arm (CS-7b): its own entry, one row per arm.
            for arm_name, arm in (model.get("arms") or {}).items():
                table_rows.append(
                    _decision_arm_row(
                        model,
                        run_id=omct_run_id,
                        arm_name=arm_name,
                        arm=arm,
                        comparison=(model.get("comparisons") or {}).get(arm_name),
                        n_pairs=view.pooled["n_pairs"],
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            continue
        stage3 = model.get("stage3")
        if stage3 is not None:
            has_stage3 = True
            table_rows.append(
                _stage3_table_row(
                    model,
                    run_id=omct_run_id,
                    stage3=stage3,
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
            continue
        arms = model.get("arms", {})
        comparisons = model.get("comparisons", {})
        if not arms:
            table_rows.append(
                _analyze_omct_row(
                    model,
                    run_id=omct_run_id,
                    arm_name=BASELINE_ARM,
                    arm=model,
                    comparison=None,
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
            continue
        table_rows.append(
            _analyze_omct_row(
                model,
                run_id=omct_run_id,
                arm_name=BASELINE_ARM,
                arm=model,
                comparison=None,
                n_pairs=view.pooled["n_pairs"],
                n_control=view.control.n_cells,
                n_docs=view.pooled["n_documents"],
                seconds=payload["total_seconds"],
            )
        )
        for arm_name, arm in arms.items():
            table_rows.append(
                _analyze_omct_row(
                    model,
                    run_id=omct_run_id,
                    arm_name=arm_name,
                    arm=arm,
                    comparison=comparisons.get(arm_name),
                    n_pairs=view.pooled["n_pairs"],
                    n_control=view.control.n_cells,
                    n_docs=view.pooled["n_documents"],
                    seconds=payload["total_seconds"],
                )
            )
    has_arms = any(model.get("arms") for model in models)
    has_reporting = any((model.get("stage3") or {}).get("reporting") for model in models)
    if has_stage3:
        columns = (
            ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS if has_reporting else ANALYZE_OMCT_STAGE3_COLUMNS
        )
        key_columns = ANALYZE_OMCT_STAGE3_KEY_COLUMNS
    elif has_arms:
        columns = ANALYZE_OMCT_ARMS_COLUMNS
        key_columns = ANALYZE_OMCT_ARMS_KEY_COLUMNS
    else:
        columns = ANALYZE_OMCT_COLUMNS
        key_columns = ANALYZE_OMCT_KEY_COLUMNS
    _append_command_rows(
        config,
        "analyze-omct",
        columns=columns,
        key_columns=key_columns,
        rows=table_rows,
    )

    print(f"OMCT analysis (Stage 0): {omct_run_id}")
    print(
        f"  grid: {view.pooled['grid_cells']} same-document cells, "
        f"{view.pooled['judged_cells']} judged "
        f"({view.pooled['grid_coverage']:.1%} covered), "
        f"{view.pooled['n_paragraphs']} paragraphs, {view.pooled['n_questions']} questions"
    )
    print(
        f"  control: {view.control.n_cells} unjudged cells (seed {config.seed}, "
        f"rule {analysis.sampling_rule}); "
        f"quotas: "
        + ", ".join(
            f"{document} {quota}" for document, quota in sorted(view.control.quotas.items())
        )
    )
    for model in models:
        if model.get("kind") == "decision":
            # The decision arm (CS-7b): its own entry, printed on its own lines.
            decision_block = model["arms"][DECISION_ARM]
            decision_measurement = decision_block.get("measurement", decision_block)
            decomposition = decision_measurement.get("pair_auc_decomposition") or {}
            rule = (model.get("rule") or {}).get("pooled") or {}
            fixed = rule.get(DECISION_RULE_FIXED_CUT) or {}
            counterpart = fixed.get("counterpart") or {}
            print(
                f"  [decision:{model['model']['name']}] {model['model']['model_id']} "
                f"(digest {model['model']['digest'][:12]}, {model['model']['state']}/"
                f"{model['model']['question_type']}): within-document AUC "
                f"{_fmt(decomposition.get('within_group_auc'), 4)}, pooled AUC "
                f"{_fmt(decision_measurement['pair_auc'].get('overall'), 4)}, macro "
                f"{_fmt(decision_measurement['pair_auc'].get('macro'), 4)}"
            )
            print(
                f"      the rule's fixed cut: the reach "
                f"{_fmt(fixed.get('coverage'), 4)} over the unjudged grid, the harm "
                f"{_fmt(fixed.get('harm'), 4)} on the verified positives; the "
                "label-blind counterpart's harm "
                f"{_fmt((counterpart.get('mean') or {}).get('harm'), 4)}"
            )
            continue
        info = model["embedder"]
        distributions = model["score_distributions"]
        enrichment = model["negative_enrichment"]["rank_percentile"]
        top1 = model["top1_candidate_provenance"]
        print(
            f"  [{model['name']}] {info['backend']}:{info['model_id']} "
            f"(dim {info['dimension']}): pair-AUC "
            f"{model['pair_auc']['overall']:.4f}; control mean "
            f"{distributions['control']['mean']:.3f}, p50 "
            f"{distributions['control']['quantiles']['0.5']:.3f}"
        )
        print(
            f"      rank percentile (0 = top of the candidate list): positives "
            f"{enrichment['label1']['quantiles']['0.5']:.3f}, negatives "
            f"{enrichment['label0']['quantiles']['0.5']:.3f}, control "
            f"{enrichment['control']['quantiles']['0.5']:.3f}"
        )
        print(
            f"      top-1 candidate per paragraph (model-dependent): positive "
            f"{top1['given_positive']}, negative {top1['given_negative']}, unjudged "
            f"{top1['unjudged']} of {top1['n_paragraphs']}"
        )
        for arm_name, comparison in model.get("comparisons", {}).items():
            paired = comparison["paired_auc"]
            shift = comparison["control_shift"]
            gap = comparison["floor_gap"]
            print(
                f"      [{arm_name}] dAUC {paired['delta_auc']:+.4f} "
                f"(95% CI [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}], "
                f"P(d>0) {paired['p_delta_positive']:.3f}); control mean "
                f"{shift['mean_delta']:+.4f}; floor gap {gap['delta']:+.4f}; "
                f"de-shifted AUC {comparison['deshifted_auc']['overall']:.4f}"
            )
        stage3 = model.get("stage3")
        if stage3 is not None:
            comparison = stage3["comparison"]
            decomposition = comparison["auc_decomposition"]
            within_paired = decomposition.get("within_group_paired") or {}
            floor = stage3["permutation_floor"]
            print(
                f"      [stage3] out-of-fold: within-document AUC "
                f"{_fmt(decomposition['arm'].get('within_group_auc'), 4)} "
                f"(baseline {_fmt(decomposition['reference'].get('within_group_auc'), 4)}, "
                f"delta {_fmt(decomposition.get('within_group_delta'), 4)}, "
                f"95% CI [{_fmt(within_paired.get('ci_low'), 4)}, "
                f"{_fmt(within_paired.get('ci_high'), 4)}]); pooled AUC "
                f"{_fmt(comparison['pair_auc']['arm'], 4)} "
                f"(delta {_fmt(comparison['paired_auc']['delta_auc'], 4)}); "
                f"chance floor {_fmt(floor['within_document']['chance'], 4)}, "
                f"control-role floor {_fmt(floor['within_document']['control_role'], 4)}"
            )
            groups = stage3["feature_pack"].get("groups") or []
            if groups:
                print(
                    f"        feature groups: {'+'.join(groups)} "
                    f"({stage3['feature_pack'].get('n_features')} features)"
                )
            diagnostics = (stage3.get("feature_diagnostics") or {}).get("per_feature") or {}
            selected = set(stage3["feature_pack"].get("feature_names") or [])
            if diagnostics and selected:
                top = sorted(
                    ((name, entry) for name, entry in diagnostics.items() if name in selected),
                    key=lambda item: (
                        item[1].get("within_auc") if item[1].get("within_auc") is not None else -1.0
                    ),
                    reverse=True,
                )[:3]
                print(
                    "        best features of the arm by their own within-document AUC: "
                    + ", ".join(
                        f"{name} ({_fmt(entry.get('within_auc'), 4)}, "
                        f"control mean {_fmt(entry.get('control', {}).get('mean'), 4)})"
                        for name, entry in top
                    )
                )
            length_check = stage3.get("length_check") or {}
            if length_check:
                arm_pearson = (length_check.get("arm") or {}).get("overall", {}).get("pearson")
                reference_pearson = (
                    (length_check.get("reference") or {}).get("overall", {}).get("pearson")
                )
                shift_pearson = (
                    (length_check.get("rank_shift") or {}).get("overall", {}).get("pearson")
                )
                print(
                    "        length vs control score (Pearson): arm "
                    f"{_fmt(arm_pearson, 4)}, "
                    f"cosine {_fmt(reference_pearson, 4)}, "
                    f"rank shift {_fmt(shift_pearson, 4)}"
                )
            reporting = stage3.get("reporting") or {}
            if reporting:
                calibration = reporting.get("calibration") or {}
                arm_calibration = calibration.get("arm") or {}
                cosine_calibration = calibration.get("cosine") or {}
                print(
                    "        calibration (the reliability diagram's expected "
                    "calibration error): arm "
                    f"{_fmt((arm_calibration.get('pooled') or {}).get('ece'), 4)} "
                    f"(per-document macro {_fmt(arm_calibration.get('macro_ece'), 4)}), "
                    f"cosine {_fmt((cosine_calibration.get('pooled') or {}).get('ece'), 4)} "
                    f"(per-document macro {_fmt(cosine_calibration.get('macro_ece'), 4)}); "
                    "the Brier scores "
                    f"{_fmt((arm_calibration.get('pooled') or {}).get('brier'), 4)} / "
                    f"{_fmt((cosine_calibration.get('pooled') or {}).get('brier'), 4)}"
                )
                thresholds = reporting.get("thresholds") or {}
                for document, block in (thresholds.get("per_document") or {}).items():
                    operating = (block.get("holdout") or {}).get("operating_point") or {}
                    print(
                        f"        threshold ({thresholds.get('criterion')}) for "
                        f"{document[:40]}: {_fmt(block.get('fit_side', {}).get('threshold'), 4)} "
                        f"(the holdout's recall {_fmt(operating.get('recall'), 4)}, "
                        f"precision {_fmt(operating.get('precision'), 4)})"
                    )
                out_of_fold = thresholds.get("out_of_fold") or {}
                median_operating = out_of_fold.get("operating_point_at_median") or {}
                print(
                    "        the pooled out-of-fold cut (mixes three calibrations, "
                    f"not like-for-like): median threshold "
                    f"{_fmt(out_of_fold.get('median_threshold'), 4)}, recall "
                    f"{_fmt(median_operating.get('recall'), 4)}"
                )
                safe_negative = reporting.get("safe_negative") or {}
                for variant, block in (safe_negative.get("variants") or {}).items():
                    pooled = block.get("pooled") or {}
                    floor = block.get("floor_arm") or {}
                    print(
                        f"        safe-negative rule [{variant}]: the unjudged grid's "
                        f"coverage {_fmt(pooled.get('coverage'), 4)} "
                        f"({pooled.get('n_unjudged_cleared')} of "
                        f"{pooled.get('n_unjudged_cells')}), "
                        f"the verified positives' harm {_fmt(pooled.get('harm'), 4)}; "
                        "the control-role floor's harm at the same coverage "
                        f"{_fmt((floor.get('harm_at_matched_coverage') or {}).get('harm'), 4)}"
                    )
            rotations = stage3.get("rotations") or []
            usage: dict[str, Any] = {}
            if rotations:
                usage = (rotations[0].get("feature_usage") or {}).get("per_feature") or {}
            if usage:
                top = sorted(
                    usage.items(),
                    key=lambda item: item[1].get("total_gain", 0.0),
                    reverse=True,
                )[:3]
                print(
                    "        top features by the trees' gain (first rotation): "
                    + ", ".join(
                        f"{name} ({int(entry.get('n_splits', 0))} splits, "
                        f"gain {entry.get('total_gain', 0.0):.3f})"
                        for name, entry in top
                    )
                )
            for rotation in stage3["rotations_summary"]:
                print(
                    f"        holdout {rotation['holdout'][:40]}: arm "
                    f"{_fmt(rotation['arm_pair_auc'], 4)} (baseline "
                    f"{_fmt(rotation['reference_pair_auc'], 4)}, chance floor "
                    f"{_fmt(rotation['chance_floor_mean'], 4)}, control-role floor "
                    f"{_fmt(rotation['control_role_floor_mean'], 4)})"
                )
    print(f"Report written to {results_dir / 'analysis.json'}")
    return 0


def build_decider(model_config: Any, config: AppConfig) -> Any:
    """Build the decision model client for one arm (the CLI's wiring seam).

    The client is the raw ``POST /v1/systemone`` caller (the endpoint is in
    neither the Ollama CLI nor its Python/JS libraries); it checks the pinned
    digest against the server's and **fails the run** on a mismatch. The
    decision models' own disk cache wraps it (its own root under ``cache/``,
    never the embedding cache), so a re-run is reproducible and never
    re-judges a pair.
    """
    # The probe makes thousands of calls; httpx logs one INFO line per request,
    # which would bury the run's own output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    client = SystemOneDecider(
        model_config,
        host=config.analysis.decision.host if config.analysis and config.analysis.decision else "",
        timeout=(
            config.analysis.decision.timeout
            if config.analysis and config.analysis.decision
            else 600.0
        ),
    )
    prompt_version = (
        config.analysis.decision.prompt_version
        if config.analysis and config.analysis.decision
        else "v1"
    )
    return CachedDecider(client, config.paths.cache, prompt_version=prompt_version)


def _decide_omct_row(
    model_block: dict[str, Any],
    *,
    run_id: str,
    n_pairs: int,
    n_control: int,
    seconds: float,
) -> dict[str, Any]:
    """One row of the decision probe's comparison table."""
    measurement = model_block["measurement"]
    pair_auc = measurement["pair_auc"]
    decomposition = measurement["pair_auc_decomposition"]
    calibration = measurement["calibration"]
    distributions = measurement["score_distributions"]
    competence = model_block.get("competence_control") or {}
    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "model": model_block["model"]["model_id"],
        "model_version": model_block["model"]["digest"][:12],
        "state": model_block["state"],
        "question_type": model_block["question_type"],
        "scope": model_block["scope"],
        "within_auc": _fmt(decomposition.get("within_group_auc"), 4),
        "pooled_auc": _fmt(pair_auc.get("overall"), 4),
        "macro_auc": _fmt(pair_auc.get("macro"), 4),
        "cross_auc": _fmt(decomposition.get("cross_group_auc"), 4),
        "ece": _fmt((calibration.get("pooled") or {}).get("ece"), 4),
        "ece_macro": _fmt(calibration.get("macro_ece"), 4),
        "brier": _fmt((calibration.get("pooled") or {}).get("brier"), 4),
        "pos_mean": _fmt((distributions.get("label1") or {}).get("mean"), 4),
        "neg_mean": _fmt((distributions.get("label0") or {}).get("mean"), 4),
        "ctrl_mean": _fmt((distributions.get("control") or {}).get("mean"), 4),
        "floor_gap": _fmt((measurement.get("floor_gap") or {}).get("gap"), 4),
        "comp_auc": _fmt(competence.get("pair_auc"), 4),
        "comp_pos_mean": _fmt((competence.get("positive") or {}).get("mean"), 4),
        "comp_neg_mean": _fmt((competence.get("negative") or {}).get("mean"), 4),
        "n_pairs": n_pairs,
        "n_control": n_control,
        "n_comp": competence.get("n_cells"),
        "n_calls": model_block.get("n_calls_planned"),
        "trunc": _fmt((model_block.get("truncation") or {}).get("share"), 4),
        "seconds": _fmt(seconds, 3),
    }


def _cmd_decide_omct(config: AppConfig) -> int:
    """Run decide-omct: the track's Stage-3.5 decision-model probe.

    Writes ``results/omct-<run_id>/{config.yaml, decisions.json}``: the
    per-cell probabilities, the measurement blocks (the pair AUC pooled and per
    document, the within-document AUC, the rank percentiles, the control side,
    the calibration curve), the synthetic competence control and the
    provenance (the model id, the pinned digest, the prompt version, the
    cache's hit/miss counters) — per arm.
    """
    started = time.perf_counter()
    run_id = _require_run_id(config)
    if config.analysis is None or config.analysis.decision is None:
        raise ValueError(
            "the analysis.decision: section (the decision models and the prompt "
            "grid) is required for the `decide-omct` command"
        )
    if config.paths.data_external is None:
        raise ValueError("paths.data_external must be set in the config to run decide-omct")

    decision = config.analysis.decision
    pair_report = load_pairs(config.paths.data_external)
    view = build_dataset_view(
        pair_report.records,
        control_size=config.analysis.control_size,
        seed=config.seed,
        sampling_rule=config.analysis.sampling_rule,
    )

    models: list[dict[str, Any]] = []
    for model_config in decision.models:
        decider = build_decider(model_config, config)
        arm = DecisionArm(
            name=model_config.name,
            state=model_config.state,
            question_type=model_config.question_type,
            instructions=model_config.instructions,
            options=model_config.options,
            criteria=model_config.criteria,
            positive_option=model_config.positive_option,
            max_input_tokens=model_config.max_input_tokens,
        )
        block = run_decision_arm(
            view,
            decider,
            arm,
            scope=decision.scope,
            question_transform=config.analysis.question_transform,
            question_block_markers=config.analysis.question_block_markers,
            competence_control=decision.competence_control,
            competence_pairs=decision.competence_pairs,
            max_questions_per_call=decision.max_questions_per_call,
            bins=decision.bins,
        )
        info = decider.info
        block["model"] = {
            "name": model_config.name,
            "model_id": info.model_id,
            "digest": info.digest,
            "backend": info.backend,
            "host": info.host,
            "prompt_version": decision.prompt_version,
            "extra": info.extra,
        }
        block["cache"] = {
            "hits": getattr(decider, "cache_hits", 0),
            "misses": getattr(decider, "cache_misses", 0),
            "stored": getattr(decider, "n_stored", 0),
        }
        block["usage"] = dict(getattr(decider, "usage", {}))
        models.append(block)

    omct_run_id = _omct_run_id(run_id)
    payload = {
        "run_id": omct_run_id,
        "command": "decide-omct",
        "source": str(config.paths.data_external),
        "source_sha256": file_sha256(config.paths.data_external),
        "n_valid_pairs": pair_report.n_valid,
        "n_malformed_pairs": pair_report.n_invalid,
        "seed": config.seed,
        "analysis": {
            "scope": decision.scope,
            "prompt_version": decision.prompt_version,
            "question_transform": config.analysis.question_transform,
            "question_block_markers": list(config.analysis.question_block_markers),
            "control": view.control.describe(),
            "grid": {
                "n_documents": view.pooled["n_documents"],
                "n_paragraphs": view.pooled["n_paragraphs"],
                "n_questions": view.pooled["n_questions"],
                "n_cells": view.pooled["grid_cells"],
                "n_judged_cells": view.pooled["judged_cells"],
                "n_unjudged_cells": view.pooled["unjudged_cells"],
                "coverage": view.pooled["grid_coverage"],
            },
            "strata": list(view.strata),
            "pooled": view.pooled,
            "notes": [
                "OMCT is auxiliary data (PLAN.md section 2): these numbers annotate "
                "method choice, they are never a UPR metric.",
                "The two negative subsets are never pooled: the reviewer-verified "
                "hard negatives (the reviewed set's label 0) and the unjudged-proxy "
                "control cells are separate groups.",
                "The decision models are frozen and zero-shot: they carry no OMCT "
                "label, so nothing leaks. Fine-tuning one on OMCT would break the "
                "track's leakage rule (i) (101 question numbers are positive in one "
                "document and negative in another).",
                "A decision model only scores: it never relabels, curates or "
                "corrects the data (the raw pair file is read-only).",
                "The rank percentiles and the hit@k are measured inside the cells "
                "the run scored for a paragraph (the restricted candidate list), "
                "not inside the paragraph's full same-document question list: the "
                "full list needs the grid scope.",
            ],
        },
        "models": models,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    results_dir = config.paths.results / omct_run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(results_dir, config)
    (results_dir / "decisions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    seconds = payload["total_seconds"]
    rows = [
        _decide_omct_row(
            block,
            run_id=omct_run_id,
            n_pairs=view.pooled["n_pairs"],
            n_control=view.control.n_cells,
            seconds=seconds,
        )
        for block in models
    ]
    _append_command_rows(
        config,
        "decide-omct",
        columns=DECIDE_OMCT_COLUMNS,
        key_columns=DECIDE_OMCT_KEY_COLUMNS,
        rows=rows,
    )

    print(f"OMCT decision probe (Stage 3.5): {omct_run_id}")
    print(
        f"  scope: {decision.scope}; {view.pooled['n_pairs']} reviewed pairs, "
        f"{view.control.n_cells} control cells, "
        f"{view.pooled['n_documents']} documents"
    )
    for block in models:
        measurement = block["measurement"]
        decomposition = measurement["pair_auc_decomposition"]
        calibration = measurement["calibration"]
        competence = block.get("competence_control") or {}
        print(
            f"  [{block['arm']}] {block['model']['model_id']} "
            f"(digest {block['model']['digest'][:12]}, {block['state']}/"
            f"{block['question_type']}): within-document AUC "
            f"{_fmt(decomposition.get('within_group_auc'), 4)}, pooled AUC "
            f"{_fmt(measurement['pair_auc'].get('overall'), 4)}, macro "
            f"{_fmt(measurement['pair_auc'].get('macro'), 4)}"
        )
        print(
            f"      calibration: ECE {_fmt((calibration.get('pooled') or {}).get('ece'), 4)} "
            f"(per-document macro {_fmt(calibration.get('macro_ece'), 4)}), the Brier "
            f"{_fmt((calibration.get('pooled') or {}).get('brier'), 4)}; the floor gap "
            f"{_fmt((measurement.get('floor_gap') or {}).get('gap'), 4)}"
        )
        if competence:
            print(
                f"      competence control: AUC {_fmt(competence.get('pair_auc'), 4)} "
                f"(the positives' mean {_fmt((competence.get('positive') or {}).get('mean'), 4)}, "
                f"the negatives' {_fmt((competence.get('negative') or {}).get('mean'), 4)})"
            )
        print(
            f"      calls: {(block.get('usage') or {}).get('n_calls', 0)}; the cache's "
            f"hits {(block.get('cache') or {}).get('hits', 0)}, misses "
            f"{(block.get('cache') or {}).get('misses', 0)}"
        )
    print(f"Report written to {results_dir / 'decisions.json'}")
    return 0
