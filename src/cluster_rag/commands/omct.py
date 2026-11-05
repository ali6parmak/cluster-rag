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
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np

from cluster_rag.commands._common import (
    _append_command_rows,
    _cached_embedder,
    _fmt,
    _omct_run_id,
    _require_run_id,
    _write_resolved_config,
    pos_word_classes,
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
    ANALYZE_OMCT_DICTIONARY_ABSTENTION_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_ABSTENTION_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_ARM_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_ARM_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_CLASSES_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_CLASSES_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_ESTIMATOR_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_ESTIMATOR_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_LEXICON_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_LEXICON_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_PHRASES_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_PHRASES_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_POS_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_POS_KEY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_STABILITY_COLUMNS,
    ANALYZE_OMCT_DICTIONARY_STABILITY_KEY_COLUMNS,
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
    DICTIONARY_ARM,
    BaselineArm,
    ChunkedSide,
    LateSide,
    ReportingSpec,
    build_analysis_report,
    build_dataset_view,
    build_dictionary_model,
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
from cluster_rag.text_processors.features.phrase_map import phrase_units
from cluster_rag.text_processors.features.question_map import (
    UNIT_PHRASES,
    UNIT_WORDS,
    build_question_map,
    cross_fit_availability,
    question_map_block,
)
from cluster_rag.text_processors.lexical.question_lexicon import (
    LEXICON_ARMS,
    build_question_lexicon,
    lexicon_grid,
    lexicon_pair_scores,
    lexicon_readout,
)
from cluster_rag.text_processors.lexical.question_map_score import (
    PRIMARY_SCORING,
    blind_question_permutation,
    cross_fit_grids,
    union_grids,
)
from cluster_rag.text_processors.lexical.question_map_stability import (
    estimate_question_map_stability,
    kept_share,
    stability_filter,
)
from cluster_rag.text_processors.lexical.question_map_stability import (
    stability_block as question_map_stability_block,
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


def _dictionary_seed(analysis: AnalysisConfig, run_seed: int) -> int:
    """The dictionary cross-fit's seed (the config's own, else the run's)."""
    if analysis.dictionary is not None and analysis.dictionary.seed is not None:
        return analysis.dictionary.seed
    return run_seed


def _dictionary_fold_ids(n_pairs: int, folds: int, seed: int) -> np.ndarray:
    """The cross-fit's fold assignment (the track's scheme: a seeded permutation).

    The same scheme as the UPR cross-fit: ``default_rng(seed).permutation(n)``
    split into contiguous chunks. The assignment is computed once per run and
    shared by the map's availability readout (CS-48) and the scorer's grids
    (CS-49), so the two read the same folds.
    """
    order = np.random.default_rng(seed).permutation(n_pairs)
    chunks = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(order, folds)]
    fold_ids = np.empty(n_pairs, dtype=np.int64)
    for fold, chunk in enumerate(chunks):
        fold_ids[chunk] = fold
    return fold_ids


def _dictionary_block(
    records: Sequence[Any],
    folds: int,
    seed: int,
    view: Any,
    question_map: Any,
    fold_ids: np.ndarray,
    *,
    units: Any = None,
    unit_name: str = UNIT_WORDS,
) -> dict[str, Any]:
    """Build the artifact's `dictionary` block (CS-48): the map + its availability.

    The map is built from the **reviewed pairs only** (`records` is the loaded
    pair file, i.e. subset A); the control cells (subset B) live in the view
    and are never handed to the map. The cross-fit is the track's scheme: a
    seeded permutation of the pairs split into contiguous chunks, each fold's
    map from the other folds.

    Args:
        records: The reviewed pairs (the map's corpus).
        folds: The cross-fit's fold count.
        seed: The fold split's seed.
        view: The dataset view (the paragraph count and the document count).
        question_map: The map built from the same pairs (shared with the arm).
        fold_ids: The cross-fit's fold assignment (shared with the arm).
        units: The map's unit (CS-53): the callable the availability readout
            must read the paragraphs with, so it reads the map's own tables.
        unit_name: The unit's name (``words`` or ``phrases``), recorded in the
            block.

    Returns:
        The block :func:`question_map_block` assembles.
    """
    paragraphs = [record.paragraph_text for record in records]
    questions = [record.question_text for record in records]
    labels = [record.label for record in records]
    availability = cross_fit_availability(
        paragraphs,
        questions,
        labels,
        fold_ids,
        documents=[record.document for record in records],
        pair_ids=[record.pair_id for record in records],
        units=units,
    )
    return question_map_block(
        question_map,
        availability,
        folds=folds,
        seed=seed,
        n_paragraphs=len(view.paragraphs),
        n_documents=view.pooled["n_documents"],
        unit_name=unit_name,
    )


def _dictionary_row(
    block: dict[str, Any],
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
    documents_by_code: Mapping[str, str],
) -> dict[str, Any]:
    """One table row of the dictionary section (CS-48's map readout).

    The availability columns carry the cross-fit's usability numbers, pooled
    and per document; the document codes (NU / CIDH / AI) are resolved to the
    artifact's full document names through the view's own code mapping.
    """
    corpus = block["corpus"]
    shape = block["map"]
    readout = block["readout"]
    availability = block.get("availability") or {}
    by_document = availability.get("by_document") or {}
    overall = availability.get("overall") or {}

    def _document_share(code: str, key: str) -> Any:
        document = documents_by_code.get(code, "")
        return (by_document.get(document) or {}).get(key)

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "model": "n/a",  # the map is model-independent
        "arm": "dictionary_map",
        "unit": "pair/paragraph/question",
        "n_questions": corpus["n_questions"],
        "tables": shape["n_questions_with_table"],
        "empty_tables": shape["n_questions_empty"],
        "contrastable": shape["n_contrastable_questions"],
        "n_words": shape["n_words"],
        "stored_pairs": shape["n_stored_pairs"],
        "table_median": _fmt(shape["table_size"]["median"], 1),
        "shared_word_tables": readout["thickness"]["n_tables_with_shared_word"],
        "head_diff_contrastable": readout["head_agreement"]["n_differ_contrastable"],
        "avail_table": _fmt(overall.get("with_table_share"), 4),
        "avail_evidence": _fmt(overall.get("with_evidence_share"), 4),
        "avail_table_nu": _fmt(_document_share("NU", "with_table_share"), 4),
        "avail_table_cidh": _fmt(_document_share("CIDH", "with_table_share"), 4),
        "avail_table_ai": _fmt(_document_share("AI", "with_table_share"), 4),
        "n_pairs": corpus["n_reviewed_pairs"],
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_class_row(
    model: dict[str, Any],
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **classes** section (CS-51)."""
    readout = model["readout"]
    classes = readout.get("class_arms") or {}
    primary = readout["primary"]
    full = readout["scorings"][primary]
    class_readout = (model.get("class_readout") or {}).get("classes") or {}

    def _arm(name: str, key: str) -> Any:
        return (classes.get(name) or {}).get(key)

    def _ci(name: str) -> str | None:
        block = classes.get(name) or {}
        if block.get("ci_low") is None:
            return None
        return f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]"

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "arm": primary,
        "full_within_auc": _fmt(full.get("within_auc"), 4),
        "full_nonfloor_auc": _fmt((readout.get("secondary_non_floor") or {}).get("within_auc"), 4),
        "nogeneric_within_auc": _fmt(_arm("nogeneric", "within_auc"), 4),
        "nogeneric_delta": _fmt(_arm("nogeneric", "delta_vs_full"), 4),
        "nogeneric_ci": _ci("nogeneric"),
        "nogeneric_nonfloor_auc": _fmt((_arm("nogeneric", "non_floor") or {}).get("within_auc"), 4),
        "distinctive_within_auc": _fmt(_arm("distinctive", "within_auc"), 4),
        "distinctive_delta": _fmt(_arm("distinctive", "delta_vs_full"), 4),
        "distinctive_ci": _ci("distinctive"),
        "distinctive_nonfloor_auc": _fmt(
            (_arm("distinctive", "non_floor") or {}).get("within_auc"), 4
        ),
        "class_distinctive_mean_z": _fmt(
            (class_readout.get("distinctive") or {}).get("mean_z_score"), 4
        ),
        "class_generic_mean_z": _fmt((class_readout.get("generic") or {}).get("mean_z_score"), 4),
        "class_generic_positive_share": _fmt(
            (class_readout.get("generic") or {}).get("positive_share"), 4
        ),
        "floor_full": readout["floor"]["n_floor_pairs"],
        "floor_nogeneric": _arm("nogeneric", "floor_pairs"),
        "floor_distinctive": _arm("distinctive", "floor_pairs"),
        "n_pairs": full.get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_pos_row(
    model: dict[str, Any],
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **POS** section (CS-52)."""
    readout = model["readout"]
    arms = readout.get("pos_arms") or {}
    primary = readout["primary"]
    full = readout["scorings"][primary]
    pos = model.get("pos_readout") or {}
    classes = pos.get("classes") or {}

    def _arm(name: str, key: str) -> Any:
        return (arms.get(name) or {}).get(key)

    def _ci(name: str) -> str | None:
        block = arms.get(name) or {}
        if block.get("ci_low") is None:
            return None
        return f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]"

    noun = classes.get("noun") or {}
    verb = classes.get("verb") or {}
    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "arm": primary,
        "full_within_auc": _fmt(full.get("within_auc"), 4),
        "full_nonfloor_auc": _fmt((readout.get("secondary_non_floor") or {}).get("within_auc"), 4),
        "noun_within_auc": _fmt(_arm("noun", "within_auc"), 4),
        "noun_delta": _fmt(_arm("noun", "delta_vs_full"), 4),
        "noun_ci": _ci("noun"),
        "noun_nonfloor_auc": _fmt((_arm("noun", "non_floor") or {}).get("within_auc"), 4),
        "verb_within_auc": _fmt(_arm("verb", "within_auc"), 4),
        "verb_delta": _fmt(_arm("verb", "delta_vs_full"), 4),
        "verb_ci": _ci("verb"),
        "verb_nonfloor_auc": _fmt((_arm("verb", "non_floor") or {}).get("within_auc"), 4),
        "noun_mean_z": _fmt(noun.get("mean_z_score"), 4),
        "verb_mean_z": _fmt(verb.get("mean_z_score"), 4),
        "noun_mean_logodds": _fmt(noun.get("mean_log_odds"), 4),
        "verb_mean_logodds": _fmt(verb.get("mean_log_odds"), 4),
        "floor_full": readout["floor"]["n_floor_pairs"],
        "floor_noun": _arm("noun", "floor_pairs"),
        "floor_verb": _arm("verb", "floor_pairs"),
        "n_pairs": full.get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_phrases_row(
    model: dict[str, Any],
    block: Mapping[str, Any] | None,
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **phrases** section (CS-53's unit test).

    The headline is the **union** arm against the **word** z arm (the
    registered increment test, paired), with the union's own cosine comparison
    and the two halves' readings beside it; the count arms, the floor and the
    phrase table's shape (the CS-39 readouts) close the row.
    """
    readout = model["readout"]
    union = readout["union"]
    primary = readout["primary"]
    vs_word = union["vs_word"]
    vs_cosine = union["vs_cosine"]
    profile = (block or {}).get("unit_profile") or {}
    frequencies = profile.get("df") or {}

    def _ci(low: Any, high: Any) -> str | None:
        if low is None or high is None:
            return None
        return f"[{low:+.4f}, {high:+.4f}]"

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "reference_model": model["model"].get("reference_model"),
        "arm": union.get("arm"),
        "word_within_auc": _fmt(union["word"].get("within_auc"), 4),
        "word_nonfloor_auc": _fmt(union["word"].get("non_floor_auc"), 4),
        "word_count_auc": _fmt(union["word"].get("count_within_auc"), 4),
        "phrase_within_auc": _fmt(union["phrase"].get("within_auc"), 4),
        "phrase_nonfloor_auc": _fmt(union["phrase"].get("non_floor_auc"), 4),
        "phrase_count_auc": _fmt(union["phrase"].get("count_within_auc"), 4),
        "union_within_auc": _fmt(union.get("within_auc"), 4),
        "union_delta_vs_word": _fmt(vs_word.get("delta"), 4),
        "union_ci": _ci(vs_word.get("ci_low"), vs_word.get("ci_high")),
        "union_p_delta": _fmt(vs_word.get("p_delta_positive"), 3),
        "union_nonfloor_auc": _fmt((union.get("secondary_non_floor") or {}).get("within_auc"), 4),
        "union_delta_vs_cosine": _fmt(vs_cosine.get("delta"), 4),
        "union_ci_vs_cosine": _ci(vs_cosine.get("ci_low"), vs_cosine.get("ci_high")),
        "union_pooled_auc": _fmt(union.get("pooled_auc"), 4),
        "union_pooled_d_auc": _fmt(vs_cosine.get("pooled_delta"), 4),
        "union_deshifted_auc": _fmt(vs_cosine.get("deshifted_auc"), 4),
        "phrase_units": profile.get("n_units"),
        "phrase_rows": profile.get("n_stored_pairs"),
        "phrase_generic_row_share": _fmt(profile.get("generic_row_share"), 4),
        "phrase_hapax_share": _fmt(frequencies.get("hapax_share"), 4),
        "floor_pairs": union["floor"]["n_floor_pairs"],
        "floor_share": _fmt(union["floor"]["floor_share"], 4),
        "n_pairs": (readout["scorings"].get(primary) or {}).get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_stability_row(
    model: dict[str, Any],
    block: Mapping[str, Any] | None,
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **stability** section (CS-54)."""
    readout = model["readout"]
    arms = readout.get("stability_arms") or {}
    thresholds = readout.get("stability_thresholds") or []
    full = readout["scorings"][readout["primary"]]
    readout_block = model.get("stability_readout") or {}
    kept = readout_block.get("kept_share") or {}
    single = readout_block.get("single_pair") or {}
    heads = readout_block.get("head_retention") or {}
    per_question = readout_block.get("per_question") or []

    def _shares() -> list[float]:
        return [
            row["share_surviving_075"]
            for row in per_question
            if row.get("share_surviving_075") is not None and (row.get("table_words") or 0) > 0
        ]

    def _arm(threshold: str, key: str) -> Any:
        return (arms.get(threshold) or {}).get(key)

    def _ci(threshold: str) -> str | None:
        block_arm = arms.get(threshold) or {}
        if block_arm.get("ci_low") is None:
            return None
        return f"[{block_arm['ci_low']:+.4f}, {block_arm['ci_high']:+.4f}]"

    def _threshold(index: int) -> str:
        return thresholds[index] if len(thresholds) > index else ""

    medium = _shares()
    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "arm": readout["primary"],
        "full_within_auc": _fmt(full.get("within_auc"), 4),
        "full_nonfloor_auc": _fmt((readout.get("secondary_non_floor") or {}).get("within_auc"), 4),
        "stable75_within_auc": _fmt(_arm(_threshold(0), "within_auc"), 4),
        "stable75_delta": _fmt(_arm(_threshold(0), "delta_vs_full"), 4),
        "stable75_ci": _ci(_threshold(0)),
        "stable75_floor_pairs": _arm(_threshold(0), "floor_pairs"),
        "stable75_floor_share": _fmt(_arm(_threshold(0), "floor_share"), 4),
        "stable75_nonfloor_auc": _fmt(
            (_arm(_threshold(0), "non_floor") or {}).get("within_auc"), 4
        ),
        "stable90_within_auc": _fmt(_arm(_threshold(1), "within_auc"), 4),
        "stable90_delta": _fmt(_arm(_threshold(1), "delta_vs_full"), 4),
        "stable90_ci": _ci(_threshold(1)),
        "stable90_floor_pairs": _arm(_threshold(1), "floor_pairs"),
        "stable90_floor_share": _fmt(_arm(_threshold(1), "floor_share"), 4),
        "stable90_nonfloor_auc": _fmt(
            (_arm(_threshold(1), "non_floor") or {}).get("within_auc"), 4
        ),
        "survival75_share": _fmt(kept.get("0.75"), 4),
        "survival90_share": _fmt(kept.get("0.90"), 4),
        "single_pair_survival": _fmt(single.get("mean_sign_stability"), 4),
        "median_question_stability75": _fmt(float(np.median(medium)) if medium else None, 4),
        "head_retention": _fmt(heads.get("share"), 4),
        "n_links": readout_block.get("n_links"),
        "n_pairs": full.get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_lexicon_row(
    model: dict[str, Any],
    block: Mapping[str, Any] | None,
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **lexicon** section (CS-55)."""
    readout = (model.get("readout") or {}).get("question_lexicon") or {}
    arms = readout.get("arms") or {}
    primary = readout.get("primary") or "all"
    vs_map = readout.get("vs_map") or {}
    register_test = readout.get("register_test") or {}
    blind = readout.get("blind") or {}
    profile = (block or {}).get("question_lexicon") or {}
    classes = profile.get("classes") or {}
    sizes = profile.get("vocabulary_size") or {}

    def _arm(name: str, key: str) -> Any:
        return (arms.get(name) or {}).get(key)

    def _ci(low: Any, high: Any) -> str | None:
        if low is None or high is None:
            return None
        return f"[{low:+.4f}, {high:+.4f}]"

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "arm": primary,
        "all_within_auc": _fmt(_arm(primary, "within_auc"), 4),
        "all_delta_cosine": _fmt(_arm(primary, "delta_vs_cosine"), 4),
        "all_ci": _ci(_arm(primary, "ci_low"), _arm(primary, "ci_high")),
        "idf_within_auc": _fmt(_arm("idf", "within_auc"), 4),
        "topic_within_auc": _fmt(_arm("topic", "within_auc"), 4),
        "topic_delta_vs_all": _fmt(register_test.get("delta"), 4),
        "topic_ci": _ci(register_test.get("ci_low"), register_test.get("ci_high")),
        "register_within_auc": _fmt(_arm("register", "within_auc"), 4),
        "map_within_auc": _fmt(vs_map.get("map_within_auc"), 4),
        "all_delta_vs_map": _fmt(vs_map.get("delta"), 4),
        "all_ci_vs_map": _ci(vs_map.get("ci_low"), vs_map.get("ci_high")),
        "blind_within_auc": _fmt(blind.get("within_auc"), 4),
        "blind_gap": _fmt(blind.get("gap"), 4),
        "label_words": profile.get("n_words"),
        "register_words": (classes.get("generic") or {}).get("words"),
        "topic_words": (classes.get("distinctive") or {}).get("words"),
        "vocabulary_median": _fmt(sizes.get("median"), 1),
        "zero_overlap_share": _fmt((profile.get("paragraph_overlap") or {}).get("zero_share"), 4),
        "n_pairs": (_arm(primary, "pooled_auc") is not None)
        and model["readout"]["scorings"][model["readout"]["primary"]]["n_pairs"],
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_abstention_row(
    model: dict[str, Any],
    block: Mapping[str, Any] | None,
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
    documents_by_code: Mapping[str, str],
) -> dict[str, Any]:
    """One table row of the dictionary **abstention** section (CS-56)."""
    readout = (model.get("readout") or {}).get("abstention_arms") or {}
    fills = readout.get("fills") or {}
    floor = readout.get("floor") or {}
    by_document = readout.get("floor_negative_by_document") or {}
    profile = (block or {}).get("abstention") or {}

    def _arm(name: str, key: str) -> Any:
        return (fills.get(name) or {}).get(key)

    def _ci(name: str) -> str | None:
        fill_block = fills.get(name) or {}
        if fill_block.get("ci_low") is None:
            return None
        return f"[{fill_block['ci_low']:+.4f}, {fill_block['ci_high']:+.4f}]"

    def _document_share(code: str) -> Any:
        document = documents_by_code.get(code, "")
        return (by_document.get(document) or {}).get("negative_share")

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "arm": readout.get("primary"),
        "floor_within_auc": _fmt(floor.get("within_auc"), 4),
        "floor_nonfloor_auc": _fmt((floor.get("non_floor") or {}).get("within_auc"), 4),
        "neutral_within_auc": _fmt(_arm("neutral", "within_auc"), 4),
        "neutral_delta": _fmt(_arm("neutral", "delta_vs_full"), 4),
        "neutral_ci": _ci("neutral"),
        "max_within_auc": _fmt(_arm("max", "within_auc"), 4),
        "max_delta": _fmt(_arm("max", "delta_vs_full"), 4),
        "max_ci": _ci("max"),
        "floor_pairs": readout.get("floor_pairs"),
        "floor_share": _fmt(readout.get("floor_share"), 4),
        "floor_negative_share": _fmt(readout.get("floor_negative_share"), 4),
        "silent_questions": profile.get("silent_questions"),
        "floor_negative_nu": _fmt(_document_share("NU"), 4),
        "floor_negative_cidh": _fmt(_document_share("CIDH"), 4),
        "floor_negative_ai": _fmt(_document_share("AI"), 4),
        "n_pairs": (floor.get("non_floor") or {}).get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


def _dictionary_arm_row(
    model: dict[str, Any],
    *,
    run_id: str,
    n_control: int,
    n_docs: int,
    seconds: float,
) -> dict[str, Any]:
    """One table row of the dictionary **arm** section (CS-49's scorer readout).

    The headline is the primary arm's within-document AUC and its paired
    difference against the run's plain full-text cosine; the four arms' own
    within-document AUCs are beside it, and the label-blind counterpart's gap
    and the floor's share close the row.
    """
    readout = model["readout"]
    comparison = model["comparisons"][DICTIONARY_ARM]
    arm = model["arms"][DICTIONARY_ARM]
    primary = readout["primary"]
    within = comparison["auc_decomposition"]["within_group_paired"]
    scorings = readout["scorings"]
    blind = readout.get("blind") or {}
    blind_gap = readout.get("blind_gap") or {}

    def _arm_auc(name: str) -> Any:
        block = scorings.get(name) or {}
        return _fmt(block.get("within_auc"), 4)

    return {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "reference_model": model["model"].get("reference_model"),
        "arm": primary,
        "within_auc": _fmt(within.get("auc_arm"), 4),
        "within_d_auc": _fmt(within.get("delta_auc"), 4),
        "within_d_auc_ci": (
            f"[{within['ci_low']:+.4f}, {within['ci_high']:+.4f}]"
            if within.get("ci_low") is not None
            else None
        ),
        "ref_within_auc": _fmt(within.get("auc_reference"), 4),
        "pooled_auc": _fmt(arm["pair_auc"]["overall"], 4),
        "pooled_d_auc": _fmt(comparison["paired_auc"]["delta_auc"], 4),
        "deshifted_auc": _fmt(comparison["deshifted_auc"]["overall"], 4),
        "auc_sum_logodds": _arm_auc("sum_logodds"),
        "auc_sum_positive": _arm_auc("sum_positive"),
        "auc_max_logodds": _arm_auc("max_logodds"),
        "auc_support_count": _arm_auc("support_count"),
        "auc_sum_zscore": _arm_auc("sum_zscore"),
        "auc_sum_zscore_positive": _arm_auc("sum_zscore_positive"),
        "auc_sum_positive_nonfloor": _fmt(
            (readout.get("secondary_non_floor") or {}).get("within_auc"), 4
        ),
        "blind_within_auc": _fmt((blind.get(primary) or {}).get("within_auc"), 4),
        "blind_gap": _fmt(blind_gap.get("primary"), 4),
        "floor_pairs": readout["floor"]["n_floor_pairs"],
        "floor_share": _fmt(readout["floor"]["floor_share"], 4),
        "ctrl_mean_delta": _fmt(comparison["control_shift"]["mean_delta"], 4),
        "ctrl_rank_p50_delta": _fmt(comparison["control_shift"]["rank_p50_delta"], 4),
        "floor_gap_delta": _fmt(comparison["floor_gap"]["delta"], 4),
        "hit1_pos": _fmt((arm.get("hit_at_k") or {}).get("label1", {}).get("hit@1"), 4),
        "hit1_pos_ref": _fmt(
            ((comparison.get("hit_at_k") or {}).get("reference") or {})
            .get("label1", {})
            .get("hit@1"),
            4,
        ),
        "n_pairs": (scorings.get(primary) or {}).get("n_pairs"),
        "n_control": n_control,
        "n_docs": n_docs,
        "seconds": seconds,
    }


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
    if not config.analysis.embedders and config.analysis.dictionary is None:
        raise ValueError(
            "analysis.embedders must name at least one model to score with (a run "
            "that carries only the dictionary block — CS-48's question map — needs "
            "no embedder and may leave the list empty)"
        )

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

    dictionary_block = None
    dictionary_map = None
    dictionary_fold_ids = None
    dictionary_seed = None
    if analysis.dictionary is not None:
        dictionary_seed = _dictionary_seed(analysis, config.seed)
        dictionary_fold_ids = _dictionary_fold_ids(
            len(pair_report.records), analysis.dictionary.folds, dictionary_seed
        )
        # CS-53: the map's unit. ``None`` is the track's content-word tokenizer
        # (the word table); the phrase table passes the bigram tokenizer.
        dictionary_units = phrase_units if analysis.dictionary.units == UNIT_PHRASES else None
        pos_classes: dict[str, str] = {}
        pos_provenance: dict[str, Any] = {}
        if analysis.dictionary.pos_restrictions:
            # CS-52: the grammatical classes of the reviewed paragraphs (the
            # tagger is the edge's nltk client; its answers are cached). The
            # POS arms are the tagger's on-switch for a dictionary run: the
            # ``pos:`` section's own ``enabled`` stays the UPR default.
            pos_config = config.pos.model_copy(update={"enabled": True})
            pos_classes, _pos_distribution, pos_provenance = pos_word_classes(
                config.model_copy(update={"pos": pos_config}),
                list(dict.fromkeys(r.paragraph_text for r in pair_report.records)),
            )
        dictionary_map = build_question_map(
            [record.paragraph_text for record in pair_report.records],
            [record.question_text for record in pair_report.records],
            [record.label for record in pair_report.records],
            units=dictionary_units,
            word_pos=pos_classes or None,
        )
        if analysis.dictionary.include_map:
            dictionary_block = _dictionary_block(
                pair_report.records,
                analysis.dictionary.folds,
                dictionary_seed,
                view,
                dictionary_map,
                dictionary_fold_ids,
                units=dictionary_units,
                unit_name=analysis.dictionary.units,
            )
            if dictionary_block.get("pos") is not None and pos_provenance:
                dictionary_block["pos"]["tagger"] = dict(pos_provenance)

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

    # The dictionary arm (CS-49): the question map as a pair scorer, beside the
    # run's own full-text cosine. The map reads no embedding, so the arm enters
    # the artifact as its own entry (the decision arm's pattern).
    if analysis.dictionary is not None and analysis.dictionary.scorings:
        records = pair_report.records
        paragraphs = [record.paragraph_text for record in records]
        questions = [record.question_text for record in records]
        labels = [record.label for record in records]
        primary = analysis.dictionary.primary or PRIMARY_SCORING
        probe_cells = [(row, column) for _, row, column in view.control.cells]
        grids = cross_fit_grids(
            paragraphs,
            questions,
            labels,
            dictionary_fold_ids,
            paragraphs=tuple(view.paragraphs),
            questions=tuple(view.questions),
            scorings=analysis.dictionary.scorings,
            units=dictionary_units,
            probe_cells=probe_cells,
        )
        blind_grids = None
        if analysis.dictionary.blind:
            permutation = blind_question_permutation(len(records), dictionary_seed)
            blind_grids = cross_fit_grids(
                paragraphs,
                [records[int(index)].question_text for index in permutation],
                labels,
                dictionary_fold_ids,
                paragraphs=tuple(view.paragraphs),
                questions=tuple(view.questions),
                scorings=analysis.dictionary.scorings,
                units=dictionary_units,
            )
        class_grids: dict[str, Any] = {}
        for restriction in analysis.dictionary.class_restrictions:
            class_grids[restriction] = cross_fit_grids(
                paragraphs,
                questions,
                labels,
                dictionary_fold_ids,
                paragraphs=tuple(view.paragraphs),
                questions=tuple(view.questions),
                scorings=(primary,),
                units=dictionary_units,
                restrict=(restriction,),
            )
        pos_grids: dict[str, Any] = {}
        for restriction in analysis.dictionary.pos_restrictions:
            pos_grids[restriction] = cross_fit_grids(
                paragraphs,
                questions,
                labels,
                dictionary_fold_ids,
                paragraphs=tuple(view.paragraphs),
                questions=tuple(view.questions),
                scorings=(primary,),
                restrict_pos=(restriction,),
                word_pos=pos_classes or None,
            )
        union = None
        if analysis.dictionary.union:
            # CS-53: the union arm needs BOTH maps behind the same folds — the
            # run's map is the phrase table, so the word map is built here.
            # Each half also carries its measure-free count arm (CS-49's
            # instrument, kept beside the union's number).
            word_grids = cross_fit_grids(
                paragraphs,
                questions,
                labels,
                dictionary_fold_ids,
                paragraphs=tuple(view.paragraphs),
                questions=tuple(view.questions),
                scorings=(primary, "support_count"),
            )
            phrase_grids = cross_fit_grids(
                paragraphs,
                questions,
                labels,
                dictionary_fold_ids,
                paragraphs=tuple(view.paragraphs),
                questions=tuple(view.questions),
                scorings=(primary, "support_count"),
                units=dictionary_units,
            )
            union = union_grids(word_grids, phrase_grids, scoring=primary)
        stability_grids: dict[str, Any] = {}
        stability_readout = None
        if analysis.dictionary.stability:
            # CS-54: the bootstrap over the reviewed pairs (the readout) and
            # the stability-filtered arms against the full z arm. Each fold's
            # filter is estimated on that fold's OWN training pairs, so a
            # judged pair is scored by a filter that never saw it.
            thresholds = analysis.dictionary.stability_thresholds
            report = estimate_question_map_stability(
                paragraphs,
                questions,
                labels,
                dictionary_map,
                resamples=analysis.dictionary.stability_resamples,
                seed=dictionary_seed,
                units=dictionary_units,
            )
            stability_readout = question_map_stability_block(report)
            stability_readout["kept_share"] = {
                f"{threshold:.2f}": kept_share(report, threshold) for threshold in thresholds
            }
            keep_by_fold_all: dict[str, dict[int, dict[str, frozenset[str]]]] = {
                f"{threshold:.2f}": {} for threshold in thresholds
            }
            for fold in sorted({int(value) for value in dictionary_fold_ids}):
                keep = [
                    index for index in range(len(labels)) if int(dictionary_fold_ids[index]) != fold
                ]
                fold_map = build_question_map(
                    [paragraphs[index] for index in keep],
                    [questions[index] for index in keep],
                    [labels[index] for index in keep],
                    units=dictionary_units,
                )
                fold_report = estimate_question_map_stability(
                    [paragraphs[index] for index in keep],
                    [questions[index] for index in keep],
                    [labels[index] for index in keep],
                    fold_map,
                    resamples=analysis.dictionary.stability_fold_resamples,
                    seed=dictionary_seed,
                    units=dictionary_units,
                )
                for threshold in thresholds:
                    keep_by_fold_all[f"{threshold:.2f}"][fold] = stability_filter(
                        fold_report, threshold
                    )
            for threshold in thresholds:
                name = f"{threshold:.2f}"
                stability_grids[name] = cross_fit_grids(
                    paragraphs,
                    questions,
                    labels,
                    dictionary_fold_ids,
                    paragraphs=tuple(view.paragraphs),
                    questions=tuple(view.questions),
                    scorings=(primary, "support_count"),
                    units=dictionary_units,
                    keep=stability_filter(report, threshold),
                    keep_by_fold=keep_by_fold_all[name],
                )
            stability_readout["thresholds"] = [f"{threshold:.2f}" for threshold in thresholds]
            if dictionary_block is not None:
                dictionary_block["stability"] = dict(stability_readout)
        abstention_fills: dict[str, Any] = {}
        if analysis.dictionary.abstention_fills:
            # CS-56: the empty-table fills beside the floor (the fills change
            # only the cells whose fold map is silent).
            for fill in analysis.dictionary.abstention_fills:
                abstention_fills[fill] = cross_fit_grids(
                    paragraphs,
                    questions,
                    labels,
                    dictionary_fold_ids,
                    paragraphs=tuple(view.paragraphs),
                    questions=tuple(view.questions),
                    scorings=(primary,),
                    units=dictionary_units,
                    fill=fill,
                )
            if dictionary_block is not None:
                dictionary_block["abstention"] = {
                    "silent_questions": len(dictionary_map.empty_questions),
                    "fills": list(analysis.dictionary.abstention_fills),
                    "note": (
                        "the empty-table fills (CS-56) scored beside the floor; the "
                        "silent questions are the ones with no positive reviewed pair"
                    ),
                }
        question_lexicon = None
        if analysis.dictionary.question_lexicon:
            # CS-55: the label side's own words — the question text's
            # vocabulary, read against the paragraphs. No pair label, no map
            # and no embedding enters the channel.
            lexicon = build_question_lexicon(questions)
            lexicon_matrices = {
                arm: lexicon_grid(
                    tuple(view.paragraphs),
                    tuple(view.questions),
                    lexicon.vocabularies,
                    lexicon.classes,
                    arm=arm,
                    idf=lexicon.idf,
                )
                for arm in LEXICON_ARMS
            }
            permutation = blind_question_permutation(len(records), dictionary_seed)
            blind_scores = lexicon_pair_scores(
                paragraphs,
                [questions[int(index)] for index in permutation],
                lexicon.vocabularies,
                lexicon.classes,
                idf=lexicon.idf,
            )
            question_lexicon = {
                "matrices": lexicon_matrices,
                "blind_scores": blind_scores,
            }
            if dictionary_block is not None:
                dictionary_block["question_lexicon"] = lexicon_readout(
                    lexicon,
                    pair_paragraphs=paragraphs,
                    pair_questions=questions,
                )
        models.append(
            build_dictionary_model(
                view,
                grids=grids,
                blind_grids=blind_grids,
                class_grids=class_grids,
                pos_grids=pos_grids,
                class_readout=(dictionary_block or {}).get("classes"),
                pos_readout=(dictionary_block or {}).get("pos"),
                reference_similarities=baseline.similarities,
                reference_name=entry.name,
                folds=analysis.dictionary.folds,
                seed=dictionary_seed,
                resamples=analysis.bootstrap.resamples,
                bootstrap_seed=bootstrap_seed,
                primary=primary,
                union=union,
                stability_grids=stability_grids,
                stability_readout=stability_readout,
                question_lexicon=question_lexicon,
                abstention_fills=abstention_fills,
                unit_name=analysis.dictionary.units,
                probe=(grids.get("probe") or {}).get(primary),
            )
        )

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
        dictionary=dictionary_block,
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
        if model.get("kind") == "dictionary":
            # The dictionary arm (CS-49): its own entry and its own section.
            if (model.get("readout") or {}).get("abstention_arms"):
                table_rows.append(
                    _dictionary_abstention_row(
                        model,
                        dictionary_block,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                        documents_by_code={
                            report["code"]: report["document"] for report in view.documents
                        },
                    )
                )
            elif (model.get("readout") or {}).get("question_lexicon"):
                table_rows.append(
                    _dictionary_lexicon_row(
                        model,
                        dictionary_block,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            elif (model.get("readout") or {}).get("stability_arms"):
                table_rows.append(
                    _dictionary_stability_row(
                        model,
                        dictionary_block,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            elif (model.get("readout") or {}).get("union"):
                table_rows.append(
                    _dictionary_phrases_row(
                        model,
                        dictionary_block,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            elif (model.get("readout") or {}).get("pos_arms"):
                table_rows.append(
                    _dictionary_pos_row(
                        model,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            elif (model.get("readout") or {}).get("class_arms"):
                table_rows.append(
                    _dictionary_class_row(
                        model,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            else:
                table_rows.append(
                    _dictionary_arm_row(
                        model,
                        run_id=omct_run_id,
                        n_control=view.control.n_cells,
                        n_docs=view.pooled["n_documents"],
                        seconds=payload["total_seconds"],
                    )
                )
            continue
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
    has_dictionary = dictionary_block is not None
    has_dictionary_arm = any(model.get("kind") == "dictionary" for model in models)
    if has_dictionary_arm:
        # The arm's own section (CS-49): its row shape carries the AUCs and
        # the comparison, so it is not the map's section (CS-48's). The
        # estimator's run (CS-50) adds the two z-scored arms' columns.
        has_pos_arms = any(
            (model.get("readout") or {}).get("pos_arms")
            for model in models
            if model.get("kind") == "dictionary"
        )
        has_class_arms = any(
            (model.get("readout") or {}).get("class_arms")
            for model in models
            if model.get("kind") == "dictionary"
        )
        has_union = any(
            (model.get("readout") or {}).get("union")
            for model in models
            if model.get("kind") == "dictionary"
        )
        has_abstention = any(
            (model.get("readout") or {}).get("abstention_arms")
            for model in models
            if model.get("kind") == "dictionary"
        )
        has_question_lexicon = any(
            (model.get("readout") or {}).get("question_lexicon")
            for model in models
            if model.get("kind") == "dictionary"
        )
        has_stability = any(
            (model.get("readout") or {}).get("stability_arms")
            for model in models
            if model.get("kind") == "dictionary"
        )
        if has_abstention:
            columns = ANALYZE_OMCT_DICTIONARY_ABSTENTION_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_ABSTENTION_KEY_COLUMNS
        elif has_question_lexicon:
            columns = ANALYZE_OMCT_DICTIONARY_LEXICON_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_LEXICON_KEY_COLUMNS
        elif has_stability:
            columns = ANALYZE_OMCT_DICTIONARY_STABILITY_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_STABILITY_KEY_COLUMNS
        elif has_union:
            columns = ANALYZE_OMCT_DICTIONARY_PHRASES_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_PHRASES_KEY_COLUMNS
        elif has_pos_arms:
            columns = ANALYZE_OMCT_DICTIONARY_POS_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_POS_KEY_COLUMNS
        elif has_class_arms:
            columns = ANALYZE_OMCT_DICTIONARY_CLASSES_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_CLASSES_KEY_COLUMNS
        elif any(
            (model.get("readout") or {}).get("scorings", {}).keys()
            & {"sum_zscore", "sum_zscore_positive"}
            for model in models
            if model.get("kind") == "dictionary"
        ):
            columns = ANALYZE_OMCT_DICTIONARY_ESTIMATOR_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_ESTIMATOR_KEY_COLUMNS
        else:
            columns = ANALYZE_OMCT_DICTIONARY_ARM_COLUMNS
            key_columns = ANALYZE_OMCT_DICTIONARY_ARM_KEY_COLUMNS
    elif has_dictionary:
        # The dictionary's own section (CS-48): the map is a description and
        # its rows are not arms, so the section and its columns are its own.
        # A dictionary-only run has no model rows; a later change set that
        # scores an arm brings its own row shape and section.
        table_rows = [
            _dictionary_row(
                dictionary_block,
                run_id=omct_run_id,
                n_control=view.control.n_cells,
                n_docs=view.pooled["n_documents"],
                seconds=payload["total_seconds"],
                documents_by_code={report["code"]: report["document"] for report in view.documents},
            )
        ]
        columns = ANALYZE_OMCT_DICTIONARY_COLUMNS
        key_columns = ANALYZE_OMCT_DICTIONARY_KEY_COLUMNS
    elif has_stage3:
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
    if dictionary_block is not None:
        shape = dictionary_block["map"]
        overall_availability = (dictionary_block.get("availability") or {}).get("overall") or {}
        units_label = shape.get("units") or UNIT_WORDS
        print(
            f"  dictionary map (CS-48): {shape['n_questions']} questions, "
            f"{shape['n_words']} {units_label}, {shape['n_stored_pairs']} "
            f"(question, unit) pairs; "
            f"{shape['n_questions_with_table']} tables, {shape['n_questions_empty']} silent; "
            f"cross-fit availability {_fmt(overall_availability.get('with_table_share'), 4)} "
            f"(evidence {_fmt(overall_availability.get('with_evidence_share'), 4)})"
        )
    for model in models:
        if model.get("kind") == "dictionary":
            # The dictionary arm (CS-49): its own entry, printed on its own lines.
            readout = model["readout"]
            comparison = model["comparisons"][DICTIONARY_ARM]
            within = comparison["auc_decomposition"]["within_group_paired"]
            blind_gap = (readout.get("blind_gap") or {}).get("primary")
            print(
                f"  [{DICTIONARY_ARM}] the question map ({readout['primary']}) against "
                f"{readout['reference']['model']}'s full-text cosine: within-document AUC "
                f"{_fmt(within.get('auc_arm'), 4)} vs {_fmt(within.get('auc_reference'), 4)} "
                f"(delta {_fmt(within.get('delta_auc'), 4)}, 95% CI "
                f"[{_fmt(within.get('ci_low'), 4)}, {_fmt(within.get('ci_high'), 4)}])"
            )
            blind_block = (readout.get("blind") or {}).get(readout["primary"]) or {}
            print(
                f"      the blind map's within-document AUC "
                f"{_fmt(blind_block.get('within_auc'), 4)} "
                f"(gap {_fmt(blind_gap, 4)}); the floor carries "
                f"{readout['floor']['n_floor_pairs']} pairs "
                f"({_fmt(readout['floor']['floor_share'], 4)}); control mean "
                f"{_fmt(comparison['control_shift']['mean_delta'], 4)}"
            )
            abstention = readout.get("abstention_arms")
            if abstention:
                fill_parts = ", ".join(
                    f"{name} {_fmt((block or {}).get('within_auc'), 4)} "
                    f"({_fmt((block or {}).get('delta_vs_full'), 4)})"
                    for name, block in abstention["fills"].items()
                )
                print(
                    f"      [fill] the floor {_fmt(abstention['floor']['within_auc'], 4)} "
                    f"over {abstention['floor_pairs']} pairs "
                    f"({_fmt(abstention['floor_negative_share'], 4)} of them negative); "
                    f"the fills: {fill_parts}"
                )
            label_side = readout.get("question_lexicon")
            if label_side:
                label_arms = label_side["arms"]
                print(
                    f"      [label side] the question's own words: all "
                    f"{_fmt(label_arms['all']['within_auc'], 4)}, idf "
                    f"{_fmt(label_arms['idf']['within_auc'], 4)}, topic-only "
                    f"{_fmt(label_arms['topic']['within_auc'], 4)}, register-only "
                    f"{_fmt(label_arms['register']['within_auc'], 4)}; against the map "
                    f"{_fmt(label_side['vs_map']['delta'], 4)} "
                    f"[{_fmt(label_side['vs_map']['ci_low'], 4)}, "
                    f"{_fmt(label_side['vs_map']['ci_high'], 4)}]; the blind gap "
                    f"{_fmt(label_side['blind']['gap'], 4)}"
                )
            stability_arms = readout.get("stability_arms")
            if stability_arms:
                # CS-54: the filtered arms against the full z arm (CS-36's test).
                for name, arm in stability_arms.items():
                    print(
                        f"      [stable {name}] the filtered z arm "
                        f"{_fmt(arm.get('within_auc'), 4)} against the full arm "
                        f"{_fmt(readout['scorings'][readout['primary']]['within_auc'], 4)} "
                        f"(delta {_fmt(arm.get('delta_vs_full'), 4)}, 95% CI "
                        f"[{_fmt(arm.get('ci_low'), 4)}, {_fmt(arm.get('ci_high'), 4)}]); "
                        f"the floor {arm.get('floor_pairs')} pairs"
                    )
            union = readout.get("union")
            if union is not None:
                # CS-53: the union arm and its registered increment.
                vs_word = union["vs_word"]
                print(
                    f"      [{union['arm']}] the word z arm {_fmt(union['word']['within_auc'], 4)} "
                    f"plus the phrase z arm {_fmt(union['phrase']['within_auc'], 4)} "
                    f"-> the union {_fmt(union.get('within_auc'), 4)} "
                    f"(vs the word arm {_fmt(vs_word.get('delta'), 4)}, 95% CI "
                    f"[{_fmt(vs_word.get('ci_low'), 4)}, {_fmt(vs_word.get('ci_high'), 4)}]; "
                    f"the bar {vs_word.get('bar')}); non-floor "
                    f"{_fmt((union.get('secondary_non_floor') or {}).get('within_auc'), 4)} "
                    f"over {union['floor']['n_floor_pairs']} floor pairs"
                )
            continue
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
