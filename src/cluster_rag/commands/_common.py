"""The CLI's shared helpers: the run-id plumbing, the table append, the guard.

Every command needs the same few things: the run id (and the dataset
prefix rules), the resolved-config writer, the comparison-table append and
the whole-text guard. They live here so the command modules hold only
their own logic.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from cluster_rag.config import AppConfig
from cluster_rag.data import Record, load_jsonl, load_split_ids, select_records_by_ids
from cluster_rag.dataset_stats import file_sha256
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.experiment_table import (
    append_table_rows,
    intro_for,
    section_heading,
    table_path,
    table_target,
)
from cluster_rag.label_catalog import LabelEntry, load_label_catalog
from cluster_rag.pos.cache import CachedTagger
from cluster_rag.pos.nltk_tagger import NltkTagger
from cluster_rag.text_processors.features.corpus_map import CorpusMap, build_corpus_map
from cluster_rag.text_processors.lexical.pos_class import (
    merge_distributions,
    modal_classes,
    word_tag_counts,
)

logger = logging.getLogger(__name__)


def _require_run_id(config: AppConfig) -> str:
    """Return the run_id from the config or raise, since artifacts need it."""
    if not config.run_id:
        raise ValueError(
            "run_id must be set in the config: commands write to results/<run_id>/ "
            "and the experiment log references the same run_id."
        )
    return config.run_id


def _append_command_rows(
    config: AppConfig,
    command: str,
    *,
    columns: Sequence[str],
    key_columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
    groups: bool = False,
) -> None:
    """Append one command's rows to its comparison table (command-derived).

    The command decides the ``(dataset, track)`` pair and therefore the file:
    the main pipeline's commands write to the per-dataset tables, the
    text-processing track's commands to the track's own pair of tables
    (AGENTS.md "Runs and where things live"). An unset path skips the append
    (throwaway configs in tests).
    """
    path = table_path(config.paths, command)
    if path is None:
        return
    target = table_target(command)
    append_table_rows(
        path,
        heading=section_heading(
            target.dataset,
            command=command,
            sweep_label=config.table.sweep_label,
            groups=groups,
        ),
        columns=columns,
        key_columns=key_columns,
        intro=intro_for(target.dataset, target.track),
        rows=rows,
    )


UPR_PREFIX = "upr-"  # UPR dataset runs (AGENTS.md "Runs and where things live")
OMCT_PREFIX = "omct-"  # auxiliary OMCT pairwise-data runs


def _warn_upr_prefix(run_id: str) -> None:
    """Warn (never fail) when a UPR command's run_id lacks the `upr-` prefix.

    UPR commands accept any run_id so existing configs keep working; the
    warning keeps the `<dataset>-<command>-<detail>[-<split>]` grammar visible.
    """
    if not run_id.startswith(UPR_PREFIX):
        logger.warning(
            "run_id %r does not start with the UPR dataset prefix %r "
            "(expected grammar: <dataset>-<command>-<detail>[-<split>] per AGENTS.md); "
            "the run proceeds without it.",
            run_id,
            UPR_PREFIX,
        )


def _omct_run_id(run_id: str) -> str:
    """Return the run_id with the `omct-` dataset prefix added by the command.

    OMCT commands prefix the run id themselves (the config carries the bare
    detail), so a config that already carries `omct-`/`upr-` would double up —
    warned about, never fatal.
    """
    if run_id.startswith((OMCT_PREFIX, UPR_PREFIX)):
        logger.warning(
            "run_id %r already carries a dataset prefix; the command adds %r "
            "itself, so artifacts land under results/%s%s/ (off-grammar but not fatal).",
            run_id,
            OMCT_PREFIX,
            OMCT_PREFIX,
            run_id,
        )
    return f"{OMCT_PREFIX}{run_id}"


def _fmt(value: float | None, decimals: int) -> str | None:
    """Fixed-precision rendering for table cells (None passes through)."""
    if value is None:
        return None
    return f"{value:.{decimals}f}"


def _write_resolved_config(results_dir: Path, config: AppConfig) -> None:
    """Freeze the resolved config into results/<run_id>/config.yaml."""
    payload = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    (results_dir / "config.yaml").write_text(payload, encoding="utf-8")


def _cached_embedder(embedder: Any, config: AppConfig, version: str) -> Any:
    """The embedder of one side, cached under that side's version slot."""
    if not config.cache.enabled:
        return embedder
    return CachedEmbedder(embedder, config.paths.cache, preprocessing_version=version)


def _check_reference_arm(
    config: AppConfig,
    reference_run_id: str | None,
    reference_block: dict[str, Any],
) -> dict[str, Any] | None:
    """The guard: the whole-text arm must reproduce the ``evaluate`` run.

    The UPR analogue of ``tools/verify_analysis_baseline.py``: the track's
    whole-text reference arm is the main pipeline's own scoring, so it must
    match the matching ``evaluate`` run's report **field by field**. A
    mismatch means the run's numbers cannot be compared with the pipeline's.

    Args:
        config: The resolved app config (the results folder).
        reference_run_id: The ``evaluate`` run to compare with (``None``
            disables the check).
        reference_block: The reference arm's block (its ``metrics``).

    Returns:
        The guard's outcome (recorded in the artifact), or ``None`` when the
        check was not requested.

    Raises:
        ValueError: If the check fails — a mismatch is never a warning.
    """
    if not reference_run_id:
        return None
    path = config.paths.results / reference_run_id / "metrics.json"
    if not path.is_file():
        raise ValueError(
            f"the reference evaluate run {reference_run_id!r} has no metrics artifact "
            f"({path}); the guard cannot check the whole-text arm"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("report")
    actual = reference_block["metrics"]["overall"]
    differences: list[str] = []
    if expected is None:
        differences.append("the reference run has no report block")
    else:
        if expected.get("overall") != actual:
            differences.append("the overall report differs")
        if expected.get("per_bucket") != reference_block["metrics"]["per_bucket"]:
            differences.append("the per-bucket report differs")
    if payload.get("n_records") != actual.get("n_records"):
        differences.append("the record count differs")
    if payload.get("n_labels") != actual.get("n_labels"):
        differences.append("the label count differs")
    if differences:
        raise ValueError(
            f"the whole-text reference arm does not reproduce the evaluate run "
            f"{reference_run_id!r}: {'; '.join(differences)}"
        )
    return {
        "reference_run": reference_run_id,
        "checked": ["overall", "per_bucket", "n_records", "n_labels"],
        "matches": True,
    }


def _upr_label_catalog(config: AppConfig) -> dict[str, LabelEntry]:
    """The optional label catalog (``paths.label_catalog``).

    An absent catalog is not an error: the pack's two catalog-gated feature
    groups are then unavailable and the artifact records the gap.
    """
    path = config.paths.label_catalog
    if path is None:
        return {}
    catalog = load_label_catalog(path)
    logger.info("the label catalog %s carries %d entries", path, len(catalog))
    return catalog


def _train_records(config: AppConfig) -> tuple[list[Record], str]:
    """The **train** split's records, from the fixed id list.

    The corpus map (CS-23) and the label lexicon (CS-35) are built from the
    train split only, so this is the corpus every train-derived artifact is
    built on. The id list is checked against the dataset's sha256 — the same
    consistency check the ``evaluate`` command makes — so a stale id list
    fails loudly instead of silently shifting the corpus.

    Args:
        config: The resolved app config.

    Returns:
        ``(records, source_sha256)`` — the train records in the id list's own
        order, and the dataset's sha256.

    Raises:
        ValueError: If the fixed split and the dataset disagree.
    """
    report = load_jsonl(config.paths.data_raw)
    source_sha = file_sha256(config.paths.data_raw)
    ids, provenance = load_split_ids(config.paths.data_splits, "train")
    recorded = provenance.get("source_sha256")
    if recorded and recorded != source_sha:
        raise ValueError(
            "train_ids.json was built from a different data source (sha256 mismatch): "
            "the fixed splits and the dataset disagree; do not re-split silently."
        )
    return select_records_by_ids(report.records, ids), source_sha


def _train_corpus_map(
    records: Sequence[Record],
    labels: Sequence[str],
    *,
    word_pos: Mapping[str, str] | None = None,
) -> CorpusMap:
    """The **corpus map** of the train split (the map's own builder).

    The map is the train split's own word↔label association (CS-23), so no val
    text, no val label and no test text enters it. It is the corpus every
    train-derived artifact of the track is built on: the map's own feature
    (CS-32), the word classes (CS-33) and the label lexicon (CS-35).

    Args:
        records: The train split's records.
        labels: The label space (every label, even the ones the train split
            never carries — they get an empty table).
        word_pos: The words' grammatical classes (CS-34), when the corpus was
            tagged.

    Returns:
        The corpus map.
    """
    return build_corpus_map(
        [record.text for record in records],
        [record.labels for record in records],
        labels,
        word_pos=word_pos,
    )


def build_pos_tagger(config: AppConfig) -> Any:
    """Build the part-of-speech tagger (the CLI's wiring seam, the track's CS-34).

    The tagger is a **model client**: it downloads its model into the run's own
    cache root and its answers are cached on disk (its own root under
    ``cache/``, never the embedding cache), so a re-run never re-tags a text.
    The track's pure package never imports the client — only the port.

    Returns:
        The cached tagger, or ``None`` when the tagger is off (the default).

    Raises:
        RuntimeError: If the model cannot be found or downloaded.
    """
    pos = config.pos
    if not pos.enabled:
        return None
    if pos.backend != "nltk":
        raise ValueError(f"Unknown part-of-speech backend {pos.backend!r}; expected 'nltk'")
    client = NltkTagger(
        pos.model_id,
        data_root=Path(config.paths.cache) / "pos" / "nltk_data",
        language=pos.language,
    )
    return CachedTagger(client, config.paths.cache)


def pos_word_classes(
    config: AppConfig,
    texts: Sequence[str],
) -> tuple[dict[str, str], dict[str, Counter[str]], dict[str, Any]]:
    """The train corpus's **grammatical classes** (the track's CS-34).

    The corpus is tagged with the track's own tokenization, so the tagger's
    tokens and the map's words line up by position; the per-word tag counts
    are merged over the corpus and each word's class is the class of its
    **modal** tag.

    Args:
        config: The resolved app config (the ``pos:`` section decides).
        texts: The corpus's texts (the train split).

    Returns:
        The per-word class, the per-word tag counts and the tagger's
        provenance (with the cache's counters). All three are empty when the
        tagger is off.
    """
    tagger = build_pos_tagger(config)
    if tagger is None:
        return {}, {}, {}
    distributions = []
    for text in texts:
        tagged = tagger.tag(text)
        distributions.append(word_tag_counts(text, tagged))
    merged = merge_distributions(distributions)
    classes = modal_classes(merged)
    info = tagger.info
    provenance = {
        "model_id": info.model_id,
        "digest": info.digest,
        "backend": info.backend,
        "cache_hits": getattr(tagger, "cache_hits", 0),
        "cache_misses": getattr(tagger, "cache_misses", 0),
        "n_stored": getattr(tagger, "n_stored", 0),
        "n_texts_tagged": len(texts),
        **info.extra,
    }
    return classes, merged, provenance
