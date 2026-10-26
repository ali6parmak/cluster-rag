"""Per-dataset experiment tables (Phase 3.5): ``docs/experiments_table_upr.md`` /
``docs/experiments_table_omct.md``.

CLI commands append one row per run to their **dataset's** table file
(``upr`` for stats/split/embed/evaluate, ``omct`` for probe-omct and
``stats --omct`` — derived from the command, never from the run name),
under a section heading: ``## <UPR|OMCT> — <sweep-label>`` for runs coming
from a sweep (the sweep label is injected into the run's config as
``table.sweep_label``; the probe's per-document breakdown appends
``… - groups``), or the standalone ``## <UPR|OMCT> — <command>`` heading when
no sweep was involved.

Rows upsert by their key columns (``run_id``, plus ``transform``/group
columns for multi-row commands): the table always shows the latest run per
key, while the narrative log ``docs/experiments.md`` stays append-only and
records every pass. Values come from each run's own artifacts (written by
the command right before the row), counts sit next to numbers, and "n/a"
marks unreported values. A table path is config
(``paths.experiment_table_upr`` / ``paths.experiment_table_omct`` and the
track's two keys); commands skip the append when their file is unset (e.g.
tests with throwaway results folders).

The text-processing track (``TEXT_PROCESSING_PLAN.md`` section 2) owns its
own pair of table files, ``docs/experiments_table_text_processing_omct.md``
and ``docs/experiments_table_text_processing_upr.md``. Which file a command
writes to is **command-derived**, never a heuristic: every command maps to a
``(dataset, track)`` pair in :data:`COMMAND_TARGETS`, and the track picks the
file (main pipeline -> the per-dataset files, text processing -> the track's
files).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

DATASET_TAGS: dict[str, str] = {"upr": "UPR", "omct": "OMCT"}

# The main pipeline's tables and the text-processing track's tables are
# separate files; a command's track decides which pair it writes to.
MAIN_TRACK = "main"
TEXT_PROCESSING_TRACK = "text_processing"


@dataclass(frozen=True)
class TableTarget:
    """Where a command's comparison rows go: one dataset, one track."""

    dataset: str
    track: str = MAIN_TRACK


# The command -> (dataset, track) mapping. Command-derived, never a heuristic:
# the track's command family (analyze-omct / analyze-upr) maps to
# dataset=omct|upr + track=text_processing, i.e. the track's own table files.
COMMAND_TARGETS: dict[str, TableTarget] = {
    "stats": TableTarget("upr"),
    "stats-omct": TableTarget("omct"),
    "split": TableTarget("upr"),
    "embed": TableTarget("upr"),
    "labelembed": TableTarget("upr"),
    "evaluate": TableTarget("upr"),
    "probe-omct": TableTarget("omct"),
    # Text-processing track (TEXT_PROCESSING_PLAN.md section 2).
    "analyze-omct": TableTarget("omct", TEXT_PROCESSING_TRACK),
    "decide-omct": TableTarget("omct", TEXT_PROCESSING_TRACK),
    "analyze-upr": TableTarget("upr", TEXT_PROCESSING_TRACK),
}


def table_target(command: str) -> TableTarget:
    """The ``(dataset, track)`` a command's rows belong to.

    Raises:
        ValueError: If the command is not registered (a typo must fail loudly
            instead of silently writing to the wrong file).
    """
    try:
        return COMMAND_TARGETS[command]
    except KeyError as exc:
        raise ValueError(
            f"Unknown command {command!r}; expected one of {sorted(COMMAND_TARGETS)}"
        ) from exc


class TablePaths(Protocol):
    """The paths-config attributes the table writer needs (structural type)."""

    experiment_table_upr: Path | None
    experiment_table_omct: Path | None
    experiment_table_text_processing_upr: Path | None
    experiment_table_text_processing_omct: Path | None


def table_path(paths: TablePaths, command: str) -> Path | None:
    """The table file a command appends to (``None`` = unset -> skip).

    The path comes from the command's ``(dataset, track)`` pair, so a track
    command can never write into the main pipeline's table file.
    """
    target = table_target(command)
    if target.track == TEXT_PROCESSING_TRACK:
        if target.dataset == "upr":
            return paths.experiment_table_text_processing_upr
        return paths.experiment_table_text_processing_omct
    if target.dataset == "upr":
        return paths.experiment_table_upr
    return paths.experiment_table_omct


INTRO = """\
# Experiment tables

Comparison tables keyed by run — auto-appended by the CLI commands. The
append-only narrative log is `docs/experiments.md` (one row per run, with
full revisions/digests). Values come from each run's own artifacts; counts
sit next to numbers; "n/a" = not reported by the run.
"""


def intro_for(dataset: str, track: str = MAIN_TRACK) -> str:
    """Intro text of a table file (names the dataset and the track).

    The text-processing track's files point at the track's own narrative log
    (``docs/experiments_text_processing.md``), not at the main pipeline's.
    """
    tag = DATASET_TAGS[dataset]
    if track == TEXT_PROCESSING_TRACK:
        return (
            f"# {tag} text-processing experiment tables\n"
            "\n"
            "Comparison tables for the text-processing track "
            "(`TEXT_PROCESSING_PLAN.md`), keyed by run — auto-appended by the "
            "track's CLI commands. The append-only narrative log is "
            "`docs/experiments_text_processing.md`. Values come from each run's "
            "own artifacts; counts sit next to numbers; "
            '"n/a" = not reported by the run.\n'
        )
    return (
        f"# {tag} experiment tables\n"
        "\n"
        "Comparison tables keyed by run — auto-appended by the CLI commands "
        "(one file per dataset). The append-only narrative log is "
        f"`docs/experiments.md` (one row per run, with full revisions/digests). "
        "Values come from each run's own artifacts; counts sit next to numbers; "
        '"n/a" = not reported by the run.\n'
    )


def section_heading(
    dataset: str, *, command: str, sweep_label: str | None = None, groups: bool = False
) -> str:
    """Heading of one table section: the sweep label wins, else the command.

    ``groups=True`` appends the probe's per-document breakdown suffix; the
    standalone (no-sweep) heading is ``## <UPR|OMCT> — <command>``.
    """
    tag = DATASET_TAGS[dataset]
    return f"{tag} — {sweep_label or command}" + (" - groups" if groups else "")


# Columns of the auto-appended tables.
EMBED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "backend",
    "model",
    "model_version",
    "dim",
    "ctx",
    "kinds",
    "preprocess",
    "n_texts",
    "seconds",
    "cache_hits",
    "cache_misses",
)
EMBED_KEY_COLUMNS: tuple[str, ...] = ("run_id",)

# The evaluate table, regrouped (Phase 3.5): the f1 decision view first
# (decision + its F1/coverage numbers), the ranking view (decision-rule
# independent) second. "embedding" (Phase 4 opening D) is the detail of the
# embedding run the predictor consumed (eval.embedding_run_id minus its
# "upr-embed-" prefix); empty for embedding-free predictors — full lineage
# (model, digest, prefixes) lives in the run's metrics.json.
EVALUATE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "split",
    "predictor",
    "embedding",
    "f1 decision",
    "micro-F1",
    "macro-F1",
    "sample-F1",
    "coverage",
    "P@5",
    "R@10",
    "nDCG@5",
    "shortlist R@50",
    "n_texts",
    "seconds",
)
EVALUATE_KEY_COLUMNS: tuple[str, ...] = ("run_id",)

PROBE_OMCT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "backend",
    "model",
    "model_version",
    "dim",
    "ctx",
    "transform",
    "pair_auc",
    "same-doc hit@1",
    "same-doc hit@10",
    "same-doc recall@10",
    "paras>ctx",
    "n_paras",
    "n_pairs(pos)",
    "seconds",
)
PROBE_OMCT_KEY_COLUMNS: tuple[str, ...] = ("run_id", "transform")

PROBE_OMCT_GROUPS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "transform",
    "document",
    "source_csv",
    "n_pairs",
    "pos",
    "neg",
    "pair_auc",
    "same-doc hit@1",
    "same-doc hit@10",
    "n_rows_with_gold",
)
PROBE_OMCT_GROUPS_KEY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "transform",
    "document",
    "source_csv",
)

# The text-processing track's Stage-0 analysis table (the `analyze-omct`
# command): one row per model of a run. The discrimination number (the
# reviewed set's pair AUC) sits next to the control-side counterparts (the
# control's score level and the rank percentiles of the positives, the
# verified negatives and the control) — the track's reporting rule: no
# discrimination number is quoted without its control-side counterpart.
ANALYZE_OMCT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "backend",
    "model",
    "model_version",
    "dim",
    "ctx",
    "transform",
    "chunker",
    "pair_auc",
    "ctrl_mean",
    "ctrl_p50",
    "pos_rank_p50",
    "neg_rank_p50",
    "ctrl_rank_p50",
    "n_pairs",
    "n_control",
    "n_docs",
    "seconds",
)
ANALYZE_OMCT_KEY_COLUMNS: tuple[str, ...] = ("run_id", "model")

# The text-processing track's Stage-1 arms table: one row per (run, model,
# arm). The discrimination number (the reviewed set's pair AUC) sits next to
# its **paired** difference against the full-text baseline arm (the bootstrap
# CI of the AUC difference), next to the **control-side counterpart** (the
# control's score level and the shift against the baseline) and next to the
# **de-shifted AUC** (the control's per-document mean shift subtracted from
# the arm's reviewed-set scores) — the track's reporting rule: no
# discrimination number is quoted without the control side next to it, and
# the comparison is paired.
#
# The Stage-2 ladder adds three columns: the per-document **macro AUC**
# (the macro beside every pooled number), the **floor-gap delta** (the
# track's falsification statistic: the verified positives' mean minus the
# control's mean, against the baseline) and the **chunks per text** (the k of
# the max-over-k inflation the control side measures).
#
# The Stage-0 shape (ANALYZE_OMCT_COLUMNS) stays as it is, so a run without
# arms keeps its own section's columns; a run *with* arms uses this shape.
ANALYZE_OMCT_ARMS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "backend",
    "model",
    "model_version",
    "dim",
    "ctx",
    "arm",
    "chunker",
    "aggregation",
    "transform",
    "pair_auc",
    "macro_auc",
    "within_auc",
    "cross_auc",
    "within_d_auc",
    "within_d_auc_ci",
    "d_auc",
    "d_auc_ci",
    "deshift_auc",
    "floor_gap_delta",
    "ctrl_mean",
    "ctrl_mean_delta",
    "ctrl_p50",
    "ctrl_rank_p50",
    "pos_rank_p50",
    "neg_rank_p50",
    "ctrl_rank_p50",
    "chunks_per_text",
    "n_pairs",
    "n_control",
    "n_docs",
    "seconds",
)
ANALYZE_OMCT_ARMS_KEY_COLUMNS: tuple[str, ...] = ("run_id", "model", "arm")

# The text-processing track's Stage-3 table: one row per (run, model) of a run
# that carries the **learned** pair scorer (the feature pack + the combiner
# behind the document-level holdout).
#
# The headline is the **within-document** AUC (the pooled number sits beside
# it, never alone: 58.7% of the reviewed set's positive-negative pairs are
# cross-document, so the pooled number moves with the documents' score
# levels). Beside it: the paired difference against the full-text baseline
# arm (the bootstrap CI of the difference), the label-permutation floor (the
# shuffled labels must land at chance), the control-side counterpart (the
# control's rank position, which is scale-free — the learned arm's level is
# a probability and the baseline's a cosine) and the standardized floor gap.
#
# Two changes came with Stage 3.1b (the feature-group ablation): the
# **feature groups** the pack kept (the ablation's arm) and the **hit@1**
# numbers (the positives' hit@1 for the arm and for the cosine — the arm
# ranks pairs better than the cosine but retrieves worse). The two floor
# columns carry the **within-document** weighted floors (the pair-count-
# weighted mean of the documents' own AUCs), because that is the statistic
# the headline is measured against; the unweighted per-rotation means are in
# the artifact.
#
# The Stage-0 shape (ANALYZE_OMCT_COLUMNS) and the Stage-2 arms shape
# (ANALYZE_OMCT_ARMS_COLUMNS) stay as they are: a run without a learned
# scorer keeps its own section's shape.
ANALYZE_OMCT_STAGE3_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "backend",
    "model",
    "model_version",
    "dim",
    "ctx",
    "groups",
    "features",
    "combiner",
    "within_auc",
    "within_d_auc",
    "within_d_auc_ci",
    "pooled_auc",
    "pooled_d_auc",
    "pooled_d_auc_ci",
    "macro_auc",
    "perm_auc_within",
    "ctrl_role_auc_within",
    "ctrl_mean",
    "ctrl_rank_p50",
    "ctrl_rank_p50_delta",
    "pos_rank_p50",
    "neg_rank_p50",
    "floor_gap",
    "hit1_pos",
    "hit1_pos_ref",
    "n_pairs",
    "n_control",
    "n_docs",
    "seconds",
)
ANALYZE_OMCT_STAGE3_KEY_COLUMNS: tuple[str, ...] = ("run_id", "model")

# The Stage-3 **reporting** (CS-6e) adds its own columns to the Stage-3
# shape: the calibration's errors, the per-document threshold's operating
# point and the safe-negative rule's reach. The rule's columns are the
# **two-part** variant (the cosine's margin *and* the arm's margin) and the
# control-role floor's harm at the same reach; the track log carries all
# three variants side by side.
ANALYZE_OMCT_STAGE3_REPORTING_COLUMNS: tuple[str, ...] = (
    *ANALYZE_OMCT_STAGE3_COLUMNS,
    "ece_arm",
    "ece_arm_macro",
    "ece_cos",
    "brier_arm",
    "thr_criterion",
    "thr_macro_recall",
    "thr_oof_recall",
    "rule_cov",
    "rule_harm",
    "rule_floor_harm",
)

# The decision probe's table (the `decide-omct` command, Stage 3.5): one row
# per arm of a run. The headline is the **within-document** AUC (the pooled
# number sits beside it, never alone); the **calibration** columns sit next to
# it because a decision model is only useful as a decider when its
# probability is calibrated (tev1's card says its calibration is untested);
# and the **competence** columns carry the synthetic control's AUC, which is
# what tells "the reviewed set is at the ceiling" from "the model is the wrong
# shape".
DECIDE_OMCT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "model",
    "model_version",
    "state",
    "question_type",
    "scope",
    "within_auc",
    "pooled_auc",
    "macro_auc",
    "cross_auc",
    "ece",
    "ece_macro",
    "brier",
    "pos_mean",
    "neg_mean",
    "ctrl_mean",
    "floor_gap",
    "comp_auc",
    "comp_pos_mean",
    "comp_neg_mean",
    "n_pairs",
    "n_control",
    "n_comp",
    "n_calls",
    "trunc",
    "seconds",
)
DECIDE_OMCT_KEY_COLUMNS: tuple[str, ...] = ("run_id", "model", "state", "question_type")

# The text-processing track's UPR table (the `analyze-upr` command): one row
# per arm of a run. The **decision view** (the F1s and the coverage) comes
# first, then the **ranking view** (the decision-rule-independent P@k / R@k /
# nDCG@k and the shortlist recall), then the **background side** — the
# non-gold labels' rank position and score level, and their shift against the
# whole-text reference arm. The track's reporting rule carries over: no
# headline number is quoted without its background-side counterpart beside it.
#
# The `hit@k` columns are the gold cells' hit@k (is the gold label among the
# text's top-k labels?): the UPR analogue of the OMCT probe's same-document
# hit@k.
ANALYZE_UPR_COLUMNS: tuple[str, ...] = (
    "run_id",
    "date",
    "split",
    "arm",
    "predictor",
    "chunker",
    "aggregation",
    "micro-F1",
    "macro-F1",
    "sample-F1",
    "coverage",
    "P@5",
    "R@10",
    "nDCG@5",
    "shortlist R@50",
    "hit@1",
    "hit@10",
    "gold_rank_p50",
    "bg_rank_p50",
    "bg_mean",
    "bg_mean_delta",
    "bg_rank_delta",
    "chunks_per_text",
    "n_texts",
    "seconds",
)
ANALYZE_UPR_KEY_COLUMNS: tuple[str, ...] = ("run_id", "arm")


def _cell(value: Any) -> str:
    """Render one table cell; ``None``/missing/empty becomes "n/a"."""
    if value is None:
        return "n/a"
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    return text or "n/a"


def _render_row(cells: Sequence[str]) -> str:
    """One markdown table row line."""
    return "| " + " | ".join(cells) + " |"


def _separator(n_columns: int) -> str:
    return "|" + "|".join(["---"] * n_columns) + "|"


def _cells_of(line: str) -> list[str]:
    """Parse an existing data-row line into cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _heading_line_of(lines: list[str], heading: str) -> int | None:
    """Line index of ``## <heading>`` (exactly once), else ``None``. Raises on duplicates."""
    target = f"## {heading}"
    matches = [index for index, line in enumerate(lines) if line.rstrip() == target]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"duplicate table heading {target!r} in the experiment tables file")
    return matches[0]


def _row_indexes_of_table(lines: list[str], heading_index: int) -> list[int]:
    """Line indexes of the data rows that follow the heading (header/separator skipped)."""
    cursor = heading_index + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
        cursor += 1  # header row
    if cursor < len(lines) and _cells_of(lines[cursor])[:1] == ["---"]:
        cursor += 1  # separator row
    data_rows: list[int] = []
    while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
        data_rows.append(cursor)
        cursor += 1
    return data_rows


def _sync_section_header(lines: list[str], heading_index: int, columns: Sequence[str]) -> None:
    """Rewrite a section's header when the command's column set changed.

    The rows of a section are upserted by run key, so a re-run rewrites them
    in the command's *current* shape; the header has to follow, or the table
    would show rows with a different number of cells than the header.
    """
    cursor = heading_index + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines) or not lines[cursor].lstrip().startswith("|"):
        return
    if _cells_of(lines[cursor]) != list(columns):
        lines[cursor] = _render_row(columns)
        if cursor + 1 < len(lines) and _cells_of(lines[cursor + 1])[:1] == ["---"]:
            lines[cursor + 1] = _separator(len(columns))


def append_table_rows(
    path: Path | str,
    *,
    heading: str,
    columns: Sequence[str],
    key_columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    intro: str = INTRO,
) -> None:
    """Append (or replace) run rows in the markdown table under ``heading``.

    Each row is a mapping of column name -> cell value; keys must appear in
    ``columns``. A row whose key cells match an existing data row replaces it
    in place (upsert-by-run); others append after the table's last data row.
    A missing file starts with ``intro``; a missing heading starts a new
    section at the end.
    """
    keys = list(key_columns)
    unknown = [key for key in keys if key not in columns]
    if unknown:
        raise ValueError(f"key columns {unknown} are not part of the table columns")
    key_positions = [list(columns).index(key) for key in keys]

    target = Path(path)
    if target.is_file():
        lines = target.read_text(encoding="utf-8").splitlines()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = intro.splitlines()

    heading_index = _heading_line_of(lines, heading)
    if heading_index is None:
        lines += ["", f"## {heading}", "", _render_row(columns), _separator(len(columns))]
        heading_index = len(lines) - 2
    else:
        _sync_section_header(lines, heading_index, columns)
    data_rows = _row_indexes_of_table(lines, heading_index)

    for row in rows:
        cells = [_cell(row.get(column)) for column in columns]
        row_keys = [cells[position] for position in key_positions]
        replaced = False
        for line_index in data_rows:
            existing = _cells_of(lines[line_index])
            if [existing[position] for position in key_positions] == row_keys:
                lines[line_index] = _render_row(cells)
                replaced = True
                break
        if not replaced:
            insert_at = data_rows[-1] + 1 if data_rows else len(lines)
            lines.insert(insert_at, _render_row(cells))
            data_rows = [index + 1 if index >= insert_at else index for index in data_rows] + [
                insert_at
            ]

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
