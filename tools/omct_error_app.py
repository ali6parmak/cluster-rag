"""The OMCT dictionary error viewer's **Streamlit app** (CS-58).

Four read-only views over the round's own artifacts:

1. **One pair** — the paragraph, the question, the human verdict, the score,
   the rank, the fold map's contributing words, the floor's reason and the
   floor / neutral / max rank move.
2. **Errors** — false positives and false negatives under the evidence cut,
   with the floored pairs as their own class and the 4,000 unjudged control
   cells in a separate strip.
3. **One question** — the full map's table, the pair/positive counts, the
   per-fold availability, the derived word classes and (when the stability run
   is selected) the per-link stability.
4. **Compare arms** — the same 1,520 pairs under two ``(run, arm)`` series,
   with the rank movement.

The app is the **only** file that imports Streamlit (and pandas for display).
Every number is the data layer's: the artifact's own, or a recomputation that
is labelled "attribution" and guarded at 1e-12. A failed guard blocks every
view.

Run it from the repository root::

    uv run streamlit run tools/omct_error_app.py

The paths can be overridden in the sidebar or with the ``OMCT_VIEW_*``
environment variables (which the tests use).
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import omct_error_view as ev  # noqa: E402

VIEWS = ("One pair", "Errors", "One question", "Compare arms")
ERROR_LABELS = {
    "true_positive": "true positive",
    "false_positive": "false positive",
    "false_negative": "false negative",
    "true_negative": "true negative",
    "floored": "floored",
}
FILL_HELP = (
    "CS-56's three empty-table fills: floor = the row's smallest seen score "
    "minus one; neutral = the row's median; max = the row's maximum. Only the "
    "floor arm is the artifact's own; the other two are recomputed attribution."
)


def _env(name: str, default: str) -> str:
    """One source path: the environment variable, else the default."""
    return os.environ.get(name, default)


@st.cache_resource(show_spinner="reading the artifacts and rebuilding the five fold maps…")
def load_data(
    config: str,
    results: str,
    run_id: str,
    abstain_run: str,
    stability_run: str,
) -> dict[str, Any]:
    """Load one run, run the guard, and return the context and the report."""
    sources = ev.resolve_sources(
        Path(config),
        results=Path(results),
        run_id=run_id,
        abstain_run=abstain_run,
        stability_run=stability_run,
    )
    context = ev.load_context(sources)
    report = ev.check_all(sources, context=context)
    return {"sources": sources, "context": context, "report": report}


def _bundle(config: str, results: str, run_id: str) -> dict[str, Any] | None:
    """Load one run, or show the refusal (map-only run / missing artifact)."""
    try:
        return load_data(config, results, run_id, ev.ABSTAIN_RUN, ev.STABILITY_RUN)
    except ev.MapOnlyRunError as exc:
        st.warning(str(exc))
        return None
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"the tool could not read {run_id!r}: {exc}")
        return None


def _guard(report: ev.GuardReport) -> None:
    """The guard's own block, in the sidebar."""
    with st.sidebar.expander(f"Provenance and the guard ({'OK' if report.ok else 'MISMATCH'})"):
        st.code("\n".join(report.provenance.lines()), language="text")
        st.code("\n".join(report.lines()), language="text")


def _guard_failure(report: ev.GuardReport) -> None:
    """Show the failure and block every view."""
    st.error(
        "THE RUN AND THE DATA DISAGREE — the guard did not reproduce the run. "
        "The views are blocked; stale numbers are never shown."
    )
    for check in report.checks:
        if not check.ok:
            st.code(f"{check.name}: {check.detail}", language="text")
    with st.expander("the whole guard report"):
        st.code("\n".join(report.lines()), language="text")
    st.stop()


def _frame(rows: list[Any], fields: list[str]) -> pd.DataFrame:
    """A DataFrame with a fixed column order."""
    return pd.DataFrame([{field: asdict(row)[field] for field in fields} for row in rows])


def _open_pair(pair_id: str) -> None:
    """Open a pair in the One pair view (a button callback: before the widgets)."""
    st.session_state["pair_id"] = pair_id
    st.session_state["pair_document"] = "All"
    st.session_state["pair_verdict"] = "All"
    st.session_state["pair_floored"] = "All"
    st.session_state["view"] = "One pair"


def _pair_label(pair: ev.SourcePair) -> str:
    """The pair selector's label."""
    verdict = "positive" if pair.label == 1 else "negative"
    return f"{pair.pair_id} · {pair.document} · {verdict}"


def _question_label(context: ev.Context, question: str) -> str:
    """The question selector's label (its number and a short prefix)."""
    number = next(
        (pair.question_number for pair in context.pairs if pair.question_text == question), 0
    )
    short = " ".join(question.split())
    return f"Q{number} · {short[:90]}"


# --- 1. One pair ------------------------------------------------------------


def one_pair_view(context: ev.Context) -> None:
    """The core view: one reviewed pair, its evidence and its rank moves."""
    st.subheader("One pair — the core")
    st.caption(
        "The paragraph and the question are the source's; the score, the rank and the "
        "contributions are rebuilt from the pair's own fold map (attribution; guarded at "
        "1e-12 against the artifact)."
    )
    documents = sorted({pair.document for pair in context.pairs})
    columns = st.columns(3)
    document = columns[0].selectbox("document", ["All", *documents], key="pair_document")
    verdict = columns[1].selectbox(
        "human verdict", ["All", "positive", "negative"], key="pair_verdict"
    )
    floored_filter = columns[2].selectbox(
        "floored", ["All", "floored", "not floored"], key="pair_floored"
    )
    candidates = [
        pair
        for pair in context.pairs
        if (document == "All" or pair.document == document)
        and (verdict == "All" or (pair.label == 1) == (verdict == "positive"))
        and (
            floored_filter == "All"
            or ev.floored(context, context.pair_index[pair.pair_id])
            == (floored_filter == "floored")
        )
    ]
    if not candidates:
        st.info("No reviewed pair matches these filters.")
        return
    ids = [pair.pair_id for pair in candidates]
    if st.session_state.get("pair_id") not in ids:
        st.session_state["pair_id"] = ids[0]
    arms = list(context.run.arms)
    arm = st.selectbox(
        "arm",
        arms,
        index=arms.index(context.run.scoring) if context.run.scoring in arms else 0,
        key="pair_arm",
        help="The artifact stores the primary arm's judged pairs; the others are recomputed.",
    )
    fill = st.radio(
        "empty-table fill", list(ev.FILLS), horizontal=True, key="pair_fill", help=FILL_HELP
    )
    pair_id = st.selectbox(
        "pair",
        ids,
        key="pair_id",
        format_func=lambda value: _pair_label(next(p for p in candidates if p.pair_id == value)),
    )
    view = ev.pair_view(context, pair_id, arm=arm, fill=fill)
    left, right = st.columns(2)
    with left:
        st.markdown("**the paragraph**")
        st.write(view.pair.paragraph_text)
    with right:
        st.markdown("**the question**")
        st.write(view.pair.question_text)
    metrics = st.columns(6)
    metrics[0].metric("human verdict", "positive" if view.pair.label == 1 else "negative")
    metrics[1].metric("artifact score", f"{view.artifact_score:.6f}")
    metrics[2].metric(
        "rebuilt score",
        f"{view.rebuilt_score:.6f}",
        delta=f"{view.rebuilt_score - view.artifact_score:.1e}",
    )
    metrics[3].metric("rank", f"{view.rank} / {view.n_candidates}")
    metrics[4].metric("rank percentile", f"{view.rank_percentile:.4f}")
    metrics[5].metric("floored", "yes" if view.floored else "no")
    if not view.reproduction_ok:
        st.error(
            "THE REBUILT SCORE DOES NOT REPRODUCE THE ARTIFACT'S — the contributions are "
            "withheld and the guard blocks the views."
        )
    if view.floored:
        st.warning(
            f"FLOORED — the question has **no table** in fold {view.fold}'s map: "
            f"{view.fold_positive} positive and {view.fold_pairs} reviewed pairs for this "
            "question in that fold. Its score is the row's fill, not evidence."
        )
    else:
        st.info(
            f"Not floored — the question has {view.fold_positive} positive and "
            f"{view.fold_pairs} reviewed pairs in fold {view.fold}'s map."
        )
    st.markdown("**the rank move under the three fills**")
    moves = pd.DataFrame([asdict(move) for move in view.moves])
    st.dataframe(
        moves,
        hide_index=True,
        width="stretch",
        column_config={
            "fill": "fill",
            "score": st.column_config.NumberColumn("score", format="%.6f"),
            "rank": st.column_config.NumberColumn("rank", format="%d"),
            "n_candidates": st.column_config.NumberColumn("candidates", format="%d"),
            "delta_rank": st.column_config.NumberColumn("Δ rank", format="%+d"),
            "delta_percentile": st.column_config.NumberColumn("Δ percentile", format="%+.4f"),
        },
    )
    st.caption(
        "Δ is against the selected fill. Only the floor fill is the artifact's own; the "
        "neutral and max fills are CS-56's recomputed attribution."
    )
    st.markdown(f"**the contributing words — the fold {view.fold} map**")
    if not view.reproduction_ok:
        st.caption("withheld: the score did not reproduce.")
    elif not view.contributions:
        extra = (
            "the question has no table in this fold's map"
            if view.floored
            else "no word of the paragraph gave the pair points under this arm"
        )
        st.info(f"No contributing words: {extra}.")
    else:
        contributions = pd.DataFrame([asdict(row) for row in view.contributions])
        st.dataframe(
            contributions,
            hide_index=True,
            width="stretch",
            column_config={
                "word": "word",
                "n_with": st.column_config.NumberColumn("n_with", format="%d"),
                "df": st.column_config.NumberColumn("df", format="%d"),
                "log_odds": st.column_config.NumberColumn("log-odds", format="%+.6f"),
                "z_score": st.column_config.NumberColumn("z", format="%+.6f"),
                "contribution": st.column_config.NumberColumn("contribution", format="%+.6f"),
            },
        )
        st.caption(
            "The fold map's own rows (the full map is in One question). The contributions "
            "sum to the score under the floor fill; a non-floor fill replaces a floored "
            "pair's score and has no words behind it."
        )
    if arm != context.run.scoring:
        st.caption(
            f"Arm {arm!r} is **recomputed attribution**: the artifact stores the "
            f"{context.run.scoring!r} judged pairs only."
        )


# --- 2. Errors --------------------------------------------------------------


def errors_view(context: ev.Context) -> None:
    """False positives and false negatives, the floored class, the control strip."""
    st.subheader("Errors — false positives and false negatives")
    st.caption(
        "The evidence cut: a pair the map gives points to (score > 0) is a positive "
        "prediction — a display rule, not the run's metric (the run reports AUCs). The "
        "floored pairs are their own class; the control cells are never errors."
    )
    columns = st.columns(2)
    arms = list(context.run.arms)
    arm = columns[0].selectbox(
        "arm",
        arms,
        index=arms.index(context.run.scoring) if context.run.scoring in arms else 0,
        key="err_arm",
    )
    fill = columns[1].selectbox("empty-table fill", list(ev.FILLS), key="err_fill", help=FILL_HELP)
    rows = ev.error_rows(context, arm=arm, fill=fill)
    counts = Counter(row.error for row in rows)
    metrics = st.columns(6)
    metrics[0].metric("false positives", counts.get("false_positive", 0))
    metrics[1].metric("false negatives", counts.get("false_negative", 0))
    metrics[2].metric("true positives", counts.get("true_positive", 0))
    metrics[3].metric("true negatives", counts.get("true_negative", 0))
    metrics[4].metric("floored", counts.get("floored", 0))
    floored_positive = sum(1 for row in rows if row.floored and row.label == 1)
    metrics[5].metric("floored positives", floored_positive)
    filters = st.columns(5)
    documents = sorted({row.document for row in rows})
    document = filters[0].selectbox("document", ["All", *documents], key="err_document")
    shown = filters[1].selectbox(
        "what to show", ["the errors (FP + FN)", "floored", "all classes"], key="err_shown"
    )
    verdict = filters[2].selectbox(
        "human verdict", ["All", "positive", "negative"], key="err_verdict"
    )
    floored_filter = filters[3].selectbox(
        "floored", ["All", "floored", "not floored"], key="err_floored"
    )
    sort = filters[4].selectbox("sort", ["score (high first)", "rank (best first)"], key="err_sort")
    if shown == "the errors (FP + FN)":
        selected = [row for row in rows if row.error in ("false_positive", "false_negative")]
    elif shown == "floored":
        selected = [row for row in rows if row.error == "floored"]
    else:
        selected = list(rows)
    selected = [
        row
        for row in selected
        if (document == "All" or row.document == document)
        and (verdict == "All" or (row.label == 1) == (verdict == "positive"))
        and (floored_filter == "All" or row.floored == (floored_filter == "floored"))
    ]
    if sort == "score (high first)":
        selected.sort(key=lambda row: (-row.score, row.pair_id))
    else:
        selected.sort(key=lambda row: (row.rank, row.pair_id))
    limit = st.slider("rows", 5, 500, 100, key="err_limit")
    shown_rows = selected[:limit]
    st.caption(
        f"{len(selected)} rows match; showing {len(shown_rows)}. "
        "Floored pairs are a separate class, never folded into the false negatives."
    )
    table = pd.DataFrame(
        [
            {
                "pair": row.pair_id,
                "document": row.document,
                "human": "positive" if row.label == 1 else "negative",
                "class": ERROR_LABELS.get(row.error, row.error),
                "score": row.score,
                "rank": row.rank,
                "candidates": row.n_candidates,
                "rank percentile": row.rank_percentile,
                "floored": row.floored,
            }
            for row in shown_rows
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.6f"),
            "rank percentile": st.column_config.NumberColumn("rank percentile", format="%.4f"),
        },
    )
    if shown_rows:
        choices = [row.pair_id for row in shown_rows]
        columns = st.columns([3, 1])
        choice = columns[0].selectbox("open one of these pairs", choices, key="err_open")
        columns[1].button(
            "open in One pair",
            on_click=_open_pair,
            args=(choice,),
            key="err_open_button",
        )
    st.divider()
    with st.expander(
        f"Control strip — the {len(context.run.control)} unjudged cells (subset B)",
        expanded=False,
    ):
        st.caption(
            "Never reviewed; no label exists. These cells are never an error, never "
            "ground truth, and never pooled with the judged set. They appear here only."
        )
        control = context.run.control
        st.dataframe(
            pd.DataFrame([asdict(cell) for cell in control]),
            hide_index=True,
            width="stretch",
            column_config={
                "score": st.column_config.NumberColumn("score", format="%.6f"),
                "rank_percentile": st.column_config.NumberColumn("rank percentile", format="%.4f"),
            },
        )


# --- 3. One question --------------------------------------------------------


def one_question_view(context: ev.Context) -> None:
    """A question's table, counts, per-fold availability, classes and stability."""
    st.subheader("One question")
    st.caption(
        "The table is the artifact's **full** map (not the fold maps). The classes are "
        "derived from the artifact's own table counts; no POS is stored per word."
    )
    question = st.selectbox(
        "question",
        list(context.questions),
        key="question",
        format_func=lambda value: _question_label(context, value),
    )
    limit = st.slider("table words shown", 5, 200, 40, key="question_limit")
    stability_index = None
    stability_loaded = False
    if context.run.run_id == context.sources.stability_run:
        index = ev.stability_index_from_block(context.run.dictionary, run_id=context.run.run_id)
        stability_index = index.links
        stability_loaded = True
    view = ev.question_view(context, question, stability_index=stability_index)
    metrics = st.columns(6)
    metrics[0].metric("question", f"Q{view.question_number}")
    metrics[1].metric("reviewed pairs", view.pairs)
    metrics[2].metric("positive", view.positives)
    metrics[3].metric("negative", view.negatives)
    metrics[4].metric("table words", view.table_size)
    metrics[5].metric("labelled tables", view.n_labelled_tables)
    st.markdown("**the question**")
    st.write(view.question)
    st.markdown("**the full map's table — top words by z**")
    words = pd.DataFrame(
        [
            {
                "word": word.word,
                "n_with": word.n_with,
                "df": word.df,
                "lift": word.lift,
                "log-odds": word.log_odds,
                "z": word.z_score,
                "std err": word.standard_error,
                "tables": word.tables,
                "class": word.word_class,
            }
            for word in view.words[:limit]
        ]
    )
    st.dataframe(
        words,
        hide_index=True,
        width="stretch",
        column_config={
            "lift": st.column_config.NumberColumn("lift", format="%.4f"),
            "log-odds": st.column_config.NumberColumn("log-odds", format="%+.6f"),
            "z": st.column_config.NumberColumn("z", format="%+.6f"),
            "std err": st.column_config.NumberColumn("std err", format="%.6f"),
        },
    )
    st.caption(
        f"distinctive ≤ {view.distinctive_max_tables} tables · shared "
        f"{view.distinctive_max_tables + 1}–{view.generic_min_tables - 1} · generic ≥ "
        f"{view.generic_min_tables} (a quarter of the {view.n_labelled_tables} labelled "
        "tables). The counts are dictionary.words' own."
    )
    st.markdown("**per-fold availability**")
    folds = pd.DataFrame([asdict(row) for row in view.folds])
    st.dataframe(
        folds,
        hide_index=True,
        width="stretch",
        column_config={
            "fold": st.column_config.NumberColumn("fold", format="%d"),
            "pairs": st.column_config.NumberColumn("pairs", format="%d"),
            "positives": st.column_config.NumberColumn("positives", format="%d"),
            "table_size": st.column_config.NumberColumn("table size", format="%d"),
            "with_table": st.column_config.NumberColumn("with table", format="%d"),
            "with_evidence": st.column_config.NumberColumn("with evidence", format="%d"),
            "floored": st.column_config.NumberColumn("floored", format="%d"),
        },
    )
    st.caption(
        "with table / with evidence are the artifact's availability block; the table size "
        "is the rebuilt fold map's own (attribution)."
    )
    st.markdown("**word classes (derived from `dictionary.words`)**")
    class_columns = st.columns(3)
    for column, (name, count) in zip(class_columns, view.class_counts.items(), strict=False):
        column.metric(name, count)
    if stability_loaded:
        st.markdown("**stability of the table's links (CS-54)**")
        if view.stability:
            stability = pd.DataFrame([asdict(row) for row in view.stability])
            st.dataframe(
                stability,
                hide_index=True,
                width="stretch",
                column_config={
                    "present_share": st.column_config.NumberColumn("present_share", format="%.4f"),
                    "sign_stability": st.column_config.NumberColumn(
                        "sign_stability", format="%.4f"
                    ),
                    "head_stability": st.column_config.NumberColumn(
                        "head_stability", format="%.4f"
                    ),
                },
            )
            st.caption("A link survives τ when its sign stability is at least τ.")
        else:
            st.info("No stability links for this question's table.")
    else:
        st.caption(
            "Select the stability run in the sidebar to show its 27,979 per-link stability "
            "rows (indexed only here; the default run never reads the 60 MB artifact)."
        )


# --- 4. Compare arms --------------------------------------------------------


def _side_metrics(view: ev.RunView) -> None:
    """The artifact's own metrics for one (run, arm) side."""
    metrics = view.metrics
    if not metrics:
        st.info("This arm is not in the artifact's readout; only the movement is shown.")
        return
    columns = st.columns(4)
    columns[0].metric("within AUC", f"{metrics.get('within_auc', float('nan')):.4f}")
    columns[1].metric("pooled AUC", f"{metrics.get('pooled_auc', float('nan')):.4f}")
    columns[2].metric("floor pairs", metrics.get("n_floor_pairs", "n/a"))
    columns[3].metric("pairs", metrics.get("n_pairs", "n/a"))
    st.caption(
        "These four numbers are the artifact's own readout; the movement below is recomputed."
    )


def compare_view(context: ev.Context, config: str, results: str) -> None:
    """The same pairs under two (run, arm) series, with the rank movement."""
    st.subheader("Compare arms")
    st.caption(
        "The same 1,520 reviewed pairs under two series. The scores are not comparable "
        "across units (word vs phrase); the rank percentiles are."
    )
    left_column, right_column = st.columns(2)
    left_run = left_column.selectbox("left run", list(ev.RUN_CHOICES), index=0, key="cmp_left_run")
    right_run = right_column.selectbox(
        "right run", list(ev.RUN_CHOICES), index=0, key="cmp_right_run"
    )
    left_bundle = _bundle(config, results, left_run)
    right_bundle = _bundle(config, results, right_run)
    if left_bundle is None or right_bundle is None:
        return
    left_context = left_bundle["context"]
    right_context = right_bundle["context"]
    left_report = left_bundle["report"]
    right_report = right_bundle["report"]
    if not left_report.ok:
        st.error(f"The left run ({left_run}) fails its guard; the comparison is blocked.")
        with st.expander("left guard"):
            st.code("\n".join(left_report.lines()), language="text")
        return
    if not right_report.ok:
        st.error(f"The right run ({right_run}) fails its guard; the comparison is blocked.")
        with st.expander("right guard"):
            st.code("\n".join(right_report.lines()), language="text")
        return
    left_arms = list(left_context.run.arms)
    right_arms = list(right_context.run.arms)
    left_arm = left_column.selectbox(
        "left arm",
        left_arms,
        index=left_arms.index("sum_positive") if "sum_positive" in left_arms else 0,
        key="cmp_left_arm",
    )
    right_arm = right_column.selectbox(
        "right arm",
        right_arms,
        index=(
            right_arms.index("sum_zscore_positive") if "sum_zscore_positive" in right_arms else 0
        ),
        key="cmp_right_arm",
    )
    left_view = ev.run_view(left_context, left_arm)
    right_view = ev.run_view(right_context, right_arm)
    left_column.markdown(f"**left — {left_run} / {left_arm}**")
    _side_metrics(left_view)
    right_column.markdown(f"**right — {right_run} / {right_arm}**")
    _side_metrics(right_view)
    rows = ev.compare_views(left_view, right_view, left_context, right_context)
    if not rows:
        st.warning("The two runs share no pair_id.")
        return
    summary = Counter(
        "up" if row.delta_rank < 0 else "down" if row.delta_rank > 0 else "same" for row in rows
    )
    metrics = st.columns(4)
    metrics[0].metric("moved up", summary.get("up", 0))
    metrics[1].metric("moved down", summary.get("down", 0))
    metrics[2].metric("unchanged", summary.get("same", 0))
    metrics[3].metric("pairs", len(rows))
    st.caption(
        "Movement is right minus left: 'up' means a better (smaller) rank under the right "
        "series. It is recomputed for attribution, not an artifact metric."
    )
    filters = st.columns(3)
    documents = sorted({row.document for row in rows})
    document = filters[0].selectbox("document", ["All", *documents], key="cmp_document")
    verdict = filters[1].selectbox(
        "human verdict", ["All", "positive", "negative"], key="cmp_verdict"
    )
    order = filters[2].selectbox(
        "sort",
        ["largest movement", "pair id", "human verdict"],
        key="cmp_sort",
    )
    selected = [
        row
        for row in rows
        if (document == "All" or row.document == document)
        and (verdict == "All" or (row.label == 1) == (verdict == "positive"))
    ]
    if order == "largest movement":
        selected.sort(key=lambda row: (-abs(row.delta_percentile), row.pair_id))
    elif order == "human verdict":
        selected.sort(key=lambda row: (-row.label, row.pair_id))
    else:
        selected.sort(key=lambda row: row.pair_id)
    limit = st.slider("rows", 5, 500, 100, key="cmp_limit")
    shown = selected[:limit]
    table = pd.DataFrame(
        [
            {
                "pair": row.pair_id,
                "document": row.document,
                "human": "positive" if row.label == 1 else "negative",
                "left score": row.left_score,
                "left rank": row.left_rank,
                "left percentile": row.left_percentile,
                "right score": row.right_score,
                "right rank": row.right_rank,
                "right percentile": row.right_percentile,
                "Δ rank": row.delta_rank,
                "Δ percentile": row.delta_percentile,
            }
            for row in shown
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "left score": st.column_config.NumberColumn("left score", format="%.6f"),
            "right score": st.column_config.NumberColumn("right score", format="%.6f"),
            "left percentile": st.column_config.NumberColumn("left pct", format="%.4f"),
            "right percentile": st.column_config.NumberColumn("right pct", format="%.4f"),
            "Δ rank": st.column_config.NumberColumn("Δ rank", format="%+d"),
            "Δ percentile": st.column_config.NumberColumn("Δ pct", format="%+.4f"),
        },
    )
    if left_view.unit != right_view.unit:
        st.warning(
            f"The units differ ({left_view.unit} vs {right_view.unit}): the scores are not "
            "comparable; the ranks and their movement are."
        )


# --- the app ----------------------------------------------------------------


def main() -> None:
    """The app's entry point: sources, the guard, then the four views."""
    st.set_page_config(page_title="OMCT dictionary error viewer", page_icon="🔎", layout="wide")
    st.title("OMCT dictionary error viewer")
    st.caption(
        "The dictionary round's mistakes on the 1,520 reviewed pairs. OMCT auxiliary data "
        "only — never a UPR metric; read-only."
    )
    st.sidebar.header("Sources")
    config = st.sidebar.text_input(
        "config", value=_env("OMCT_VIEW_CONFIG", str(ev.DEFAULT_CONFIG)), key="src_config"
    )
    results = st.sidebar.text_input(
        "results", value=_env("OMCT_VIEW_RESULTS", "results"), key="src_results"
    )
    run_id = st.sidebar.selectbox(
        "run",
        list(ev.RUN_CHOICES),
        index=list(ev.RUN_CHOICES).index(_env("OMCT_VIEW_RUN", ev.DEFAULT_RUN)),
        key="src_run",
    )
    if st.sidebar.button("reload the artifacts", key="src_reload"):
        load_data.clear()
    bundle = _bundle(config, results, run_id)
    if bundle is None:
        st.stop()
    context = bundle["context"]
    report = bundle["report"]
    assert isinstance(context, ev.Context)
    assert isinstance(report, ev.GuardReport)
    if not report.ok:
        _guard_failure(report)
    _guard(report)
    st.sidebar.caption(
        f"run {context.run.run_id} · {context.unit_name} · primary arm {context.run.scoring}"
    )
    view = st.sidebar.radio("view", list(VIEWS), index=0, key="view")
    if view == "One pair":
        one_pair_view(context)
    elif view == "Errors":
        errors_view(context)
    elif view == "One question":
        one_question_view(context)
    else:
        compare_view(context, config, results)


if __name__ == "__main__":
    main()
