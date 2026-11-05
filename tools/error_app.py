"""The error viewer's **Streamlit app** (V1).

Five views over the same three tidy tables (``tools/error_view.py``): the
mistakes sorted by confidence, one text with its word x label matrix, the
aggregates (confusion, per-label rates, the words behind false positives,
coverage holes), two runs side by side, and an export.

The app is the **only** file that imports Streamlit (and pandas for display).
Every number is the data layer's: the dictionary predictions are recomputed
from the corpus map's artifact, the model column is the LLM judge's surrogate
(never a prediction), and the guard runs before any view is shown. A failed
guard blocks the app: stale numbers are never displayed beside fresh ones.

Run it from the repository root:

    uv run streamlit run tools/error_app.py

The source paths can be overridden in the sidebar (or with the ``ERROR_VIEW_*``
environment variables, which the tests use).
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import error_view as ev  # noqa: E402

VIEWS = ("Errors", "One text", "Aggregates", "Compare runs", "Export")


def _env(name: str, default: str) -> str:
    """One source path: the environment variable, else the default."""
    return os.environ.get(name, default)


@st.cache_data(show_spinner="reading the artifacts and recomputing the classifier…")
def load_data(
    config: str,
    results: str,
    split: str,
    corpus_run: str,
    eval_run: str,
    surrogate_run: str,
    near_miss_k: int,
) -> dict[str, object]:
    """Load the context, run the guard, and build the tables (cached)."""
    sources = ev.resolve_sources(
        Path(config),
        results=Path(results),
        split=split,
        corpus_run=corpus_run,
        eval_run=eval_run,
        surrogate_run=surrogate_run,
    )
    context = ev.load_context(sources)
    report = ev.check_all(sources, near_miss_k=near_miss_k, context=context)
    if not report.ok:
        return {"report": report, "context": context}
    tables = ev.build_tables(context, k=report.provenance.k, near_miss_k=near_miss_k)
    catalog = ev.load_run_series(
        Path(results), expected_n=len(context.records), expected_split=split
    )
    return {
        "report": report,
        "context": context,
        "tables": tables,
        "catalog": catalog,
    }


def _texts_frame(tables: ev.Tables) -> pd.DataFrame:
    """The texts table as a DataFrame."""
    return pd.DataFrame([asdict(row) for row in tables.texts])


def _cells_frame(tables: ev.Tables) -> pd.DataFrame:
    """The cells table as a DataFrame."""
    return pd.DataFrame([asdict(row) for row in tables.cells])


def _words_frame(tables: ev.Tables) -> pd.DataFrame:
    """The words table as a DataFrame."""
    return pd.DataFrame([asdict(row) for row in tables.words])


def _top_words(tables: ev.Tables) -> dict[tuple[str, str], str]:
    """Each cell's two strongest contributing words, as one display string."""
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in tables.words:
        grouped.setdefault((row.text_id, row.label), []).append((row.word, row.z_score))
    out: dict[tuple[str, str], str] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda item: (-item[1], item[0]))
        out[key] = ", ".join(f"{word} {z:+.3f}" for word, z in entries[:2])
    return out


def _text_lookup(context: ev.Context) -> dict[str, str]:
    """Text id -> the text (for snippets and the one-text select)."""
    return {str(record.id): record.text for record in context.records}


def _open_selected(text_id: str) -> None:
    """Open a text in the One text view (a button callback: before the radio)."""
    st.session_state["selected_text"] = text_id
    st.session_state["view"] = "One text"


def _snippet(text: str, limit: int = 140) -> str:
    """A one-line snippet of a text."""
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _sign_colour(value: object) -> str:
    """A CSS colour for a signed number (green up, red down, grey nothing)."""
    if value is None or pd.isna(value):
        return "color: #8b949e"
    number = float(value)  # type: ignore[arg-type]
    if number > 0.0:
        return "color: #1a7f37"
    if number < 0.0:
        return "color: #cf222e"
    return "color: #8b949e"


def _render_guard(report: ev.GuardReport, *, compact: bool = False) -> None:
    """The guard's own checks (and the provenance), in the sidebar/top."""
    if compact:
        with st.expander(f"Provenance and the guard ({'OK' if report.ok else 'MISMATCH'})"):
            st.code("\n".join(report.provenance.lines()), language="text")
            st.code("\n".join(line for line in report.lines() if line), language="text")
        return
    st.code("\n".join(report.lines()), language="text")


def errors_view(
    tables: ev.Tables,
    context: ev.Context,
    *,
    top_words: dict[tuple[str, str], str],
) -> None:
    """The mistakes, filterable and sorted by how confident they were."""
    cells = _cells_frame(tables)
    lookup = _text_lookup(context)
    texts = _texts_frame(tables)[["text_id", "length_bucket", "text"]]
    cells = cells.merge(texts, on="text_id", how="left")
    cells["top_words"] = [
        top_words.get((text_id, label), "")
        for text_id, label in zip(cells.text_id, cells.label, strict=False)
    ]
    labels = list(context.labels)
    st.subheader("Errors — what the dictionary classifier got wrong")
    st.caption(
        "The cut is top_k k=2. A near-miss is a gold label at rank 3..K. "
        "The score is the map's own sum of positive z-scores; the run's metric is in the guard."
    )
    columns = st.columns(4)
    with columns[0]:
        gold_labels = st.multiselect("gold label", labels, key="err_gold")
        predicted_labels = st.multiselect("predicted label", labels, key="err_pred")
    with columns[1]:
        error_types = st.multiselect(
            "error type",
            ["fp", "near_miss", "fn", "tp", "ok"],
            default=["fp", "near_miss", "fn"],
            key="err_type",
        )
        buckets = st.multiselect(
            "length bucket",
            list(ev.LENGTH_BUCKETS),
            default=list(ev.LENGTH_BUCKETS),
            key="err_bucket",
        )
    with columns[2]:
        evidence = st.selectbox(
            "evidence",
            ["any", "with evidence", "empty slot (score 0)"],
            key="err_evidence",
        )
        minimum = st.number_input(
            "minimum score",
            min_value=0.0,
            value=0.0,
            step=0.5,
            key="err_min_score",
        )
    with columns[3]:
        search = st.text_input("text contains", key="err_search")
        limit = st.slider("rows", 10, 500, 100, step=10, key="err_rows")

    frame = cells[cells["error"].isin(error_types)]
    if gold_labels:
        frame = frame[frame["gold"] & frame["label"].isin(gold_labels)]
    if predicted_labels:
        frame = frame[frame["predicted"] & frame["label"].isin(predicted_labels)]
    if buckets:
        frame = frame[frame["length_bucket"].isin(buckets)]
    if evidence == "with evidence":
        frame = frame[frame["has_evidence"]]
    elif evidence == "empty slot (score 0)":
        frame = frame[~frame["has_evidence"]]
    if minimum > 0.0:
        frame = frame[frame["score"] >= float(minimum)]
    if search:
        frame = frame[frame["text"].str.contains(search, case=False, regex=False)]
    frame = frame.sort_values(["score", "text_id"], ascending=[False, True])

    st.caption(
        f"{len(frame)} mistake cells; showing the top {min(limit, len(frame))} by confidence "
        "(the score that made the pick; for an FN, the score that fell short)."
    )
    display = frame.head(limit)[
        [
            "text_id",
            "error",
            "label",
            "score",
            "rank",
            "gap_to_cut",
            "has_evidence",
            "gold",
            "predicted",
            "length_bucket",
            "top_words",
            "text",
        ]
    ].copy()
    display["text"] = display["text"].map(_snippet)
    display["score"] = display["score"].map(lambda value: f"{value:.3f}")
    display["gap_to_cut"] = display["gap_to_cut"].map(lambda value: f"{value:+.3f}")
    st.dataframe(display, hide_index=True, width="stretch")
    if not frame.empty:
        choices = list(dict.fromkeys(frame["text_id"]))
        picked = st.selectbox(
            "open one of these texts",
            choices,
            format_func=lambda text_id: f"{text_id} — {_snippet(lookup.get(text_id, ''), 80)}",
            key="err_open",
        )
        st.button(
            "Open in One text",
            key="err_open_button",
            on_click=_open_selected,
            args=(picked,),
        )


def one_text_view(tables: ev.Tables, context: ev.Context) -> None:
    """One text: the ranking, the word x label matrix, and the contrast."""
    lookup = _text_lookup(context)
    ids = list(lookup)
    selected = st.session_state.get("selected_text", ids[0])
    if selected not in lookup:
        selected = ids[0]
    picked = st.selectbox(
        "text",
        ids,
        index=ids.index(selected),
        format_func=lambda text_id: f"{text_id} — {_snippet(lookup[text_id], 90)}",
        key="one_text_pick",
    )
    st.session_state["selected_text"] = picked
    record = next(record for record in context.records if str(record.id) == picked)
    text_row = next(row for row in tables.texts if row.text_id == picked)
    cells = [row for row in tables.cells if row.text_id == picked]
    by_rank = sorted(cells, key=lambda row: row.rank)
    st.subheader(f"One text — {picked}")
    st.write(record.text)
    st.caption(
        f"gold: {' · '.join(record.labels)} — the classifier's pick: "
        f"{' · '.join(text_row.predicted_labels) or '(none)'}"
    )
    columns = st.columns(5)
    columns[0].metric("tp", text_row.tp)
    columns[1].metric("fp", text_row.fp)
    columns[2].metric("fn", text_row.fn)
    columns[3].metric("F1", f"{text_row.text_f1:.3f}")
    columns[4].metric("margin r2−r3", f"{text_row.margin:+.3f}")
    st.caption(
        f"length {text_row.n_chars} chars ({text_row.length_bucket}); "
        f"{text_row.n_evidence} labels carry evidence; "
        + (
            "an empty second slot (tie-break)"
            if text_row.has_empty_slot
            else "both picks carry evidence"
        )
        + "."
    )

    st.markdown("**The ranking around the cut**")
    st.caption(
        "rank and score are the classifier's own (the empty tables sit last, never at zero); "
        "'empty slot' marks a pick with no contributing word."
    )
    top_n = st.slider("rows", 5, len(by_rank), min(15, len(by_rank)), key="one_text_rows")
    shown = by_rank[:top_n]
    gold_set = set(record.labels)
    extra = [row for row in by_rank if row.label in gold_set and row not in shown]
    rows = shown + extra
    frame = pd.DataFrame(
        [
            {
                "rank": row.rank,
                "label": row.label,
                "score": row.score,
                "gold": row.gold,
                "predicted": row.predicted,
                "error": row.error,
                "has_evidence": row.has_evidence,
                "gap_to_cut": row.gap_to_cut,
                "has_table": row.has_table,
            }
            for row in rows
        ]
    )
    st.dataframe(
        frame.style.format({"score": "{:.3f}", "gap_to_cut": "{:+.3f}"}, na_rep="—").map(
            _sign_colour, subset=["score", "gap_to_cut"]
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("**The words x labels matrix — the dictionary and the model**")
    st.caption(
        "dictionary = the map's positive z-score contribution (the score's exact decomposition); "
        "model = the LLM judge's ridge surrogate (CS-24), never a prediction. "
        "'—' means no contribution / not in the decision model's vocabulary."
    )
    top_ranked = [row.label for row in by_rank[:5]]
    matrix_labels: list[str] = []
    for label in [*record.labels, *text_row.predicted_labels, *top_ranked]:
        if label not in matrix_labels:
            matrix_labels.append(label)
    against = ev.contrast_label(context.tables, record.text, record.labels, context.labels)
    if against and against not in matrix_labels:
        matrix_labels.append(against)
    st.caption(
        "contrast B: "
        + (
            f"{against} (the map's runner-up among the labels the text does not carry)"
            if against
            else "—"
        )
    )
    word_rows = [
        row for row in tables.words if row.text_id == picked and row.label in matrix_labels
    ]
    if word_rows:
        dictionary = pd.DataFrame(
            [{"label": row.label, "word": row.word, "value": row.contribution} for row in word_rows]
        ).pivot_table(index="word", columns="label", values="value", aggfunc="sum")
        model = pd.DataFrame(
            [
                {"label": row.label, "word": row.word, "value": row.surrogate_coefficient}
                for row in word_rows
            ]
        ).pivot_table(index="word", columns="label", values="value", aggfunc="first")
        dictionary = dictionary.reindex(columns=matrix_labels)
        model = model.reindex(columns=matrix_labels)
        dictionary.columns = pd.MultiIndex.from_product([["dictionary"], dictionary.columns])
        model.columns = pd.MultiIndex.from_product([["model"], model.columns])
        matrix = pd.concat([dictionary, model], axis=1).sort_index()
        st.dataframe(
            matrix.style.format("{:+.3f}", na_rep="—").map(_sign_colour),
            width="stretch",
        )
    else:
        st.info("No word of this text gives any of these labels a positive z-score.")
    with st.expander("The dictionary's evidence-only ranking (CS-47)"):
        st.caption(
            "The same map arm, but only the labels with a positive score: this is the "
            "evidence-bearing view. The classifier's ranking above also carries the "
            "zero-score tie-break picks."
        )
        ranking = ev.evidence_ranking(context.tables, record.text, context.labels)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": row.rank,
                        "label": row.label,
                        "score": f"{row.score:.3f}",
                        "words": ", ".join(f"{word} {score:+.3f}" for word, score in row.words[:2]),
                    }
                    for row in ranking[:15]
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def aggregates_view(tables: ev.Tables, context: ev.Context) -> None:
    """The aggregates: confusion, per-label rates, FP words, holes, buckets."""
    st.subheader("Aggregates — where the errors live")
    st.caption(
        "Every number is recomputed from the map artifact. The near-miss counts as a miss "
        "in the miss rate; the FP rate uses the non-gold cells as its denominator."
    )
    confusion = ev.label_confusion(tables.cells)
    rates = ev.label_rates(tables.cells, context.labels, context.bucket_of_label)
    words = ev.false_positive_words(tables.cells, tables.words)
    holes = ev.coverage_holes(tables.cells, context.labels)
    buckets = ev.bucket_rates(tables.texts, tables.cells, context.labels, context.bucket_of_label)
    tabs = st.tabs(
        [
            "Label confusion",
            "Per-label rates",
            "Words in false positives",
            "Coverage holes",
            "Buckets",
        ]
    )
    with tabs[0]:
        st.caption(
            "gold A -> predicted B, over the cells the classifier predicted: which labels "
            "does it swap? (A and B differ; the support is A's gold cells.)"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "gold": row.gold_label,
                        "predicted": row.predicted_label,
                        "n": row.n,
                        "support": row.support,
                        "share of A": f"{row.share:.3f}",
                    }
                    for row in confusion[:40]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with tabs[1]:
        st.caption(
            "Per label: precision, recall, the miss rate (fn + near_miss over gold) and the "
            "FP rate (fp over the non-gold cells)."
        )
        frame = pd.DataFrame([asdict(row) for row in rates])
        frame = frame.sort_values(["miss_rate", "fp_rate"], ascending=False)
        st.dataframe(
            frame.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "miss_rate": "{:.3f}",
                    "fp_rate": "{:.4f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with tabs[2]:
        st.caption(
            "The words whose contributions sit in predicted cells: n_cells counts the "
            "predicted cells a word contributed to, split into hits and false positives."
        )
        st.dataframe(
            pd.DataFrame([asdict(row) for row in words[:50]]).style.format(
                {"mean_z": "{:+.3f}", "precision": "{:.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
    with tabs[3]:
        st.caption(
            "The labels the classifier never reaches — and, separately, the labels it never "
            "reaches with a word that gave points (the empty-slot picks are not evidence)."
        )
        columns = st.columns(3)
        columns[0].metric("never predicted", len(holes.never_predicted))
        columns[1].metric("never with evidence", len(holes.never_predicted_with_evidence))
        columns[2].metric("no train table", len(holes.no_table))
        st.write("**Never predicted:** " + (", ".join(holes.never_predicted) or "—"))
        st.write(
            "**Never predicted with evidence:** "
            + (", ".join(holes.never_predicted_with_evidence) or "—")
        )
        st.write("**No train table (held out):** " + (", ".join(holes.no_table) or "—"))
    with tabs[4]:
        frame = pd.DataFrame([asdict(row) for row in buckets])
        st.dataframe(
            frame.style.format({"micro_f1": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )


def _series_choices(
    catalog: ev.SeriesCatalog,
    context: ev.Context,
    tables: ev.Tables,
) -> tuple[list[ev.RunSeries], list[str], str, str]:
    """The series to choose from, their labels, and the preferred default pair."""
    series = [*catalog.series, ev.dictionary_series(context, tables, k=2)]
    labels = [item.label for item in series]
    preferred_a = next(
        (label for label in labels if "upr-analyze-final-champion :: full_text" in label),
        labels[0],
    )
    preferred_b = next(
        (
            label
            for label in labels
            if "upr-analyze-final-decision-feature :: learned_histgb" in label
        ),
        labels[1] if len(labels) > 1 else labels[0],
    )
    return series, labels, preferred_a, preferred_b


def compare_view(
    tables: ev.Tables,
    context: ev.Context,
    catalog: ev.SeriesCatalog,
) -> None:
    """Two series side by side, with label detail only where predictions exist."""
    st.subheader("Compare runs — what improved and what worsened")
    st.caption(
        "The per-text f1_counts are the artifacts' own; the counts are joined by row = the "
        "val split's own order. Label-level detail exists only for the dictionary series "
        "(V1); the pack and the LLM judge store counts only."
    )
    if catalog.rejected:
        with st.expander(f"{len(catalog.rejected)} series refused by the guard"):
            st.code("\n".join(catalog.rejected), language="text")
    series, labels, preferred_a, preferred_b = _series_choices(catalog, context, tables)
    columns = st.columns(2)
    a_label = columns[0].selectbox("series A", labels, index=labels.index(preferred_a), key="cmp_a")
    b_label = columns[1].selectbox("series B", labels, index=labels.index(preferred_b), key="cmp_b")
    a = series[labels.index(a_label)]
    b = series[labels.index(b_label)]
    if a.label == b.label:
        st.info("Pick two different series.")
        return
    try:
        rows = ev.compare_series(a, b, context.val_ids)
    except ValueError as exc:
        st.error(str(exc))
        return
    columns = st.columns(3)
    columns[0].metric(f"A micro-F1 ({a.label})", f"{a.micro_f1:.4f}")
    columns[1].metric(f"B micro-F1 ({b.label})", f"{b.micro_f1:.4f}")
    columns[2].metric("Δ micro-F1 (B − A)", f"{b.micro_f1 - a.micro_f1:+.4f}")
    lookup = _text_lookup(context)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame["text"] = frame["text_id"].map(lambda text_id: _snippet(lookup.get(text_id, ""), 120))
    frame["direction"] = frame.apply(
        lambda row: "improved" if row["improved"] else ("worsened" if row["worsened"] else "same"),
        axis=1,
    )
    frame = frame[frame["direction"] != "same"]
    frame = frame.reindex(frame["delta_f1"].abs().sort_values(ascending=False).index)
    st.caption(
        f"{len(frame)} texts changed (of {len(rows)}): "
        f"{int(frame['improved'].sum())} improved, {int(frame['worsened'].sum())} worsened."
    )
    show_all = st.checkbox("show every changed text", key="cmp_all")
    shown = frame if show_all else frame.head(50)
    st.dataframe(
        shown[
            [
                "text_id",
                "direction",
                "delta_f1",
                "delta_tp",
                "delta_fp",
                "delta_fn",
                "f1_a",
                "f1_b",
                "text",
            ]
        ].style.format({"delta_f1": "{:+.3f}", "f1_a": "{:.3f}", "f1_b": "{:.3f}"}),
        hide_index=True,
        width="stretch",
    )
    dictionary_side = [item for item in (a, b) if item.arm == "dictionary (recomputed)"]
    if dictionary_side and not shown.empty:
        text_id = st.selectbox(
            "label detail for one changed text",
            list(shown["text_id"]),
            key="cmp_text",
        )
        cells = [row for row in tables.cells if row.text_id == text_id]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": row.rank,
                        "label": row.label,
                        "gold": row.gold,
                        "predicted": row.predicted,
                        "score": f"{row.score:.3f}",
                        "error": row.error,
                        "has_evidence": row.has_evidence,
                    }
                    for row in sorted(cells, key=lambda item: item.rank)
                    if row.gold or row.predicted
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        dictionary_text = next(row for row in tables.texts if row.text_id == text_id)
        st.caption(
            "These labels are the dictionary series'; the other series has no label detail. "
            f"The dictionary's margin between rank 2 and rank 3 on this text is "
            f"{dictionary_text.margin:+.3f}; the margin's change is not readable from counts."
        )
    else:
        st.info(
            "Neither series stores predictions: the counts tell you which texts moved, "
            "not which labels moved. V2 would add a predictions block."
        )


def export_view(
    tables: ev.Tables,
    context: ev.Context,
    report: ev.GuardReport,
    catalog: ev.SeriesCatalog,
) -> None:
    """Pull the filtered rows for the log or a pre-registration."""
    st.subheader("Export — the rows and their provenance")
    st.caption("Downloads only; nothing is written under results/ or docs/.")
    table_name = st.selectbox(
        "table",
        ["errors (cells, filtered)", "texts", "cells (all)", "words", "compare (A/B)"],
        key="export_table",
    )
    provenance = report.provenance.lines()
    rows: list[object]
    if table_name.startswith("errors"):
        error_types = st.multiselect(
            "error type",
            ["fp", "near_miss", "fn"],
            default=["fp", "near_miss", "fn"],
            key="export_errors",
        )
        rows = [row for row in tables.cells if row.error in error_types]
        fields = (
            "text_id",
            "label",
            "gold",
            "predicted",
            "score",
            "rank",
            "error",
            "has_evidence",
            "gap_to_cut",
        )
    elif table_name == "texts":
        rows = list(tables.texts)
        fields = (
            "text_id",
            "text",
            "n_chars",
            "length_bucket",
            "gold_labels",
            "predicted_labels",
            "tp",
            "fp",
            "fn",
            "text_f1",
            "margin",
        )
    elif table_name == "cells (all)":
        rows = list(tables.cells)
        fields = (
            "text_id",
            "label",
            "gold",
            "predicted",
            "score",
            "rank",
            "error",
            "has_evidence",
            "gap_to_cut",
        )
    elif table_name == "words":
        rows = list(tables.words)
        fields = (
            "text_id",
            "label",
            "word",
            "z_score",
            "log_odds",
            "n_with",
            "df",
            "word_class",
            "is_name_word",
            "surrogate_coefficient",
        )
    else:
        series, labels, preferred_a, preferred_b = _series_choices(catalog, context, tables)
        a_label = st.selectbox("series A", labels, index=labels.index(preferred_a), key="exp_a")
        b_label = st.selectbox("series B", labels, index=labels.index(preferred_b), key="exp_b")
        a = series[labels.index(a_label)]
        b = series[labels.index(b_label)]
        try:
            rows = ev.compare_series(a, b, context.val_ids)
        except ValueError as exc:
            st.error(str(exc))
            return
        fields = (
            "text_id",
            "f1_a",
            "f1_b",
            "delta_f1",
            "delta_tp",
            "delta_fp",
            "delta_fn",
            "improved",
            "worsened",
        )
    payload = ev.to_csv(rows, fields, provenance=provenance)
    st.caption(f"{len(rows)} rows; the provenance block is at the top as # comments.")
    st.code("\n".join(payload.splitlines()[:14]), language="text")
    st.download_button(
        "download CSV",
        payload,
        file_name=f"error_view_{table_name.split()[0]}.csv",
        mime="text/csv",
        key="export_download",
    )
    st.download_button(
        "download provenance",
        "\n".join(provenance) + "\n",
        file_name="error_view_provenance.txt",
        mime="text/plain",
        key="export_provenance",
    )


def main() -> None:
    """The app's entry point: sources, the guard, then the five views."""
    st.set_page_config(page_title="Error viewer — UPR val", layout="wide")
    st.title("Error viewer — UPR val")
    st.caption(
        "The dictionary classifier's mistakes, recomputed from the train corpus map. "
        "UPR val only; test untouched; read-only."
    )
    st.sidebar.header("Sources")
    config = st.sidebar.text_input(
        "config", value=_env("ERROR_VIEW_CONFIG", str(ev.DEFAULT_CONFIG)), key="src_config"
    )
    results = st.sidebar.text_input(
        "results", value=_env("ERROR_VIEW_RESULTS", "results"), key="src_results"
    )
    split = st.sidebar.text_input("split", value=_env("ERROR_VIEW_SPLIT", "val"), key="src_split")
    near_miss_k = st.sidebar.selectbox(
        "near-miss window K",
        [5, 10],
        index=0,
        key="src_near_miss",
        help="gold at ranks 3..K is a near-miss",
    )
    with st.sidebar.expander("run ids"):
        corpus_run = st.text_input(
            "corpus map", value=_env("ERROR_VIEW_CORPUS_RUN", ev.CORPUS_MAP_RUN), key="src_corpus"
        )
        eval_run = st.text_input(
            "evaluation", value=_env("ERROR_VIEW_EVAL_RUN", ev.EVAL_RUN), key="src_eval"
        )
        surrogate_run = st.text_input(
            "surrogate",
            value=_env("ERROR_VIEW_SURROGATE_RUN", ev.SURROGATE_RUN),
            key="src_surrogate",
        )
    if st.sidebar.button("reload the artifacts", key="src_reload"):
        load_data.clear()
    try:
        data = load_data(
            config, results, split, corpus_run, eval_run, surrogate_run, int(near_miss_k)
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"the tool could not read its sources: {exc}")
        st.stop()
    report = data["report"]
    assert isinstance(report, ev.GuardReport)
    if not report.ok:
        st.error(
            "THE ARTIFACTS AND THE DATA DISAGREE — the guard did not reproduce the run. "
            "The views are blocked; stale numbers are never shown."
        )
        for check in report.checks:
            if not check.ok:
                st.code(f"{check.name}: {check.detail}", language="text")
        with st.expander("the whole guard report"):
            st.code("\n".join(report.lines()), language="text")
        st.stop()
    context = data["context"]
    tables = data["tables"]
    catalog = data["catalog"]
    assert isinstance(context, ev.Context)
    assert isinstance(tables, ev.Tables)
    assert isinstance(catalog, ev.SeriesCatalog)
    _render_guard(report, compact=True)
    view = st.sidebar.radio(
        "view",
        VIEWS,
        index=0,
        key="view",
    )
    top_words = _top_words(tables)
    if view == "Errors":
        errors_view(tables, context, top_words=top_words)
    elif view == "One text":
        one_text_view(tables, context)
    elif view == "Aggregates":
        aggregates_view(tables, context)
    elif view == "Compare runs":
        compare_view(tables, context, catalog)
    else:
        export_view(tables, context, report, catalog)


if __name__ == "__main__":
    main()
