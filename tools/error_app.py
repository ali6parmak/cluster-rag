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

import html as html_lib
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

TOOLS = Path(__file__).resolve().parent
CSS_PATH = TOOLS / "error_app.css"
sys.path.insert(0, str(TOOLS))
import error_view as ev  # noqa: E402

VIEWS = ("Errors", "One text", "Aggregates", "Compare runs", "Compare hitmaps", "Export")


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


def _top_words(tables: ev.Tables) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Each cell's three strongest contributing words, strongest first."""
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in tables.words:
        grouped.setdefault((row.text_id, row.label), []).append((row.word, row.z_score))
    out: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for key, entries in grouped.items():
        entries.sort(key=lambda item: (-item[1], item[0]))
        out[key] = entries[:3]
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
        return "color: #6c7086"  # Catppuccin overlay0
    number = float(value)  # type: ignore[arg-type]
    if number > 0.0:
        return "color: #a6e3a1"  # Catppuccin green
    if number < 0.0:
        return "color: #f38ba8"  # Catppuccin red
    return "color: #6c7086"


# --- the read-only HTML table -------------------------------------------------
#
# The table cells carry the diagnostics in place: a native tooltip on the text,
# coloured badges for the error type, dots for the booleans (never checkboxes),
# and chips with a small intensity bar for the scores. Everything is escaped;
# the helpers return HTML fragments for :func:`_html_table`.


def _esc(value: object) -> str:
    """The value as escaped HTML text."""
    return html_lib.escape(str(value), quote=True)


def _expandable_text(text: str) -> str:
    """The full text in a cell clipped by CSS; hovering it expands it in place."""
    return f'<span class="ev-text">{_esc(" ".join(text.split()))}</span>'


def _dot(active: bool, tone: str, *, on: str = "●", off: str = "·") -> str:
    """A boolean as a coloured dot (a read-only indicator, never a checkbox)."""
    css = f"ev-dot ev-dot-{tone}" if active else "ev-dot ev-dot-off"
    return f'<span class="{css}">{on if active else off}</span>'


def _gold_dot(active: bool) -> str:
    """The gold flag: a gold star, or a dim dot."""
    return _dot(active, "gold", on="★", off="·")


def _predicted_dot(active: bool) -> str:
    """The prediction flag: a mauve dot, or a dim dot."""
    return _dot(active, "mauve", on="●", off="·")


def _evidence_dot(active: bool) -> str:
    """The evidence flag: a green dot, or the amber empty-slot ring."""
    if active:
        return '<span class="ev-dot ev-dot-green">●</span>'
    return '<span class="ev-dot ev-dot-amber">○</span>'


def _error_badge(error: str) -> str:
    """The error type as a coloured badge (fp red, near-miss amber, tp green)."""
    tone = {"tp": "green", "fp": "red", "fn": "red", "near_miss": "amber", "ok": "muted"}.get(
        error, "muted"
    )
    return f'<span class="ev-badge ev-badge-{tone}">{_esc(error)}</span>'


def _direction_badge(direction: str) -> str:
    """The compare direction as a coloured badge."""
    tone = {"improved": "improved", "worsened": "worsened", "same": "same"}.get(direction, "same")
    return f'<span class="ev-badge ev-badge-{tone}">{_esc(direction)}</span>'


def _chip(
    value: float | int | None,
    maximum: float,
    *,
    signed: bool = False,
    tone: str = "auto",
    digits: int = 3,
) -> str:
    """A number as a coloured chip; the tint is the value's size on its scale."""
    if value is None or pd.isna(value):
        return '<span class="ev-muted">—</span>'
    number = float(value)
    ratio = min(abs(number) / maximum, 1.0) if maximum > 0.0 else 0.0
    css_tone = ("pos" if number >= 0.0 else "neg") if tone == "auto" else tone
    text = f"{number:+.{digits}f}" if signed and number else f"{number:.{digits}f}"
    alpha = 0.10 + 0.45 * ratio
    return (
        f'<span class="ev-chip ev-chip-{css_tone}" style="--alpha: {alpha:.3f}">{_esc(text)}</span>'
    )


def _meter(
    value: float | int | None,
    maximum: float,
    *,
    signed: bool = False,
    tone: str = "auto",
    digits: int = 3,
) -> str:
    """A chip plus a small intensity bar (the value's share of the scale)."""
    if value is None or pd.isna(value):
        return '<span class="ev-muted">—</span>'
    number = float(value)
    ratio = min(abs(number) / maximum, 1.0) if maximum > 0.0 else 0.0
    bar_tone = (
        tone
        if tone in {"score", "neutral", "good", "warn", "danger"}
        else ("pos" if number >= 0.0 else "neg")
    )
    return (
        f'<span class="ev-meter">{_chip(number, maximum, signed=signed, tone=tone, digits=digits)}'
        f'<span class="ev-bar"><span class="ev-bar-fill ev-bar-{bar_tone}" '
        f'style="width: {ratio * 100:.1f}%"></span></span></span>'
    )


def _word_pills(entries: list[tuple[str, float]], maximum: float) -> str:
    """The contributing words as pills, each with its weight as an inline bar."""
    if not entries:
        return '<span class="ev-muted">no contributing word</span>'
    pills = []
    for word, z_score in entries:
        ratio = min(abs(z_score) / maximum, 1.0) if maximum > 0.0 else 0.0
        pills.append(
            f'<span class="ev-pill">'
            f'<span class="ev-pill-fill" style="width: {ratio * 100:.1f}%"></span>'
            f'<span class="ev-pill-text">{_esc(word)} <b>{z_score:+.3f}</b></span></span>'
        )
    return '<span class="ev-pills">' + "".join(pills) + "</span>"


def _name_pills(names: Sequence[str]) -> str:
    """A list of names as plain pills (the coverage holes, the gold labels)."""
    if not names:
        return '<span class="ev-muted">—</span>'
    return (
        '<span class="ev-pills">'
        + "".join(
            f'<span class="ev-pill"><span class="ev-pill-text">{_esc(name)}</span></span>'
            for name in names
        )
        + "</span>"
    )


def _html_table(
    headers: list[str],
    rows: list[list[str]],
    aligns: list[str] | None = None,
    *,
    columns: list[str] | None = None,
) -> str:
    """A read-only table: zebra rows, a sticky header and escaped headers.

    The cells are HTML fragments (built by the helpers above); the header
    labels and the ``data-columns`` hook are escaped. The hook is what the
    tests pin the column order with.
    """
    align = list(aligns or ["left"] * len(headers))
    keys = list(columns or headers)
    head = "".join(
        f'<th style="text-align:{a}">{_esc(h)}</th>' for h, a in zip(headers, align, strict=True)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td style="text-align:{a}">{cell}</td>' for cell, a in zip(row, align, strict=True)
        )
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body.append(f'<tr><td class="ev-empty" colspan="{len(headers)}">no rows</td></tr>')
    return (
        f'<div class="ev-table-wrap" data-columns="{_esc("|".join(keys))}">'
        f'<table class="ev-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


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
    top_words: dict[tuple[str, str], list[tuple[str, float]]],
) -> None:
    """The mistakes, filterable and sorted by how confident they were."""
    cells = _cells_frame(tables)
    lookup = _text_lookup(context)
    texts = _texts_frame(tables)[["text_id", "length_bucket", "text"]]
    cells = cells.merge(texts, on="text_id", how="left")
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
    records = frame.head(limit).to_dict("records")
    max_score = max((float(record["score"]) for record in records), default=1.0)
    max_gap = max((abs(float(record["gap_to_cut"])) for record in records), default=1.0)
    rows = []
    for record in records:
        entries = top_words.get((record["text_id"], record["label"]), [])
        max_word = max((z for _, z in entries), default=1.0)
        rows.append(
            [
                _expandable_text(lookup.get(record["text_id"], record["text"])),
                _error_badge(str(record["error"])),
                _esc(record["label"]),
                _meter(float(record["score"]), max_score, tone="score"),
                f'<span class="ev-rank">{int(record["rank"])}</span>',
                _meter(float(record["gap_to_cut"]), max_gap, signed=True),
                _gold_dot(bool(record["gold"])),
                _predicted_dot(bool(record["predicted"])),
                _evidence_dot(bool(record["has_evidence"])),
                f'<span class="ev-muted">{_esc(record["length_bucket"])}</span>',
                _word_pills(entries, max_word),
                f'<span class="ev-id">{_esc(record["text_id"])}</span>',
            ]
        )
    st.markdown(
        _html_table(
            [
                "text",
                "error",
                "label",
                "score",
                "rank",
                "gap to cut",
                "gold",
                "pred",
                "evidence",
                "length",
                "top words",
                "text id",
            ],
            rows,
            [
                "left",
                "center",
                "left",
                "right",
                "right",
                "right",
                "center",
                "center",
                "center",
                "left",
                "left",
                "left",
            ],
            columns=[
                "text",
                "error",
                "label",
                "score",
                "rank",
                "gap_to_cut",
                "gold",
                "predicted",
                "has_evidence",
                "length_bucket",
                "top_words",
                "text_id",
            ],
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Hover a text to expand it in place; scroll inside the table to keep the header pinned. "
        "★ gold · ● predicted · evidence: ● a word gave points, ○ empty slot."
    )
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
    max_score = max((row.score for row in rows), default=1.0)
    max_gap = max((abs(row.gap_to_cut) for row in rows), default=1.0)
    table_rows = []
    for row in rows:
        label = _esc(row.label)
        if not row.has_table:
            label += ' <span class="ev-badge ev-badge-muted">no table</span>'
        table_rows.append(
            [
                f'<span class="ev-rank">{row.rank}</span>',
                label,
                _meter(row.score, max_score, tone="score"),
                _gold_dot(row.gold),
                _predicted_dot(row.predicted),
                _error_badge(row.error),
                _evidence_dot(row.has_evidence),
                _meter(row.gap_to_cut, max_gap, signed=True),
            ]
        )
    st.markdown(
        _html_table(
            ["rank", "label", "score", "gold", "pred", "error", "evidence", "gap to cut"],
            table_rows,
            ["right", "left", "right", "center", "center", "center", "center", "right"],
            columns=[
                "rank",
                "label",
                "score",
                "gold",
                "predicted",
                "error",
                "has_evidence",
                "gap_to_cut",
            ],
        ),
        unsafe_allow_html=True,
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
        top = ranking[:15]
        max_evidence = max((row.score for row in top), default=1.0)
        evidence_rows = []
        for row in top:
            entries = list(row.words[:2])
            evidence_rows.append(
                [
                    f'<span class="ev-rank">{row.rank}</span>',
                    _esc(row.label),
                    _meter(row.score, max_evidence, tone="score"),
                    _word_pills(entries, max((z for _, z in entries), default=1.0)),
                ]
            )
        st.markdown(
            _html_table(
                ["rank", "label", "score", "words"],
                evidence_rows,
                ["right", "left", "right", "left"],
                columns=["rank", "label", "score", "words"],
            ),
            unsafe_allow_html=True,
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
        top = confusion[:40]
        confusion_rows = [
            [
                _esc(row.gold_label),
                _esc(row.predicted_label),
                f'<span class="ev-rank">{row.n}</span>',
                f'<span class="ev-muted">{row.support}</span>',
                _meter(row.share, 1.0, tone="neutral"),
            ]
            for row in top
        ]
        st.markdown(
            _html_table(
                ["gold", "predicted", "n", "support", "share of A"],
                confusion_rows,
                ["left", "left", "right", "right", "right"],
                columns=["gold_label", "predicted_label", "n", "support", "share"],
            ),
            unsafe_allow_html=True,
        )
    with tabs[1]:
        st.caption(
            "Per label: precision, recall, the miss rate (fn + near_miss over gold) and the "
            "FP rate (fp over the non-gold cells)."
        )
        ordered = sorted(rates, key=lambda row: (-row.miss_rate, -row.fp_rate, row.label))
        rate_rows = []
        for row in ordered:
            rate_rows.append(
                [
                    _esc(row.label),
                    f'<span class="ev-muted">{_esc(row.bucket)}</span>',
                    str(row.n_gold),
                    str(row.n_pred),
                    str(row.tp),
                    str(row.fp),
                    str(row.fn),
                    str(row.near_miss),
                    _chip(row.precision, 1.0, tone="good"),
                    _chip(row.recall, 1.0, tone="good"),
                    _meter(row.miss_rate, 1.0, tone="danger"),
                    _meter(row.fp_rate, 1.0, tone="danger"),
                ]
            )
        st.markdown(
            _html_table(
                [
                    "label",
                    "bucket",
                    "gold",
                    "pred",
                    "tp",
                    "fp",
                    "fn",
                    "near",
                    "precision",
                    "recall",
                    "miss rate",
                    "fp rate",
                ],
                rate_rows,
                [
                    "left",
                    "left",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                    "right",
                ],
                columns=[
                    "label",
                    "bucket",
                    "n_gold",
                    "n_pred",
                    "tp",
                    "fp",
                    "fn",
                    "near_miss",
                    "precision",
                    "recall",
                    "miss_rate",
                    "fp_rate",
                ],
            ),
            unsafe_allow_html=True,
        )
    with tabs[2]:
        st.caption(
            "The words whose contributions sit in predicted cells: n_cells counts the "
            "predicted cells a word contributed to, split into hits and false positives."
        )
        top = words[:50]
        max_z = max((abs(row.mean_z) for row in top), default=1.0)
        max_cells = max((row.n_cells for row in top), default=1)
        word_rows = [
            [
                _word_pills([(row.word, row.mean_z)], max_z),
                _chip(row.n_cells, max_cells, tone="neutral", digits=0),
                str(row.n_tp),
                _chip(row.n_fp, max_cells, tone="danger", digits=0),
                _chip(row.mean_z, max_z, signed=True),
                _meter(row.precision, 1.0, tone="good"),
            ]
            for row in top
        ]
        st.markdown(
            _html_table(
                ["word", "cells", "hits", "FPs", "mean z", "precision"],
                word_rows,
                ["left", "right", "right", "right", "right", "right"],
                columns=["word", "n_cells", "n_tp", "n_fp", "mean_z", "precision"],
            ),
            unsafe_allow_html=True,
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
        st.markdown("**Never predicted:**", unsafe_allow_html=True)
        st.markdown(_name_pills(holes.never_predicted), unsafe_allow_html=True)
        st.markdown("**Never predicted with evidence:**", unsafe_allow_html=True)
        st.markdown(_name_pills(holes.never_predicted_with_evidence), unsafe_allow_html=True)
        st.markdown("**No train table (held out):**", unsafe_allow_html=True)
        st.markdown(_name_pills(holes.no_table), unsafe_allow_html=True)
    with tabs[4]:
        bucket_rows = [
            [
                f'<span class="ev-muted">{_esc(row.kind)}</span>',
                _esc(row.bucket),
                str(row.n_cells),
                str(row.tp),
                str(row.fp),
                str(row.fn),
                str(row.near_miss),
                _meter(row.micro_f1, 1.0, tone="good"),
            ]
            for row in buckets
        ]
        st.markdown(
            _html_table(
                ["kind", "bucket", "cells", "tp", "fp", "fn", "near", "micro-F1"],
                bucket_rows,
                ["left", "left", "right", "right", "right", "right", "right", "right"],
                columns=[
                    "kind",
                    "bucket",
                    "n_cells",
                    "tp",
                    "fp",
                    "fn",
                    "near_miss",
                    "micro_f1",
                ],
            ),
            unsafe_allow_html=True,
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
    records = shown.to_dict("records")
    max_delta = max((abs(float(record["delta_f1"])) for record in records), default=1.0)
    max_count = max(
        (
            abs(int(record[field]))
            for record in records
            for field in ("delta_tp", "delta_fp", "delta_fn")
        ),
        default=1,
    )
    table_rows = []
    for record in records:
        table_rows.append(
            [
                _expandable_text(lookup.get(record["text_id"], "")),
                _meter(float(record["delta_f1"]), max_delta, signed=True),
                _chip(float(record["f1_a"]), 1.0, tone="neutral"),
                _chip(float(record["f1_b"]), 1.0, tone="neutral"),
                _direction_badge(str(record["direction"])),
                _chip(int(record["delta_tp"]), max_count, signed=True, digits=0),
                _chip(int(record["delta_fp"]), max_count, signed=True, digits=0),
                _chip(int(record["delta_fn"]), max_count, signed=True, digits=0),
                f'<span class="ev-id">{_esc(record["text_id"])}</span>',
            ]
        )
    st.markdown(
        _html_table(
            ["text", "ΔF1", "F1 A", "F1 B", "direction", "Δtp", "Δfp", "Δfn", "text id"],
            table_rows,
            ["left", "right", "right", "right", "center", "right", "right", "right", "left"],
            columns=[
                "text",
                "delta_f1",
                "f1_a",
                "f1_b",
                "direction",
                "delta_tp",
                "delta_fp",
                "delta_fn",
                "text_id",
            ],
        ),
        unsafe_allow_html=True,
    )
    dictionary_side = [item for item in (a, b) if item.arm == "dictionary (recomputed)"]
    if dictionary_side and not shown.empty:
        text_id = st.selectbox(
            "label detail for one changed text",
            list(shown["text_id"]),
            key="cmp_text",
        )
        cells = [row for row in tables.cells if row.text_id == text_id]
        detail = [
            row for row in sorted(cells, key=lambda item: item.rank) if row.gold or row.predicted
        ]
        max_cell_score = max((row.score for row in detail), default=1.0)
        detail_rows = [
            [
                f'<span class="ev-rank">{row.rank}</span>',
                _esc(row.label),
                _gold_dot(row.gold),
                _predicted_dot(row.predicted),
                _meter(row.score, max_cell_score, tone="score"),
                _error_badge(row.error),
                _evidence_dot(row.has_evidence),
            ]
            for row in detail
        ]
        st.markdown(
            _html_table(
                ["rank", "label", "gold", "pred", "score", "error", "evidence"],
                detail_rows,
                ["right", "left", "center", "center", "right", "center", "center"],
                columns=["rank", "label", "gold", "predicted", "score", "error", "has_evidence"],
            ),
            unsafe_allow_html=True,
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


def _text_select(context: ev.Context, key: str) -> str:
    """A searchable text picker, shared by the one-text views."""
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
        key=key,
    )
    st.session_state["selected_text"] = picked
    return picked


def _hitmap_columns(
    context: ev.Context,
    tables: ev.Tables,
    record: Any,
    extra: int,
) -> list[tuple[str, str, float, int, bool]]:
    """The hitmap's columns: gold, false positives, then the strongest alternatives.

    The false positives are every predicted non-gold label (all of them, not a
    sample); the ``extra`` slider adds the strongest non-gold labels that were
    **not** predicted, so a text with no false positive still has a comparison.
    Empty tables are left out of the alternatives (they carry no evidence).
    """
    text_id = str(record.id)
    cells = {row.label: row for row in tables.cells if row.text_id == text_id}
    gold = list(record.labels)
    gold_set = set(gold)
    labels = list(context.labels)
    false_positives = [
        label for label in labels if cells[label].predicted and label not in gold_set
    ]
    false_positives.sort(key=lambda label: (-cells[label].score, labels.index(label)))
    alternatives = [
        label
        for label in labels
        if label not in gold_set and label not in false_positives and cells[label].has_table
    ]
    alternatives.sort(key=lambda label: (-cells[label].score, labels.index(label)))
    columns: list[tuple[str, str, float, int, bool]] = []
    for label in [*gold, *false_positives, *alternatives[:extra]]:
        kind = "gold" if label in gold_set else ("fp" if label in false_positives else "alt")
        cell = cells[label]
        columns.append((label, kind, cell.score, cell.rank, cell.has_table))
    return columns


def _hitmap_header(
    label: str,
    kind: str,
    score: float,
    rank: int,
    has_table: bool,
) -> str:
    """One hitmap column header: the kind tag, the label and its score/rank."""
    tag = {"gold": "★ gold", "fp": "fp", "alt": "alt"}[kind]
    tag_class = {"gold": "ev-h-gold", "fp": "ev-h-fp", "alt": "ev-h-alt"}[kind]
    meta = "no table" if not has_table else f"score {score:.3f} · rank {rank}"
    return (
        '<th style="text-align:left">'
        f'<span class="ev-h-label"><span class="ev-h-tag {tag_class}">{tag}</span>'
        f"{_esc(label)}</span>"
        f'<span class="ev-h-meta">{_esc(meta)}</span></th>'
    )


def _z_cell(value: float | None, maximum: float) -> str:
    """One matrix cell: a tinted z-score; ``None`` is the map's silence."""
    if value is None:
        return '<span class="ev-z ev-z-none">·</span>'
    if value == 0.0:
        return '<span class="ev-z ev-z-zero">0.000</span>'
    ratio = min(abs(value) / maximum, 1.0) if maximum > 0.0 else 0.0
    tone = "pos" if value > 0.0 else "neg"
    return (
        f'<span class="ev-z ev-z-{tone}" style="--alpha: {0.08 + 0.50 * ratio:.3f}">'
        f"{value:+.3f}</span>"
    )


def _hitmap_table(
    columns: list[tuple[str, str, float, int, bool]],
    rows: Sequence[ev.HitmapRow],
) -> str:
    """The word x label matrix, with the header pinned by the shared CSS."""
    maximum = max(
        (abs(z) for row in rows for z in row.z_scores if z is not None),
        default=1.0,
    )
    header = [
        '<th style="text-align:left">word</th>',
        '<th style="text-align:right">n</th>',
        *[_hitmap_header(*column) for column in columns],
    ]
    body = []
    for row in rows:
        cells = [
            f'<td style="text-align:left"><span class="ev-word">{_esc(row.word)}</span></td>',
            f'<td style="text-align:right"><span class="ev-muted">{row.n_occurrences}</span></td>',
            *[f'<td style="text-align:right">{_z_cell(z, maximum)}</td>' for z in row.z_scores],
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(
            f'<tr><td class="ev-empty" colspan="{2 + len(columns)}">no content words</td></tr>'
        )
    keys = "|".join(["word", "n", *(column[0] for column in columns)])
    return (
        f'<div class="ev-table-wrap" data-columns="{_esc(keys)}">'
        f'<table class="ev-table ev-hitmap"><thead><tr>{"".join(header)}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def hitmap_compare_view(tables: ev.Tables, context: ev.Context) -> None:
    """The z-score matrix: this text's words against gold, FP and alternative labels."""
    st.subheader("Compare hitmaps — the z-scores behind this text, label by label")
    picked = _text_select(context, "hitmap_text_pick")
    record = next(record for record in context.records if str(record.id) == picked)
    text_row = next(row for row in tables.texts if row.text_id == picked)
    controls = st.columns(2)
    extra = controls[0].slider(
        "extra labels (k)",
        0,
        8,
        3,
        key="hitmap_extra",
        help=(
            "Gold labels and all of the classifier's false positives are always shown; "
            "this adds the k strongest non-gold labels that were not predicted (alt)."
        ),
    )
    sort_signal = controls[1].checkbox(
        "sort words by strongest |z|",
        value=False,
        key="hitmap_sort",
        help="Off = the text's own word order; on = the strongest word first.",
    )
    columns = _hitmap_columns(context, tables, record, int(extra))
    labels = [column[0] for column in columns]
    rows = ev.hitmap_rows(context.tables, record.text, labels)
    if sort_signal:
        rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    -max((abs(z) for z in row.z_scores if z is not None), default=0.0),
                    row.word,
                ),
            )
        )
    st.write(record.text)
    st.caption(
        f"gold: {' · '.join(record.labels)} — the classifier's pick: "
        f"{' · '.join(text_row.predicted_labels) or '(none)'}"
    )
    st.caption(
        "Rows are this text's content words; columns are the gold labels (★ gold), the "
        "classifier's false positives (fp), and the k strongest alternatives (alt). "
        "A cell is the stored corpus-map z-score: green pulls the label up, red pushes it "
        "down; · means the word is not in that label's table (it contributes 0)."
    )
    st.markdown(_hitmap_table(columns, rows), unsafe_allow_html=True)
    n_alt = sum(1 for column in columns if column[1] == "alt")
    st.caption(f"{len(rows)} content words × {len(columns)} labels ({n_alt} alternatives).")


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
    lookup = _text_lookup(context)
    rows: list[object]
    if table_name.startswith("errors"):
        error_types = st.multiselect(
            "error type",
            ["fp", "near_miss", "fn"],
            default=["fp", "near_miss", "fn"],
            key="export_errors",
        )
        rows = [
            {**asdict(row), "text": lookup.get(row.text_id, "")}
            for row in tables.cells
            if row.error in error_types
        ]
        fields = (
            "text",
            "label",
            "gold",
            "predicted",
            "score",
            "rank",
            "error",
            "has_evidence",
            "gap_to_cut",
            "text_id",
        )
    elif table_name == "texts":
        rows = list(tables.texts)
        fields = (
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
            "text_id",
        )
    elif table_name == "cells (all)":
        rows = [{**asdict(row), "text": lookup.get(row.text_id, "")} for row in tables.cells]
        fields = (
            "text",
            "label",
            "gold",
            "predicted",
            "score",
            "rank",
            "error",
            "has_evidence",
            "gap_to_cut",
            "text_id",
        )
    elif table_name == "words":
        rows = [{**asdict(row), "text": lookup.get(row.text_id, "")} for row in tables.words]
        fields = (
            "text",
            "label",
            "word",
            "z_score",
            "log_odds",
            "n_with",
            "df",
            "word_class",
            "is_name_word",
            "surrogate_coefficient",
            "text_id",
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
        rows = [{**asdict(row), "text": lookup.get(row.text_id, "")} for row in rows]
        fields = (
            "text",
            "delta_f1",
            "f1_a",
            "f1_b",
            "delta_tp",
            "delta_fp",
            "delta_fn",
            "improved",
            "worsened",
            "text_id",
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
    st.set_page_config(page_title="Error viewer — UPR val", page_icon="🪻", layout="wide")
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
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
    elif view == "Compare hitmaps":
        hitmap_compare_view(tables, context)
    else:
        export_view(tables, context, report, catalog)


if __name__ == "__main__":
    main()
