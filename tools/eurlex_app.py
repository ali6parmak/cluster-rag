"""The EURLEX-4K result viewer's **Streamlit app**.

Four meeting-ready views over the lesson's existing artifacts:

1. **One document** — a dev document's gold labels, its lexical and embedding
   neighbours, the five methods side by side, the label x source matrix and a
   ``why this label?`` panel.
2. **Methods** — one table read from the data layer's explicit registry, a bar
   chart per metric and the label-sparse curve.
3. **Labels** — the label-text embedding map (UMAP, PCA fallback) with a seed's
   unseen labels highlighted, and one label's parents/definition/dev documents.
4. **Errors** — the TC-17 pack's misses, per document and per bucket.

The app is the **only** file that imports Streamlit. Every number is the data
layer's (``tools/eurlex_view.py``): the metrics are **read** from
``results/eurlex-*/metrics.json``; the per-document scores are **computed on
demand** from the documented caches with the track's own rule, and the app says
so in every page's provenance expander. The guard runs before any view: a
failed check blocks the page, so stale numbers are never shown beside fresh
ones.

Run it from the repository root::

    .venv/bin/streamlit run tools/eurlex_app.py

The sources can be overridden in the sidebar or with the ``EURLEX_VIEW_*``
environment variables.
"""

from __future__ import annotations

import html as html_lib
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

TOOLS = Path(__file__).resolve().parent
CSS_PATH = TOOLS / "eurlex_app.css"
sys.path.insert(0, str(TOOLS))
import eurlex_view as ev  # noqa: E402

PAGES = ("One document", "Methods", "Labels", "Errors")

REGIME_CHOICES = ("45,000 train documents", "240-document seed")


def _env(name: str, default: str) -> str:
    """One source path: the environment variable, else the default."""
    return os.environ.get(name, default)


# --- cached reads -----------------------------------------------------------


@st.cache_resource(show_spinner="reading the EURLEX-4K artifacts and the label space…")
def load_artifacts(
    results: str,
    cache: str,
    data_dir: str,
    catalog: str,
    ranking_path: str,
    model_id: str,
) -> tuple[ev.Sources, ev.Dataset, ev.Ranking, tuple[ev.MetricRow, ...]]:
    """Load the dataset, the TC-17 ranking and every registered metric row."""
    sources = ev.resolve_sources(
        results=results,
        cache=cache,
        data_dir=data_dir,
        catalog_path=catalog,
        ranking_path=ranking_path,
        model_id=model_id,
    )
    dataset = ev.load_dataset(sources)
    ranking = ev.load_ranking(sources.ranking_path)
    rows = ev.load_metrics(sources)
    return sources, dataset, ranking, rows


@st.cache_resource(show_spinner="loading the cached vectors and fitting TF-IDF…")
def load_search_index(
    results: str,
    cache: str,
    data_dir: str,
    catalog: str,
    ranking_path: str,
    model_id: str,
    seed_key: int,
) -> ev.SearchIndex:
    """Build the retrieval index for the 45,000-document or the 240-document regime."""
    sources, dataset, _, _ = load_artifacts(
        results, cache, data_dir, catalog, ranking_path, model_id
    )
    seed = None if seed_key == 0 else int(seed_key)
    if seed is None:
        rows = tuple(range(len(dataset.train)))
    else:
        rows = tuple(int(row) for row in ev.seed_sample(seed, n_sources=len(dataset.train)))
    docs = tuple(dataset.train[row] for row in rows)
    vectors = ev.load_document_vectors(sources, docs)
    lexical = ev.build_lexical_index(docs)
    return ev.build_search_index(
        dataset,
        sources,
        seed=seed,
        vectors=vectors,
        lexical=lexical,
    )


@st.cache_resource(show_spinner="reading the cached label vectors…")
def load_label_cache(
    results: str,
    cache: str,
    data_dir: str,
    catalog: str,
    ranking_path: str,
    model_id: str,
    variant: str,
) -> Any:
    """Load one label-text cache (title or title+parents)."""
    sources, dataset, _, _ = load_artifacts(
        results, cache, data_dir, catalog, ranking_path, model_id
    )
    return ev.load_label_vectors(sources, dataset, variant)


@st.cache_data(show_spinner="computing the neighbours and the five method rankings…")
def compute_view(
    results: str,
    cache: str,
    data_dir: str,
    catalog: str,
    ranking_path: str,
    model_id: str,
    seed_key: int,
    document_row: int,
) -> ev.DocumentView:
    """Compute one document's full view (cached by its source and row)."""
    sources, dataset, ranking, _ = load_artifacts(
        results, cache, data_dir, catalog, ranking_path, model_id
    )
    index = load_search_index(results, cache, data_dir, catalog, ranking_path, model_id, seed_key)
    query_vector = ev.load_document_vectors(sources, (dataset.dev[document_row],))[0]
    title_vectors = load_label_cache(
        results, cache, data_dir, catalog, ranking_path, model_id, "title"
    )
    parent_vectors = load_label_cache(
        results, cache, data_dir, catalog, ranking_path, model_id, "title_parents"
    )
    return ev.build_document_view(
        dataset,
        ranking,
        index,
        document_row=document_row,
        query_vector=query_vector,
        label_vectors_title=title_vectors,
        label_vectors_parents=parent_vectors,
    )


@st.cache_data(show_spinner="projecting the 4,197 label vectors to two dimensions…")
def compute_label_map(
    results: str,
    cache: str,
    data_dir: str,
    catalog: str,
    ranking_path: str,
    model_id: str,
    variant: str,
) -> ev.LabelMap:
    """Project the label vectors (UMAP when available, PCA fallback)."""
    vectors = load_label_cache(results, cache, data_dir, catalog, ranking_path, model_id, variant)
    return ev.label_map(vectors)


# --- HTML helpers -----------------------------------------------------------


def _esc(value: object) -> str:
    """The value as escaped HTML text."""
    return html_lib.escape(str(value), quote=True)


def _snippet(text: str, limit: int = 90) -> str:
    """A one-line snippet of a text."""
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _truncate(text: str, limit: int) -> str:
    """A text truncated to ``limit`` characters with an ellipsis."""
    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _fmt(value: float | None, digits: int = 4) -> str:
    """A number, or an em dash when the artifact does not carry it."""
    if value is None:
        return '<span class="ev-muted">—</span>'
    return _esc(f"{value:.{digits}f}")


def _badge(text: str, tone: str) -> str:
    """A coloured badge."""
    return f'<span class="ev-badge ev-badge-{tone}">{_esc(text)}</span>'


def _bucket_badge(bucket: str) -> str:
    """The support bucket as a coloured badge (unseen is red)."""
    tone = {"head": "head", "torso": "torso", "tail": "tail", "unseen": "unseen"}.get(
        bucket, "muted"
    )
    text = "UNSEEN" if bucket == "unseen" else bucket
    return _badge(text, tone)


def _dot(active: bool, tone: str = "green", *, on: str = "●", off: str = "·") -> str:
    """A boolean as a coloured dot (read-only, never a checkbox)."""
    css = f"ev-dot ev-dot-{tone}" if active else "ev-dot ev-dot-off"
    return f'<span class="{css}">{on if active else off}</span>'


def _gold_mark(active: bool) -> str:
    """The gold flag: a gold star or a dim dot."""
    return _dot(active, "gold", on="★")


def _table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
    compact: bool = False,
    empty: str = "no rows",
) -> str:
    """A read-only, escaped table (the cells are trusted HTML fragments)."""
    body_rows = list(rows)
    align = list(aligns or ["left"] * len(headers))
    head = "".join(
        f'<th style="text-align:{align[index]}">{_esc(header)}</th>'
        for index, header in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="text-align:{align[index]}">{cell}</td>' for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in body_rows
    )
    if not body:
        body = f'<tr><td class="ev-empty" colspan="{len(headers)}">{_esc(empty)}</td></tr>'
    css = "ev-table compact" if compact else "ev-table"
    return (
        f'<div class="ev-table-wrap"><table class="{css}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _label_pill(title: str, label: str, bucket: str) -> str:
    """A gold-label pill with its bucket/unseen tag."""
    css = "ev-pill ev-pill-unseen" if bucket == "unseen" else "ev-pill ev-pill-gold"
    tag = "UNSEEN" if bucket == "unseen" else bucket
    return (
        f'<span class="{css}" title="{_esc(label)}">'
        f'{_esc(title)} <span class="ev-sub">· {_esc(tag)}</span></span>'
    )


def _label_cell(label: str, title: str | None = None) -> str:
    """The label's title with its id underneath."""
    shown = title or label
    return f'<span class="ev-label">{_esc(shown)}</span><span class="ev-id"> · {_esc(label)}</span>'


def _provenance(report: ev.GuardReport) -> None:
    """The guard and the provenance, in one collapsed expander."""
    with st.expander("Provenance, the guard, and what is computed vs read"):
        st.markdown(
            "**Read from artifacts:** every Methods row and every metric on this "
            "page (the run's own `metrics.json`). "
            "**Computed on demand:** the per-document neighbour cosines, the "
            "lexical/embedding k-NN vote scores, the label-text cosines, the "
            "label x source marks and the miss tables — from the documented "
            "caches, with the track's own `full_text`, TF-IDF config and k-NN "
            "rule (k=100, power=1.0). They are **views, not new results**, and "
            "may differ in the last decimal from the runs' means."
        )
        st.code("\n".join(report.provenance), language="text")
        st.code("\n".join(report.lines()), language="text")


# --- page 1: one document ---------------------------------------------------

SECTION_LABELS = {
    "title": "title",
    "header": "header",
    "recitals": "recitals",
    "main_body": "main body",
}


def _inline_sections(sections: Sequence[ev.DocumentSection]) -> str:
    """The four full-text parts as coloured inline spans (one clipped line)."""
    parts: list[str] = []
    for section in sections:
        text = " ".join(section.text.split())
        if not text:
            continue
        if parts:
            parts.append('<span class="ex-part-sep"> · </span>')
        parts.append(f'<span class="ex-part ex-part-{_esc(section.kind)}">{_esc(text)}</span>')
    return "".join(parts) or '<span class="ev-muted">—</span>'


def _section_legend() -> str:
    """The colour key of the full-text parts."""
    return (
        '<div class="ex-legend">'
        + "".join(
            f'<span class="ex-legend-chip ex-part-{_esc(kind)}">{_esc(label)}</span>'
            for kind, label in SECTION_LABELS.items()
        )
        + "</div>"
    )


def _document_text_panel(sections: Sequence[ev.DocumentSection]) -> str:
    """The full document text, one coloured block per field."""
    blocks: list[str] = []
    for section in sections:
        text = section.text.strip()
        label = SECTION_LABELS.get(section.kind, section.kind)
        body = (
            f'<div class="ex-part-body">{_esc(text)}</div>'
            if text
            else '<div class="ex-part-body ev-muted">—</div>'
        )
        blocks.append(
            f'<div class="ex-part-block ex-part-{_esc(section.kind)}">'
            f'<div class="ex-part-label">{_esc(label)} · {len(text):,} chars</div>'
            f"{body}</div>"
        )
    return '<div class="ex-fulltext">' + "".join(blocks) + "</div>"


def _neighbour_table(rows: Sequence[ev.NeighbourRow], dataset: ev.Dataset) -> str:
    """One neighbour list as a table (shared labels are green).

    The title and the document text use the error viewer's ``ev-text`` rule:
    one clipped line, and the full text expands **in place** on hover. The text
    is the train document's ``full_text`` (title, header, recitals, body),
    colour-coded by part.
    """
    body = []
    for row in rows:
        labels = (
            " ".join(
                f'<span class="ev-pill{" ev-pill-gold" if label in row.shared else ""}" '
                f'title="{_esc(label)}">'
                f"{_esc(ev.label_title(dataset, label))}</span>"
                for label in row.labels
            )
            or '<span class="ev-muted">—</span>'
        )
        shared = (
            " ".join(
                f'<span class="ev-pill ev-pill-gold" title="{_esc(label)}">'
                f"{_esc(ev.label_title(dataset, label))}</span>"
                for label in row.shared
            )
            or '<span class="ev-muted">—</span>'
        )
        title_cell = (
            f'<span class="ev-text ex-text-title">{_esc(row.title.strip())}</span>'
            f'<span class="ev-id"> · {_esc(row.doc_id)}</span>'
        )
        text_cell = f'<span class="ev-text">{_inline_sections(row.sections)}</span>'
        body.append(
            [
                f'<span class="ev-rank">{row.rank}</span>',
                title_cell,
                text_cell,
                f'<span class="ev-chip ev-chip-score">{row.score:.3f}</span>',
                f'<span class="ev-pills">{labels}</span>',
                f'<span class="ev-pills">{shared}</span>',
            ]
        )
    return _table(
        (
            "#",
            "train document",
            "document text (hover to expand)",
            "cosine",
            "its gold labels",
            "shared with the document",
        ),
        body,
        aligns=("right", "left", "left", "right", "left", "left"),
        compact=True,
    )


def _method_table(method: ev.MethodRanking) -> str:
    """One method's top-10 as a compact table."""
    body = []
    for row in method.rows:
        score = "—" if row.score is None else f"{row.score:.3f}"
        body.append(
            [
                f'<span class="ev-rank">{row.rank}</span>',
                _label_cell(row.label, row.title),
                f'<span class="ev-chip ev-chip-score">{_esc(score)}</span>',
                _gold_mark(row.gold),
                _badge("UNSEEN", "unseen") if row.unseen else '<span class="ev-muted">·</span>',
                _badge("hit", "hit") if row.hit else _badge("miss", "miss"),
            ]
        )
    return _table(
        ("#", "label", "score", "gold", "unseen", "hit"),
        body,
        aligns=("right", "left", "right", "center", "center", "center"),
        compact=True,
    )


def _matrix_table(view: ev.DocumentView) -> str:
    """The label x source matrix (a check mark means the source holds the label)."""
    headers = ["label", "gold", *[name for _, name in ev.MATRIX_SOURCES]]
    body = []
    for row in view.matrix:
        marks = [
            '<span class="ex-check">✓</span>' if mark else '<span class="ex-cross">·</span>'
            for mark in row.marks
        ]
        body.append(
            [
                _label_cell(row.label, row.title),
                _gold_mark(row.gold),
                *marks,
            ]
        )
    return _table(
        headers,
        body,
        aligns=("left", "center", *["center"] * len(ev.MATRIX_SOURCES)),
        compact=True,
    )


def _why_panel(
    dataset: ev.Dataset,
    view: ev.DocumentView,
    index: ev.SearchIndex,
) -> None:
    """The signals behind one selected candidate label."""
    ordered: list[str] = []
    for tag in view.gold:
        ordered.append(tag.label)
    for method in view.methods:
        for row in method.rows:
            if row.label not in ordered:
                ordered.append(row.label)
    if not ordered:
        st.caption("no candidate labels")
        return
    titles = {label: ev.label_title(dataset, label) for label in ordered}
    selected = st.selectbox(
        "candidate label (gold labels first, then the five top-10s)",
        ordered,
        format_func=lambda label: f"{titles.get(label, label)} · {label}",
        key="why_label",
    )
    support = int(index.label_matrix[:, dataset.label_index[selected]].sum())
    signals = ev.label_signals(dataset, view, selected, index_support=support)
    rows: list[list[str]] = [
        ["pack rank", f"#{signals.pack_rank}", "the TC-17 `ranked_dev.npz`"],
        [
            "gold",
            _gold_mark(signals.gold),
            "the dev document's labels",
        ],
        [
            "full-train support",
            f"{signals.support} · {_bucket_badge(signals.bucket)}",
            "train document frequency (45,000)",
        ],
    ]
    if view.index_regime == ev.REGIME_240:
        rows.append(
            [
                "240-document support",
                f"{signals.index_support} · {_badge(signals.index_bucket, 'muted')}",
                f"the seed {view.index_seed} sample",
            ]
        )
    for key, label in (
        ("lexical_knn", "lexical k-NN vote"),
        ("embedding_knn", "embedding k-NN vote"),
        ("label_text_title", "label text (title) cosine"),
        ("label_text_title_parents", "label text (title+parents) cosine"),
    ):
        rows.append(
            [
                label,
                f"{signals.scores[key]:.4f} · rank #{signals.ranks[key]}",
                "computed from the cache + the track's rule"
                if key.endswith("knn")
                else "cached MiniLM cosine",
            ]
        )
    rows.extend(
        [
            [
                "best TF-IDF neighbour cosine",
                f"{signals.best_cosine['lexical']:.4f}",
                "the best top-100 train neighbour carrying the label",
            ],
            [
                "best MiniLM neighbour cosine",
                f"{signals.best_cosine['embedding']:.4f}",
                "the best top-100 train neighbour carrying the label",
            ],
            [
                "parents",
                " · ".join(signals.parent_names) or "—",
                "the EUROVOC catalogue (immediate broader terms)",
            ],
            [
                "definition",
                signals.definition or "—",
                f"the catalogue ({signals.definition_kind})"
                if signals.definition_kind
                else "the catalogue carries none",
            ],
        ]
    )
    st.markdown(
        _table(("signal", "value", "source"), rows, aligns=("left", "left", "left"), compact=True),
        unsafe_allow_html=True,
    )


def one_document_page(
    sources: ev.Sources,
    dataset: ev.Dataset,
    ranking: ev.Ranking,
    report: ev.GuardReport,
    *,
    seed: int,
    source_args: tuple[str, str, str, str, str, str],
) -> None:
    """The document picker and the three panels."""
    st.markdown('<span class="ex-kicker">one document</span>', unsafe_allow_html=True)
    controls = st.columns([2, 1])
    with controls[0]:
        regime = st.radio(
            "retrieval index",
            REGIME_CHOICES,
            horizontal=True,
            key="regime",
            help=(
                "The 240-document regime is TC-21's: the labelled corpus is the "
                "seed's sample only. The 45,000-document regime is the track's "
                "standard train index."
            ),
        )
    with controls[1]:
        st.caption(
            f"TC-21 seed: **{seed}** (sidebar)"
            if regime == REGIME_CHOICES[1]
            else "full train index"
        )
    options = [f"{doc.celex_id} · {_snippet(doc.title, 100)}" for doc in dataset.dev]
    choice = st.selectbox(
        "search a dev document by CELEX id or title",
        options,
        index=0,
        key="document_choice",
    )
    document_row = options.index(choice)
    seed_key = 0 if regime == REGIME_CHOICES[0] else seed
    try:
        index = load_search_index(*source_args, seed_key)
        view = compute_view(*source_args, seed_key, document_row)
    except (ev.MissingVectorError, FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"the viewer could not compute this document: {exc}")
        st.stop()

    doc = view.doc
    st.markdown(
        f'<div class="ex-doc-title">{_esc(doc.title.strip())}</div>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="ex-doc-meta">{_esc(doc.celex_id)} · {_esc(doc.document_type)} · '
        f"{len(doc.main_body)} body paragraphs · {len(doc.text):,} characters</div>",
        unsafe_allow_html=True,
    )

    n_unseen = sum(1 for tag in view.gold if tag.unseen)
    pack_hits = view.methods[0].hits
    metrics = st.columns(4)
    metrics[0].metric("gold labels", len(view.gold))
    metrics[1].metric("unseen gold", n_unseen)
    metrics[2].metric("pack hits @10", f"{pack_hits}/{len(view.gold)}")
    metrics[3].metric("shared with neighbours", f"{view.overlap.n_shared}/{len(view.gold)}")
    st.markdown(
        '<div class="ev-pills">'
        + "".join(_label_pill(tag.title, tag.label, tag.bucket) for tag in view.gold)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ex-section-label">Document text</div>', unsafe_allow_html=True)
    st.markdown(_section_legend(), unsafe_allow_html=True)
    st.markdown(_document_text_panel(view.doc.sections), unsafe_allow_html=True)
    st.caption(
        "`full_text` = title + header + recitals + main body — the exact text "
        "the cache embedded and the retrievers score. The colours are only a "
        "reading aid."
    )

    st.subheader("The closest train documents")
    st.caption(
        "TF-IDF (word 1–2 grams) cosine over the chosen index, then the cached "
        "MiniLM whole-document cosine. Both read the same `full_text` the "
        "track's runs used. Hover a clipped document title or text to extend it "
        "in place."
    )
    st.markdown("**lexical TF-IDF neighbours**")
    st.markdown(_neighbour_table(view.lexical, dataset), unsafe_allow_html=True)
    st.markdown("**MiniLM embedding neighbours**")
    st.markdown(_neighbour_table(view.embedding, dataset), unsafe_allow_html=True)

    overlap = st.columns(3)
    overlap[0].metric("shared labels", view.overlap.n_shared)
    overlap[1].metric("only in the document", view.overlap.n_document_only)
    overlap[2].metric("only in the neighbours", view.overlap.n_neighbours_only)
    st.caption(
        "Shared: "
        + (" · ".join(view.overlap.shared) or "none")
        + " — document-only: "
        + (" · ".join(view.overlap.document_only) or "none")
        + " — neighbours-only: "
        + (_snippet(" · ".join(view.overlap.neighbours_only), 180) or "none")
    )

    st.subheader("The label × source matrix")
    st.caption(
        "Rows: every gold label plus every label any source's top-10 holds. "
        "A ✓ means the source's top-10 (or, for a neighbour column, at least one "
        "of the ten neighbour documents) carries the label. Gold labels are first."
    )
    st.markdown(_matrix_table(view), unsafe_allow_html=True)

    st.subheader("The methods, side by side")
    st.caption(
        "Gold ✓ marks a gold label; a hit is a gold label in the method's top-10. "
        "The TC-17 pack's npz carries ranks only, so its score column is `—`."
    )
    method_columns = st.columns(5)
    for column, method in zip(method_columns, view.methods, strict=True):
        with column:
            st.markdown(f"**{method.label}**")
            st.caption(f"{method.hits} hits · {method.unseen_hits} unseen")
            st.markdown(_method_table(method), unsafe_allow_html=True)

    st.subheader("Why this label?")
    st.caption(
        "Every signal below is either the artifact's (the pack rank), the "
        "dataset's (support, parents, definition) or computed from the cache "
        "with the track's rule. No score is invented."
    )
    _why_panel(dataset, view, index)
    _provenance(report)


# --- page 2: methods --------------------------------------------------------


METRIC_FIELDS = {
    "R@50": "recall50",
    "R-precision": "r_precision",
    "nDCG@10": "ndcg10",
    "unseen R@50": "unseen_recall50",
    "micro-F1": "micro_f1",
}


def methods_page(rows: Sequence[ev.MetricRow], report: ev.GuardReport) -> None:
    """The registry table, the per-metric bars and the label-sparse curve."""
    st.markdown('<span class="ex-kicker">methods</span>', unsafe_allow_html=True)
    st.subheader("Every registered run, read from its artifact")
    all_runs = sorted({row.run_id for row in rows})
    all_slices = sorted({row.slice for row in rows})
    filters = st.columns(4)
    with filters[0]:
        regimes = st.multiselect(
            "regime",
            [ev.REGIME_45K, ev.REGIME_240],
            default=[ev.REGIME_45K, ev.REGIME_240],
            format_func=lambda value: ev.REGIME_LABELS[value],
            key="m_regime",
        )
    with filters[1]:
        slices = st.multiselect(
            "slice",
            all_slices,
            default=list(ev.SLICE_NAMES),
            key="m_slice",
        )
    with filters[2]:
        runs = st.multiselect("run", all_runs, default=all_runs, key="m_run")
    with filters[3]:
        only_ranking = st.toggle(
            "only rows with R@50",
            value=False,
            key="m_rank_only",
            help="TC-12's decision arms carry micro-F1 but no R@50.",
        )
    filtered = [
        row
        for row in rows
        if row.regime in regimes
        and row.slice in slices
        and row.run_id in runs
        and (row.recall50 is not None or not only_ranking)
    ]
    st.caption(
        f"{len(filtered)} rows from {len({row.run_id for row in filtered})} runs. "
        "Every number is the artifact's; the table has no hand-written cell."
    )
    frame = [
        {
            "run": row.run_id,
            "method": row.method,
            "arm": row.arm,
            "regime": ev.REGIME_LABELS[row.regime],
            "slice": row.slice,
            "R@50": row.recall50,
            "R-precision": row.r_precision,
            "nDCG@10": row.ndcg10,
            "unseen R@50": row.unseen_recall50,
            "micro-F1": row.micro_f1,
            "cost (s)": row.cost_seconds,
            "artifact": row.artifact,
        }
        for row in filtered
    ]
    number_columns = ("R@50", "R-precision", "nDCG@10", "unseen R@50", "micro-F1")
    st.dataframe(
        frame,
        hide_index=True,
        height=420,
        column_config={
            **{name: st.column_config.NumberColumn(name, format="%.4f") for name in number_columns},
            "cost (s)": st.column_config.NumberColumn("cost (s)", format="%.1f"),
        },
    )

    st.subheader("One metric per method")
    selected_metric = st.selectbox("metric", list(METRIC_FIELDS), key="m_metric")
    field = METRIC_FIELDS[selected_metric]
    chart_rows = [
        {
            "method": f"{row.run_id} · {row.method}",
            "value": getattr(row, field),
            "slice": row.slice,
        }
        for row in filtered
        if getattr(row, field) is not None
    ]
    chart_rows.sort(key=lambda item: -float(item["value"]))
    if len(chart_rows) > 40:
        st.caption(
            f"Showing the top 40 of {len(chart_rows)} rows by {selected_metric}; "
            "narrow the filters to see the rest."
        )
        chart_rows = chart_rows[:40]
    if chart_rows:
        st.bar_chart(chart_rows, x="method", y="value", color="slice", height=430)
    else:
        st.info("No row in the current filter carries that metric.")

    st.subheader("The label-sparse curve — R@50 against the labelled-corpus size")
    curve = ev.label_sparse_curve(rows)
    curve_controls = st.columns(2)
    with curve_controls[0]:
        curve_slice = st.selectbox(
            "curve slice", list(ev.SLICE_NAMES), index=1, key="m_curve_slice"
        )
    with curve_controls[1]:
        curve_metric = st.radio(
            "curve metric",
            ["recall50", "unseen_recall50"],
            horizontal=True,
            format_func=lambda value: "R@50" if value == "recall50" else "unseen R@50",
            key="m_curve_metric",
        )
    points = [
        {
            "labelled documents": point.labelled_documents,
            "value": getattr(point, curve_metric),
            "family": point.family,
            "run": point.run_id,
            "method": point.method,
            "seed/run": "240-doc seed" if point.regime == ev.REGIME_240 else "45k run",
        }
        for point in curve
        if point.slice == curve_slice and getattr(point, curve_metric) is not None
    ]
    st.caption(
        "The 240-document points are the five TC-21 seeds (each pooled mean is one "
        "point); the 45,000-document points are full-train runs. They are not one "
        "continuous curve: the x-axis is a corpus-size regime, and the 45k pack "
        "variants were trained on the full train split."
    )
    if points:
        st.scatter_chart(
            points,
            x="labelled documents",
            y="value",
            color="family",
            height=380,
        )
    else:
        st.info("No curve point carries that metric on this slice.")
    st.caption(
        "`unseen R@50` is the micro recall@50 over the dev gold assignments whose "
        "label occurs in none of the 240 sampled documents (TC-21); the 45k runs "
        "carry their own unseen label class."
    )
    _provenance(report)


# --- page 3: labels ---------------------------------------------------------


def labels_page(
    sources: ev.Sources,
    dataset: ev.Dataset,
    report: ev.GuardReport,
    *,
    seed: int,
    source_args: tuple[str, str, str, str, str, str],
) -> None:
    """The label-text embedding map and one label's card."""
    st.markdown('<span class="ex-kicker">labels</span>', unsafe_allow_html=True)
    st.subheader("Where are the holes?")
    controls = st.columns(2)
    with controls[0]:
        variant = st.radio(
            "label text",
            ["title + parents", "title"],
            horizontal=True,
            key="l_variant",
        )
    with controls[1]:
        st.caption(f"unseen = absent from the {ev.LABELLED_DOCUMENTS} rows of seed **{seed}**")
    variant_key = "title_parents" if variant == "title + parents" else "title"
    try:
        label_map = compute_label_map(*source_args, variant_key)
    except (ev.MissingVectorError, FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"the viewer could not read the label cache: {exc}")
        st.stop()
    sample = ev.seed_sample(seed, n_sources=len(dataset.train))
    seen = {label for row in sample for label in dataset.train[int(row)].labels}
    unseen_labels = [label for label in dataset.labels if label not in seen]
    points = [
        {
            "x": float(label_map.coordinates[index, 0]),
            "y": float(label_map.coordinates[index, 1]),
            "label": ev.label_title(dataset, label),
            "class": "unseen in this seed" if label not in seen else "seen in this seed",
        }
        for index, label in enumerate(dataset.labels)
    ]
    st.scatter_chart(
        points,
        x="x",
        y="y",
        color="class",
        height=520,
    )
    st.caption(
        f"{label_map.method}: {label_map.note} · {len(dataset.labels):,} labels · "
        f"{len(seen):,} seen, {len(unseen_labels):,} unseen in seed {seed}. "
        "The map is a view of the cached label vectors, not a new embedding run."
    )
    parentless = sum(1 for label in dataset.labels if not dataset.catalog[label].parents)
    metrics = st.columns(3)
    metrics[0].metric("labels", f"{len(dataset.labels):,}")
    metrics[1].metric("unseen in this seed", f"{len(unseen_labels):,}")
    metrics[2].metric("parentless labels", f"{parentless:,}")

    options = [f"{ev.label_title(dataset, label)} · {label}" for label in dataset.labels]
    default_label = unseen_labels[0] if unseen_labels else dataset.labels[0]
    default_index = dataset.label_index[default_label]
    choice = st.selectbox(
        "pick a label (search by title or id)",
        options,
        index=default_index,
        key="l_label",
    )
    label = dataset.labels[options.index(choice)]
    entry = dataset.catalog[label]
    tag = ev.label_tag(dataset, label)
    st.markdown(f"### {_esc(tag.title)}")
    st.caption(
        f"label `{label}` · {tag.bucket} · train support {tag.support} · "
        + ("unseen in this seed" if label not in seen else "seen in this seed")
    )
    details = [
        ["parents", " · ".join(tag.parent_names) or "—"],
        ["ancestors", " · ".join(entry.ancestors) or "—"],
        ["definition", tag.definition or "—"],
        ["definition kind", tag.definition_kind or "—"],
        ["catalogue name", entry.name],
        ["label text (v2)", ev.label_text(dataset, label, "title_parents")],
    ]
    st.markdown(
        _table(("field", "value"), details, aligns=("left", "left")),
        unsafe_allow_html=True,
    )
    hits = [
        row
        for row, doc in enumerate(dataset.dev)
        if dataset.label_index[label] in dataset.gold[row]
    ]
    st.markdown(f"**Dev documents where this label is gold** ({len(hits)})")
    document_rows = [
        [
            f'<span class="ev-id">{_esc(dataset.dev[row].celex_id)}</span>',
            f'<span class="ev-label">{_esc(_snippet(dataset.dev[row].title, 110))}</span>',
        ]
        for row in hits[:25]
    ]
    st.markdown(
        _table(("CELEX", "title"), document_rows, empty="no dev document carries this label"),
        unsafe_allow_html=True,
    )
    if len(hits) > 25:
        st.caption(f"showing 25 of {len(hits)} documents.")
    _provenance(report)


# --- page 4: errors ---------------------------------------------------------


def errors_page(dataset: ev.Dataset, ranking: ev.Ranking, report: ev.GuardReport) -> None:
    """The pack's misses per document and per bucket (computed, not an artifact)."""
    st.markdown('<span class="ex-kicker">errors</span>', unsafe_allow_html=True)
    st.subheader("What the best arm misses")
    st.caption(
        "The TC-17 pack's dev ranking (`ranked_dev.npz`) against the dev gold "
        "labels. A miss is a gold label outside the pack's top-`depth`. This is "
        "a view of the ranking, not a logged metric."
    )
    depth = st.select_slider("depth", options=[10, 20, 50, 100], value=50, key="e_depth")
    misses = ev.pack_misses(dataset, ranking, depth=depth)
    metrics = st.columns(4)
    metrics[0].metric("gold assignments", f"{misses.n_gold_assignments:,}")
    metrics[1].metric(f"missed @{depth}", f"{misses.n_missed:,}")
    metrics[2].metric(
        "miss share",
        f"{misses.n_missed / misses.n_gold_assignments:.4f}" if misses.n_gold_assignments else "—",
    )
    docs_with_miss = sum(1 for row in misses.documents if row.n_missed)
    metrics[3].metric("documents with a miss", f"{docs_with_miss:,}")
    bucket = st.selectbox(
        "missed-label bucket",
        ["all", "head", "torso", "tail", "unseen"],
        key="e_bucket",
    )
    selected_docs = [
        row
        for row in misses.documents
        if bucket == "all" or any(dataset.bucket.get(label) == bucket for label in row.missed)
    ]
    st.markdown("**Documents with the most missed gold labels**")
    document_rows = []
    for row in selected_docs[:100]:
        missed = " ".join(
            _label_pill(
                ev.label_title(dataset, label),
                label,
                dataset.bucket.get(label, "unseen"),
            )
            for label in row.missed
        )
        document_rows.append(
            [
                f'<span class="ev-id">{_esc(row.doc_id)}</span>',
                f'<span class="ev-label">{_esc(_snippet(row.title, 90))}</span>',
                f'<span class="ev-rank">{row.n_gold}</span>',
                f'<span class="ev-rank">{row.n_missed}</span>',
                f'<span class="ev-rank">{row.n_missed / row.n_gold:.3f}</span>'
                if row.n_gold
                else '<span class="ev-muted">—</span>',
                f'<span class="ev-pills">{missed}</span>'
                if missed
                else '<span class="ev-muted">none</span>',
            ]
        )
    st.markdown(
        _table(
            ("CELEX", "title", "gold", "missed", "share", "missed labels"),
            document_rows,
            aligns=("left", "left", "right", "right", "right", "left"),
            compact=True,
            empty="no document has a miss in this bucket",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("**Labels missed most often**")
    selected_labels = [row for row in misses.labels if bucket == "all" or row.bucket == bucket]
    label_rows = [
        [
            _label_cell(row.label, row.title),
            _bucket_badge(row.bucket),
            f'<span class="ev-rank">{row.support}</span>',
            f'<span class="ev-rank">{row.n_missed}</span>',
            f'<span class="ev-rank">{row.n_gold}</span>',
            f'<span class="ev-rank">{row.n_missed / row.n_gold:.3f}</span>'
            if row.n_gold
            else '<span class="ev-muted">—</span>',
        ]
        for row in selected_labels[:100]
    ]
    st.markdown(
        _table(
            ("label", "bucket", "train support", "missed", "gold docs", "miss rate"),
            label_rows,
            aligns=("left", "center", "right", "right", "right", "right"),
            compact=True,
            empty="no label has a miss in this bucket",
        ),
        unsafe_allow_html=True,
    )
    _provenance(report)


# --- the app ----------------------------------------------------------------


def main() -> None:
    """The app's entry point: sources, the guard, then the four pages."""
    st.set_page_config(
        page_title="EURLEX-4K result viewer",
        page_icon="📚",
        layout="wide",
    )
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ex-hero"><span class="ex-kicker">EURLEX-4K · topic classification</span>'
        "<h1>Result viewer</h1>"
        '<div class="ex-meta">The track\'s artifacts, one document at a time. '
        "Read-only: dev only, test never read, no new metrics.</div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.header("Sources")
    results = st.sidebar.text_input(
        "results", value=_env("EURLEX_VIEW_RESULTS", "results"), key="src_results"
    )
    cache = st.sidebar.text_input(
        "cache", value=_env("EURLEX_VIEW_CACHE", "cache"), key="src_cache"
    )
    data_dir = st.sidebar.text_input(
        "data", value=_env("EURLEX_VIEW_DATA", "data/eurlex_4k"), key="src_data"
    )
    catalog = st.sidebar.text_input(
        "catalog",
        value=_env("EURLEX_VIEW_CATALOG", "catalogs/eurlex_labels.jsonl"),
        key="src_catalog",
    )
    ranking_path = st.sidebar.text_input(
        "TC-17 ranking",
        value=_env("EURLEX_VIEW_RANKING", "results/eurlex-embed-finetune/ranked_dev.npz"),
        key="src_ranking",
    )
    model_id = st.sidebar.text_input(
        "cache model id",
        value=_env("EURLEX_VIEW_MODEL", ev.MODEL_ID),
        key="src_model",
    )
    seed = st.sidebar.selectbox("TC-21 seed", ev.SEEDS, index=0, key="seed")
    if st.sidebar.button("reload the artifacts", key="src_reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
    source_args = (results, cache, data_dir, catalog, ranking_path, model_id)
    try:
        sources, dataset, ranking, rows = load_artifacts(*source_args)
    except (FileNotFoundError, ev.MissingVectorError, ValueError, KeyError) as exc:
        st.error(f"the tool could not read its sources: {exc}")
        st.stop()
    report = ev.check_all(dataset, ranking, sources, seed=seed)
    if not report.ok:
        st.error(
            "THE ARTIFACTS AND THE DATA DISAGREE — the guard did not reproduce "
            "the run. The views are blocked; stale numbers are never shown."
        )
        for check in report.checks:
            if not check.ok:
                st.code(f"{check.name}: {check.detail}", language="text")
        with st.expander("the whole guard report"):
            st.code("\n".join(report.lines()), language="text")
        st.stop()

    top = st.columns([5, 1])
    with top[0]:
        page = st.radio(
            "page",
            PAGES,
            horizontal=True,
            label_visibility="collapsed",
            key="page",
        )
    with top[1]:
        presenter = st.toggle(
            "presenter",
            value=False,
            key="presenter",
            help="Hides the sidebar and enlarges the text for a projected screen.",
        )
    if presenter:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"],
            [data-testid="stSidebarCollapsedControl"] { display: none !important; }
            /* full viewport: the old 1550px cap shrank the wide layout */
            .block-container {
              max-width: 100% !important; width: 100% !important;
              padding: 0.9rem 2.5rem 3rem !important;
            }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li { font-size: 1.1rem; }
            .ev-table, .ev-table.compact { font-size: 1rem !important; }
            .ev-pill, .ev-badge, .ev-chip { font-size: 0.88rem !important; }
            .ex-doc-title { font-size: 1.55rem !important; }
            .ex-part-body { font-size: 1.06rem !important; }
            .ex-part-label { font-size: 0.78rem !important; }
            [data-testid="stCaptionContainer"] p { font-size: 1rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        f"**Guard:** {'OK' if report.ok else 'MISMATCH'} · "
        f"{len(report.checks)} checks · {len(dataset.labels):,} labels"
    )
    with st.sidebar.expander("the guard report"):
        st.code("\n".join(report.lines()), language="text")

    if page == "One document":
        one_document_page(
            sources,
            dataset,
            ranking,
            report,
            seed=seed,
            source_args=source_args,
        )
    elif page == "Methods":
        methods_page(rows, report)
    elif page == "Labels":
        labels_page(
            sources,
            dataset,
            report,
            seed=seed,
            source_args=source_args,
        )
    else:
        errors_page(dataset, ranking, report)


if __name__ == "__main__":
    main()
