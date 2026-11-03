"""Measure the contrastive block's **novelty** (the track's CS-40).

The viewer's contrastive block answers one question — *why A and not B?* — by
reading the corpus map's two tables against each other. The change set's
registered falsification is that the block must answer a question **the
viewer's other blocks cannot**: if the discriminating words are the ones the
topic-side and the text-side maps already show, the block adds nothing and is
dropped.

This tool measures that. It reads the map's artifact and the split (no model
call, no embedding) and writes **its own** artifact:

    results/upr-hitmap-contrastive/contrastive.json

It is a **tool's** readout, not a CLI run's: it writes that one file and
touches no other artifact.

**The pairs.** Two kinds, both fixed in advance (the pre-registration, in
``TEXT_PROCESSING_PLAN.md`` section 7.5):

- the **alternative** pair — A is the focus (the gold label whose own name
  words the text carries the most) and B is the **map's own runner-up** among
  the labels the text does *not* carry: the non-gold label with the largest
  ``sum_zscore_positive`` score, among the labels whose table is not empty;
- the **sibling** pair — A is the focus and B is the text's **other** gold
  label with the largest map score (the case where the screen already shows
  both tables, so the novelty is a real test of the contrast).

**The measures** (per pair, then aggregated):

1. the **vocabulary share** — the share of the block's rows that are neither
   a name word of A or B nor one of the text's own words: the corpus's own
   vocabulary, which the name-overlap maps structurally cannot show;
2. the **relevance share** — the share of the pairs with at least one **text
   word** among the rows (the block must be about *this* text);
3. the **novelty share** — the share of the rows for which the screen (as it
   is) does not print **both** of the pair's log-odds: the reader could not
   have formed the difference;
4. the **sign-disagreement share** (reported, no bar) — the share of the pairs
   with at least one **text word** that is in **both** tables with **opposite
   signs**.

Usage:

    uv run python tools/contrastive_map_check.py
    uv run python tools/contrastive_map_check.py --split val --top-k 5

Exit code 0 when the measurement was made (the bars' verdict is a result, not
a failure), 1 on a usage or data error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cluster_rag.config import AppConfig, load_config
from cluster_rag.data import load_jsonl, load_split_ids, select_records_by_ids
from cluster_rag.text_processors.features.pair_features import DEFAULT_STOPWORDS, content_tokens
from cluster_rag.text_processors.lexical.contrastive import (
    DISPLAY_TOP_K,
    LEAN_LIMIT,
    NOVELTY_BAR,
    RELEVANCE_BAR,
    VOCABULARY_BAR,
    ContrastRow,
    corpus_contrast_rows,
    default_contrast_label,
    focus_label,
    map_score,
    novelty_share,
    shown_pairs,
    text_contrast_rows,
    vocabulary_share,
)

DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_SPLIT = "val"
DEFAULT_SEED = 42
CORPUS_MAP_RUN = "upr-hitmap-corpus"
DEFAULT_OUT = "upr-hitmap-contrastive"


def load_artifact(path: Path) -> dict[str, Any]:
    """One run's artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no artifact at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tables_of(artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every label's word table, from the map's artifact."""
    block = artifact.get("corpus_map") or {}
    return {
        str(row.get("label")): [dict(word) for word in row.get("words", [])]
        for row in block.get("per_label", [])
    }


def reverse_of(artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The reverse table (every word's label rows), from the map's artifact."""
    block = artifact.get("corpus_map") or {}
    return {
        str(row.get("word")): [dict(entry) for entry in row.get("labels", [])]
        for row in block.get("per_word", [])
    }


def block_rows(
    tables: dict[str, list[dict[str, Any]]],
    label_a: str,
    label_b: str,
    text_words: list[str],
) -> tuple[ContrastRow, ...]:
    """The block's rows for one pair: the two halves, deduplicated by word."""
    rows_a = tables.get(label_a) or []
    rows_b = tables.get(label_b) or []
    own = text_contrast_rows(rows_a, rows_b, text_words=text_words)
    corpus = corpus_contrast_rows(rows_a, rows_b)
    seen: dict[str, ContrastRow] = {}
    for row in (*own, *corpus):
        seen.setdefault(row.word, row)
    return tuple(sorted(seen.values(), key=lambda row: (-abs(row.contrast), row.word)))


def measure_pair(
    tables: dict[str, list[dict[str, Any]]],
    reverse: dict[str, list[dict[str, Any]]],
    label_a: str,
    label_b: str,
    text_words: list[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    """One pair's numbers.

    Args:
        tables: Every label's table (the artifact's own rows).
        reverse: The reverse table (the artifact's own rows).
        label_a: The first label of the pair.
        label_b: The second label of the pair.
        text_words: The text's own content words.
        top_k: How many rows of a table the screen displays.

    Returns:
        The pair's numbers (and the counts behind them).
    """
    rows = block_rows(tables, label_a, label_b, text_words)
    shown_a = shown_pairs(tables, reverse, label_a, text_words=text_words, top_k=top_k)
    shown_b = shown_pairs(tables, reverse, label_b, text_words=text_words, top_k=top_k)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_rows": len(rows),
        "n_text_rows": sum(1 for row in rows if row.in_text),
        "n_new_rows": sum(1 for row in rows if not (row.word in shown_a and row.word in shown_b)),
        "n_sign_disagreeing_text_rows": sum(
            1 for row in rows if row.in_text and row.sign_disagreement
        ),
        "vocabulary_share": vocabulary_share(rows, text_words=text_words),
        "novelty_share": novelty_share(rows, shown_a, shown_b),
        "has_text_row": any(row.in_text for row in rows),
        "has_sign_disagreeing_text_row": any(row.in_text and row.sign_disagreement for row in rows),
    }


def aggregate(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The aggregate of a set of pairs.

    Args:
        pairs: The pairs' own numbers.

    Returns:
        The aggregate (the means and the shares).
    """
    n = len(pairs)
    if not n:
        return {
            "n_pairs": 0,
            "mean_vocabulary_share": 0.0,
            "mean_novelty_share": 0.0,
            "share_with_a_text_row": 0.0,
            "share_with_a_new_row": 0.0,
            "share_with_a_sign_disagreeing_text_row": 0.0,
            "mean_rows": 0.0,
        }
    return {
        "n_pairs": n,
        "mean_vocabulary_share": sum(pair["vocabulary_share"] for pair in pairs) / n,
        "mean_novelty_share": sum(pair["novelty_share"] for pair in pairs) / n,
        "share_with_a_text_row": sum(1 for pair in pairs if pair["has_text_row"]) / n,
        "share_with_a_new_row": sum(1 for pair in pairs if pair["n_new_rows"] > 0) / n,
        "share_with_a_sign_disagreeing_text_row": (
            sum(1 for pair in pairs if pair["has_sign_disagreeing_text_row"]) / n
        ),
        "mean_rows": sum(pair["n_rows"] for pair in pairs) / n,
    }


def sibling_label(
    tables: dict[str, list[dict[str, Any]]],
    gold_labels: list[str],
    focus: str,
    text_words: list[str],
) -> str:
    """The text's **other** gold label with the largest map score.

    Args:
        tables: Every label's table (the artifact's own rows).
        gold_labels: The text's gold labels.
        focus: The focus label (excluded).
        text_words: The text's own content words.

    Returns:
        The sibling label ("" when the text has no other gold label).
    """
    best = ""
    best_score = 0.0
    for label in gold_labels:
        if label == focus:
            continue
        score = map_score(tables.get(label) or (), text_words)
        if score > best_score:
            best = label
            best_score = score
    return best


def run(
    config: AppConfig,
    results: Path,
    *,
    split: str,
    top_k: int,
    corpus_map_run: str,
    out_run: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """The measurement over the split's texts.

    Args:
        config: The resolved app config (the paths).
        results: The results folder.
        split: The split to measure.
        top_k: How many rows of a table the screen displays.
        corpus_map_run: The map's run id.
        out_run: The run id of this tool's own artifact.

    Returns:
        The artifact (JSON-friendly), the alternative pairs and the sibling
        pairs.

    Raises:
        FileNotFoundError: If the map's artifact is missing.
    """
    map_path = results / corpus_map_run / "hitmap.json"
    artifact = load_artifact(map_path)
    digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    tables = tables_of(artifact)
    reverse = reverse_of(artifact)

    report = load_jsonl(config.paths.data_raw)
    ids, _ = load_split_ids(config.paths.data_splits, split)
    records = select_records_by_ids(report.records, ids)
    labels = sorted({label for record in report.records for label in record.labels})

    alternative: list[dict[str, Any]] = []
    sibling: list[dict[str, Any]] = []
    for record in records:
        text_words = list(dict.fromkeys(content_tokens(record.text, DEFAULT_STOPWORDS)))
        focus = focus_label(record.text, list(record.labels), labels)
        if not focus:
            continue
        against = default_contrast_label(
            tables, text_words, exclude=list(record.labels), labels=labels
        )
        if against:
            alternative.append(
                {
                    "text_id": record.id,
                    **measure_pair(
                        tables,
                        reverse,
                        focus,
                        against,
                        text_words,
                        top_k=top_k,
                    ),
                }
            )
        other = sibling_label(tables, list(record.labels), focus, text_words)
        if other:
            sibling.append(
                {
                    "text_id": record.id,
                    **measure_pair(tables, reverse, focus, other, text_words, top_k=top_k),
                }
            )

    pooled = aggregate([*alternative, *sibling])
    summary = {
        "alternative": aggregate(alternative),
        "sibling": aggregate(sibling),
        "pooled": pooled,
    }
    verdict = {
        "vocabulary_share": {
            "value": pooled["mean_vocabulary_share"],
            "bar": VOCABULARY_BAR,
            "holds": pooled["mean_vocabulary_share"] >= VOCABULARY_BAR,
        },
        "relevance_share": {
            "value": pooled["share_with_a_text_row"],
            "bar": RELEVANCE_BAR,
            "holds": pooled["share_with_a_text_row"] >= RELEVANCE_BAR,
        },
        "novelty_share": {
            "value": pooled["mean_novelty_share"],
            "bar": NOVELTY_BAR,
            "holds": pooled["mean_novelty_share"] >= NOVELTY_BAR,
        },
        "pairs_with_a_new_row": {
            "value": pooled["share_with_a_new_row"],
            "bar": NOVELTY_BAR,
            "holds": pooled["share_with_a_new_row"] >= NOVELTY_BAR,
        },
    }
    verdict["holds"] = all(entry["holds"] for entry in verdict.values())
    payload = {
        "run_id": out_run,
        "tool": "contrastive_map_check",
        "split": split,
        "source": {
            "corpus_map_run": corpus_map_run,
            "corpus_map_artifact": str(map_path),
            "corpus_map_sha256": digest,
            "corpus": (artifact.get("corpus_map") or {}).get("corpus", {}),
        },
        "definitions": {
            "contrast": (
                "lo_A(w) minus lo_B(w): the two log-odds, the stored value when the "
                "pair is in the table and 0 when it is not (the map's silence "
                "convention)"
            ),
            "block_rows": (
                "the text's own words' rows plus the corpus's top-10 rows in each "
                "direction, deduplicated by word"
            ),
            "vocabulary_share": (
                "the share of the rows that are neither a name word of A or B nor "
                "one of the text's own words: the corpus's own vocabulary, which "
                "the topic-side and the text-side maps cannot show"
            ),
            "relevance_share": (
                "the share of the pairs with at least one text word among the rows"
            ),
            "novelty_share": (
                "the share of the rows for which the screen does not print both of "
                "the pair's log-odds"
            ),
            "sign_disagreement": (
                "the word is in both tables with opposite signs: the plan's first "
                "reading, as a flag"
            ),
        },
        "knobs": {
            "split": split,
            "display_top_k": top_k,
            "lean_limit": LEAN_LIMIT,
            "seed": DEFAULT_SEED,
            "note": (
                "the display limits are the screen's own (the viewer's --top-k and "
                "its reverse table's limit): the novelty measure reads the screen "
                "as it is"
            ),
        },
        "summary": summary,
        "verdict": verdict,
        "pairs": {
            "alternative": alternative,
            "sibling": sibling,
        },
    }
    return payload, alternative, sibling


def main(argv: list[str] | None = None) -> int:
    """The CLI: the config, the results folder and the split."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="the config with the paths"
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="the split to measure")
    parser.add_argument("--results", type=Path, default=None, help="the results folder")
    parser.add_argument("--corpus-map-run", default=CORPUS_MAP_RUN, help="the map's run id")
    parser.add_argument(
        "--out-run", default=DEFAULT_OUT, help="the run id of this tool's own artifact"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DISPLAY_TOP_K,
        help="how many rows of a table the screen displays (the novelty measure)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the summary without writing the artifact",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        results = args.results if args.results is not None else Path(config.paths.results)
        payload, _alternative, _sibling = run(
            config,
            results,
            split=args.split,
            top_k=args.top_k,
            corpus_map_run=args.corpus_map_run,
            out_run=args.out_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"contrastive_map_check: {exc}")
        return 1

    summary = payload["summary"]
    verdict = payload["verdict"]
    print(f"the contrastive map's novelty — the split {payload['split']}")
    print(
        f"  the map's artifact: {payload['source']['corpus_map_artifact']} "
        f"(sha256 {payload['source']['corpus_map_sha256'][:12]}…)"
    )
    for name in ("alternative", "sibling", "pooled"):
        row = summary[name]
        print(
            f"  {name:<12}: {row['n_pairs']:>4} pairs · the vocabulary share "
            f"{row['mean_vocabulary_share']:.4f} · the novelty share "
            f"{row['mean_novelty_share']:.4f} · with a text row "
            f"{row['share_with_a_text_row']:.4f} · with a new row "
            f"{row['share_with_a_new_row']:.4f} · with a sign-disagreeing text row "
            f"{row['share_with_a_sign_disagreeing_text_row']:.4f}"
        )
    print("  the registered bars:")
    for name in ("vocabulary_share", "relevance_share", "novelty_share", "pairs_with_a_new_row"):
        entry = verdict[name]
        print(
            f"    {name:<24} {entry['value']:.4f} "
            f"(bar {entry['bar']:.2f}) — {'holds' if entry['holds'] else 'FAILS'}"
        )
    print(
        "  the verdict: the block adds something the screen does not already have"
        if verdict["holds"]
        else "  the verdict: THE BLOCK ADDS NOTHING — it is dropped"
    )

    if not args.no_write:
        out = results / args.out_run
        out.mkdir(parents=True, exist_ok=True)
        (out / "contrastive.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  the artifact: {out / 'contrastive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
