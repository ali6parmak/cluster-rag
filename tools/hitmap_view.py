"""The hit map's **terminal viewer** (the track's CS-20).

One command, one text on screen: the text's words and **what each contributes**,
**who the words say it is**, **what the model says** (live), **what is true**
(the gold labels) and — behind a flag — **what happens if this word is removed**
(the causal mode E, one word at a time).

The map exists as four modes (CS-18's topic-side map + causal ablation, CS-19's
text-side map + surrogate attribution) and all four live in artifacts. This tool
is the **per-document view** over them: it reads the artifacts and the data, and
it perturbs nothing unless it is asked to.

CS-23 added a fifth block to the screen: the **corpus map** — the train
split's own word↔label association (the data's own statistics, not the model's
and not a feature's). It shows in two directions: the topic's own words by
log-odds (the words that are **in this text** marked), and this text's words
with the labels they lean to.

The three modes and their costs:

- **offline** (the default): the artifacts + the data + the split —
  **0 model calls**;
- **live** (``--live``): the model, re-scored through the cached calls —
  2 calls per text, and cache hits for the val texts;
- **what-if** (``--what-if word``): one re-scored grid per word —
  **2 real calls per word**, and the model is **not deterministic**.

Usage:

    uv run python tools/hitmap_view.py --sample --seed 7
    uv run python tools/hitmap_view.py --text 0000000000000047d5d150d7
    uv run python tools/hitmap_view.py --sample --seed 7 --what-if ratification,migrant
    uv run python tools/hitmap_view.py --text 0000000000000047d5d150d7 --min-df 3

**The tool is a tool, not a number source.** The offline numbers are the
artifacts' own (the map's exact per-word decomposition and the surrogate's
stored coefficients), and the live and what-if numbers are printed as the
tool's own readings, with the model and its digest named. Anything that becomes
a claim lands in an artifact first (the track's own rule).

**The attribution/causation distinction is visible in the output.** The lexical
and the surrogate columns are *attributions* (a decomposition of a feature's own
definition and a ridge surrogate's coefficients); only the what-if column
perturbs the input, so only it is *causal*. The lesson of CS-18 and CS-19 is
exactly that difference, and a viewer that blurs it would teach the wrong thing.
The **corpus map** is neither: it is a description of the *data*, and it says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cluster_rag.commands.upr import build_upr_decider
from cluster_rag.config import (
    AppConfig,
    DecisionModelConfig,
    UprAnalysisConfig,
    UprDecisionConfig,
    load_config,
)
from cluster_rag.data import (
    Record,
    build_label_matrix,
    load_jsonl,
    load_split_ids,
    select_records_by_ids,
)
from cluster_rag.decision.base import answer_probability
from cluster_rag.omct_decision import DecisionArm
from cluster_rag.text_processors.decision.probe import plan_calls
from cluster_rag.text_processors.decision.upr_probe import build_upr_cells
from cluster_rag.text_processors.features.corpus_map import build_corpus_map
from cluster_rag.text_processors.features.lexical_map import (
    cell_coverage,
    label_word_contributions,
    text_side_coverage,
    text_word_contributions,
)
from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    content_tokens,
)

# The defaults (the tool's own configuration surface; every one of them is a CLI
# flag, so nothing is hard-coded in the logic).
DEFAULT_CONFIG = Path("configs/base.yaml")
DEFAULT_SPLIT = "val"
DEFAULT_SEED = 42
DEFAULT_TOP_K = 5
DEFAULT_MAX_QUESTIONS = 64
TEXT_MAP_RUN = "upr-hitmap-text"
SURROGATE_RUN = "upr-hitmap-surrogate"
CORPUS_MAP_RUN = "upr-hitmap-corpus"

# How many labels a text's word "leans to" in the corpus map's reverse table.
CORPUS_WORD_LABELS = 3

# The artifacts' values are rounded to six decimals, so the cross-check's
# tolerance is the rounding (plus a hair).
VALUE_TOLERANCE = 1e-6

# The model's own noise band (the CS-7a precedent: the same call re-run moves
# the probability by ~0.001-0.003). A what-if delta inside the band is the
# model twitching, not the model using the word.
NOISE_BAND = 0.003

# The ANSI palette (raw escape codes: no dependency). The 256-colour ramps
# carry the shades: **the brightness is the size** of the value.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[38;5;51m"
GREEN_RAMP: tuple[str, ...] = tuple(f"\033[38;5;{code}m" for code in (22, 28, 34, 40, 46))
RED_RAMP: tuple[str, ...] = tuple(f"\033[38;5;{code}m" for code in (52, 88, 124, 160, 196))
# The gold labels: bold and bright green (the truth, in every ranking).
GOLD = BOLD + GREEN_RAMP[-1]


def paint(text: str, code: str, *, colour: bool) -> str:
    """Wrap a text in an ANSI code (or leave it alone in the plain fallback)."""
    return f"{code}{text}{RESET}" if colour else text


def ramp_code(value: float, maximum: float, ramp: Sequence[str]) -> str:
    """The shade for a value: the brightness is the size (nothing here is dim).

    Args:
        value: The value to shade.
        maximum: The scale's maximum (the brightest shade).
        ramp: The ramp's codes, the darkest first.

    Returns:
        The ANSI code for the value's shade.
    """
    if maximum <= 0.0 or value <= 0.0:
        return DIM
    index = min(len(ramp) - 1, int(round((len(ramp) - 1) * value / maximum)))
    return ramp[index]


def signed_ramp(value: float, maximum: float) -> str:
    """The ramp for a signed value: the green ramp up, the red ramp down.

    A value of zero is **dim** — nothing here — and the brightness follows the
    value's size on its own side of the scale.
    """
    if maximum <= 0.0 or value == 0.0:
        return DIM
    if value > 0.0:
        return ramp_code(value, maximum, GREEN_RAMP)
    return ramp_code(-value, maximum, RED_RAMP)


def _bar(value: float, maximum: float, width: int = 24) -> str:
    """A bar for a value, scaled by the table's maximum."""
    if maximum <= 0.0 or value <= 0.0:
        return ""
    filled = int(round(width * min(value, maximum) / maximum))
    return "█" * max(filled, 1)


@dataclass(frozen=True)
class ViewerContext:
    """Everything the viewer reads: the split, the label space and the artifacts.

    Attributes:
        config: The resolved app config (the paths).
        split: The split on screen (``val`` — the track's split).
        records: The split's records, in the split's own order (the grid's rows).
        labels: The label space (the grid's columns), sorted.
        gold: The split's gold matrix (rows x labels).
        text_map: The text-side map's artifact (CS-19's ``upr-hitmap-text``).
        surrogate: The surrogate's artifact (CS-19's ``upr-hitmap-surrogate``).
        text_map_run: The text-side map's run id (the provenance).
        surrogate_run: The surrogate's run id (the provenance).
        corpus_map: The corpus map's artifact (CS-23's ``upr-hitmap-corpus``) —
            ``None`` when the run has not been made.
        corpus_map_run: The corpus map's run id (the provenance).
        train_records: The **train** split's records (the corpus map's corpus:
            the map is built from the train texts alone).
    """

    config: AppConfig
    split: str
    records: tuple[Record, ...]
    labels: tuple[str, ...]
    gold: np.ndarray
    text_map: dict[str, Any]
    surrogate: dict[str, Any]
    text_map_run: str
    surrogate_run: str
    corpus_map: dict[str, Any] | None
    corpus_map_run: str
    train_records: tuple[Record, ...]


def load_artifact(path: Path) -> dict[str, Any]:
    """One run's artifact.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no artifact at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(
    config: AppConfig,
    *,
    split: str,
    text_map_run: str,
    surrogate_run: str,
    results: Path,
    corpus_map_run: str = CORPUS_MAP_RUN,
) -> ViewerContext:
    """Read everything the viewer needs: the split, the gold matrix, the artifacts.

    The label space is the **whole dataset's** (the harness's own space, not the
    split's), and the rows are the split's ids in their own order — the grid the
    map's rows and the decision cache's cell indexes are built on.

    The **corpus map** (CS-23) is optional: the run may not have been made, and
    the viewer says so on the screen rather than refusing to start. The **train**
    split is read for it, because the map's corpus is the train texts.

    Raises:
        FileNotFoundError: If an artifact is missing.
        ValueError: If an artifact was measured on another split.
    """
    report = load_jsonl(config.paths.data_raw)
    ids, _ = load_split_ids(config.paths.data_splits, split)
    records = select_records_by_ids(report.records, ids)
    labels = sorted({label for record in report.records for label in record.labels})
    gold = build_label_matrix(records, labels)
    text_map = load_artifact(results / text_map_run / "hitmap.json")
    surrogate = load_artifact(results / surrogate_run / "hitmap.json")
    for name, artifact in ((text_map_run, text_map), (surrogate_run, surrogate)):
        if artifact.get("split") != split:
            raise ValueError(
                f"the artifact {name} was measured on the split {artifact.get('split')!r}, "
                f"not {split!r}: the viewer's rows would not line up with it"
            )
    corpus_map: dict[str, Any] | None = None
    corpus_path = results / corpus_map_run / "hitmap.json"
    if corpus_path.is_file():
        corpus_map = load_artifact(corpus_path)
    train_ids, _ = load_split_ids(config.paths.data_splits, "train")
    train_records = select_records_by_ids(report.records, train_ids)
    return ViewerContext(
        config=config,
        split=split,
        records=tuple(records),
        labels=tuple(labels),
        gold=gold,
        text_map=text_map,
        surrogate=surrogate,
        text_map_run=text_map_run,
        surrogate_run=surrogate_run,
        corpus_map=corpus_map,
        corpus_map_run=corpus_map_run,
        train_records=tuple(train_records),
    )


def pick_row(records: Sequence[Record], *, text_id: str | None, sample: bool, seed: int) -> int:
    """The row of the text on screen: by its record id, or by a seeded draw.

    The seeded draw is the track's own convention (``sample_texts``): a
    permutation of the split's rows, and its first element — so the same seed
    picks the same text as the track's own draws.

    Raises:
        ValueError: If neither a record id nor a sample is asked for, or the
            record is not in the split.
    """
    if text_id:
        for index, record in enumerate(records):
            if record.id == text_id:
                return index
        raise ValueError(f"the record {text_id!r} is not in the split")
    if sample:
        generator = np.random.default_rng(seed)
        return int(generator.permutation(len(records))[0])
    raise ValueError("pick a text with --text <record id> or --sample --seed <n>")


def lexical_ranking(text: str, labels: Sequence[str]) -> list[tuple[str, float]]:
    """The text-side map's own ranking: the text's words against every label.

    ``lex_text_coverage`` — "the text's words, how much of each is in the
    label's name" — for every label, strongest first. The tie-break is the
    label space's own order, exactly as the map's pass breaks it.
    """
    values = [(label, text_side_coverage(text, label)) for label in labels]
    return sorted(values, key=lambda item: (-item[1], item[0]))


def topic_ranking(text: str, labels: Sequence[str]) -> list[tuple[str, float]]:
    """The topic-side map's own ranking: every label's words against the text.

    ``lex_label_coverage`` — "the label's words, how much of each is in the
    text" — for every label, strongest first (CS-18's mode A).
    """
    values = [(label, cell_coverage(text, label)) for label in labels]
    return sorted(values, key=lambda item: (-item[1], item[0]))


def word_rows(text: str, label: str) -> list[dict[str, Any]]:
    """The text's words and their exact contribution toward one label.

    The text side of the word-overlap feature's own definition (mode B): a word
    that occurs twice is **one** row with its occurrences counted and its
    contributions added — the feature averages over the token occurrences, so
    grouping them changes neither the sum nor the share.
    """
    grouped: dict[str, dict[str, float]] = {}
    for contribution in text_word_contributions(text, label):
        row = grouped.setdefault(
            contribution.word,
            {"coverage": 0.0, "contribution": 0.0, "n": 0.0},
        )
        row["coverage"] += contribution.coverage
        row["contribution"] += contribution.contribution
        row["n"] += 1.0
    rows = [
        {
            "word": word,
            "n_occurrences": int(row["n"]),
            "coverage": row["coverage"] / row["n"],
            "contribution": row["contribution"],
        }
        for word, row in grouped.items()
    ]
    rows.sort(key=lambda row: (-float(row["contribution"]), str(row["word"])))
    return rows


def topic_words(text: str, label: str) -> list[str]:
    """The topic's own content words (the topic side of the map, mode A)."""
    return [contribution.word for contribution in label_word_contributions(text, label)]


def surrogate_block(
    artifact: Mapping[str, Any], arm_name: str | None
) -> tuple[str, dict[str, Any]]:
    """The surrogate's block of one arm (the arm's name and its block).

    Raises:
        KeyError: If the artifact has no surrogate block.
    """
    arms = artifact.get("ablation_map", {}).get("arms", {})
    if arm_name:
        block = arms.get(arm_name)
        if block is None:
            raise KeyError(f"the artifact has no arm {arm_name!r}; it has {sorted(arms)}")
        chosen = arm_name
    else:
        chosen, block = next(iter(arms.items()))
    surrogate = block.get("surrogate")
    if not surrogate:
        raise KeyError(f"the arm {chosen!r} carries no surrogate block")
    return chosen, dict(surrogate)


def surrogate_full_vector(surrogate: Mapping[str, Any]) -> dict[str, Any] | None:
    """The artifact's **full** coefficient vector (CS-24), if it is there.

    The block carries a shared vocabulary and one row of numbers per topic, so
    the words are stored once and every word of the model's vocabulary has a
    coefficient. An artifact without the block (the CS-19 shape) gives
    ``None``: the viewer then reads the top-15 lists and says so.
    """
    vector = surrogate.get("full_vector")
    if not isinstance(vector, Mapping):
        return None
    vocabulary = [str(word) for word in (vector.get("vocabulary") or [])]
    topics = [str(topic) for topic in (vector.get("topics") or [])]
    rows = list(vector.get("coefficients") or [])
    per_topic: dict[str, dict[str, float]] = {}
    for index, topic in enumerate(topics):
        if index >= len(rows):
            continue
        row = rows[index]
        if not isinstance(row, list) or len(row) != len(vocabulary):
            # a malformed row is not read: the guard reports the disagreement
            continue
        per_topic[topic] = {word: float(value) for word, value in zip(vocabulary, row, strict=True)}
    return {"vocabulary": vocabulary, "topics": topics, "rows": per_topic}


def surrogate_coefficients(surrogate: Mapping[str, Any], topic: str) -> dict[str, float]:
    """One topic's stored coefficients.

    The **full** vector when the artifact carries it (CS-24: every word of the
    model's vocabulary has a number), and the **top-15 lists** otherwise (the
    CS-19 artifact's shape, where a word outside the lists has no stored
    coefficient — never a coefficient of zero).
    """
    vector = surrogate_full_vector(surrogate)
    if vector is not None:
        return dict(vector["rows"].get(topic, {}))
    for row in surrogate.get("per_topic", []):
        if row.get("topic") != topic:
            continue
        coefficients: dict[str, float] = {}
        for side in ("top_positive_words", "top_negative_words"):
            for entry in row.get(side, []):
                coefficients[str(entry["word"])] = float(entry["coefficient"])
        return coefficients
    return {}


def cross_check(context: ViewerContext, row: int, text: str) -> list[str]:
    """Compare the viewer's own numbers with the artifacts' (the stale-artifact guard).

    The viewer recomputes the text-side map with the very same pure function the
    map's run used, so the artifact's own rows are the reference: the text's top
    label and every gold cell's value. A mismatch means the artifact and the
    data disagree — a **stale artifact** — and it is reported, never smoothed
    over.
    """
    problems: list[str] = []
    labels = context.labels
    ranking = lexical_ranking(text, labels)
    stored = context.text_map.get("text_side_map", {}).get("per_text_top_label", [])
    if row < len(stored):
        entry = stored[row]
        if entry.get("label") != ranking[0][0]:
            problems.append(
                f"the artifact's top label for row {row} is {entry.get('label')!r}, "
                f"the viewer's is {ranking[0][0]!r}"
            )
        if abs(float(entry.get("value", 0.0)) - ranking[0][1]) > VALUE_TOLERANCE:
            problems.append(
                f"the artifact's top value for row {row} is {entry.get('value')}, "
                f"the viewer's is {ranking[0][1]:.6f}"
            )
    else:
        problems.append(f"the artifact has no per-text row {row}")
    for cell in context.text_map.get("text_side_map", {}).get("per_gold_cell", []):
        if int(cell.get("text_row", -1)) != row:
            continue
        expected = text_side_coverage(text, str(cell.get("label")))
        if abs(float(cell.get("value", 0.0)) - expected) > VALUE_TOLERANCE:
            problems.append(
                f"the artifact's gold cell {cell.get('label')!r} for row {row} is "
                f"{cell.get('value')}, the viewer's is {expected:.6f}"
            )
    expected_gold = int(context.gold.sum())
    stored_gold = context.text_map.get("text_set", {}).get("n_gold_cells")
    if stored_gold is not None and int(stored_gold) != expected_gold:
        problems.append(
            f"the artifact holds {stored_gold} gold cells, the split has {expected_gold}"
        )
    return problems


def corpus_block(artifact: Mapping[str, Any] | None) -> dict[str, Any]:
    """The corpus map's block from the artifact (the run's own block).

    An artifact without the block (the run was made before CS-23) gives an
    empty block: the screen says the map is not there, it never invents it.
    """
    if not artifact:
        return {}
    block = artifact.get("corpus_map")
    return dict(block) if isinstance(block, Mapping) else {}


def corpus_topic_rows(
    artifact: Mapping[str, Any] | None,
    label: str,
    *,
    min_df: int = 0,
) -> list[dict[str, Any]]:
    """One label's word rows, strongest log-odds first (the stored order).

    The rows are the artifact's own: the map stores **every** pair that
    occurs, and the ``min_df`` filter is a **display-time** filter (the counts
    sit next to the number, so a reader can judge the noise).
    """
    block = corpus_block(artifact)
    for row in block.get("per_label", []):
        if row.get("label") != label:
            continue
        rows = [dict(word) for word in row.get("words", [])]
        if min_df > 0:
            rows = [row for row in rows if int(row.get("df", 0)) >= min_df]
        return rows
    return []


def corpus_word_rows(
    artifact: Mapping[str, Any] | None,
    word: str,
    *,
    limit: int = CORPUS_WORD_LABELS,
) -> list[dict[str, Any]]:
    """One word's label rows (the reverse table), strongest log-odds first."""
    block = corpus_block(artifact)
    for row in block.get("per_word", []):
        if row.get("word") != word:
            continue
        return [dict(entry) for entry in row.get("labels", [])][:limit]
    return []


def corpus_cross_check(
    context: ViewerContext,
    label: str,
    words: Sequence[str] = (),
) -> list[str]:
    """The corpus map's numbers against the artifact's (the stale-artifact guard).

    The viewer recomputes the map with the very same pure function the run used
    and compares its own rows with the artifact's: the topic's word table (the
    counts and the two measures) and the reverse table's rows for the words of
    the text on screen. A mismatch means the artifact and the data disagree — a
    **stale artifact** — and it is reported, never smoothed over.
    """
    problems: list[str] = []
    artifact = context.corpus_map
    if not artifact:
        return problems
    corpus = build_corpus_map(
        [record.text for record in context.train_records],
        [record.labels for record in context.train_records],
        context.labels,
    )
    stored_labels = {
        row.get("label"): row for row in artifact.get("corpus_map", {}).get("per_label", [])
    }
    mine = corpus.topic_rows(label)
    theirs = [dict(word) for word in (stored_labels.get(label) or {}).get("words", [])]
    if len(mine) != len(theirs):
        problems.append(
            f"the artifact holds {len(theirs)} words for {label!r}, the viewer's map has "
            f"{len(mine)}"
        )
    for pair, row in zip(mine, theirs, strict=False):
        if row.get("word") != pair.word:
            problems.append(
                f"the artifact's word {row.get('word')!r} for {label!r} is not the viewer's "
                f"{pair.word!r}"
            )
            break
        if int(row.get("n_with", -1)) != pair.n_with or int(row.get("df", -1)) != pair.df:
            problems.append(
                f"the artifact's counts for {label!r}/{pair.word!r} are "
                f"n_with={row.get('n_with')}, df={row.get('df')}, the viewer's are "
                f"n_with={pair.n_with}, df={pair.df}"
            )
            break
        if abs(float(row.get("lift", 0.0)) - pair.lift) > VALUE_TOLERANCE:
            problems.append(
                f"the artifact's lift for {label!r}/{pair.word!r} is {row.get('lift')}, "
                f"the viewer's is {pair.lift:.6f}"
            )
            break
        if abs(float(row.get("log_odds", 0.0)) - pair.log_odds) > VALUE_TOLERANCE:
            problems.append(
                f"the artifact's log-odds for {label!r}/{pair.word!r} is {row.get('log_odds')}, "
                f"the viewer's is {pair.log_odds:.6f}"
            )
            break
    stored_words = {
        row.get("word"): row for row in artifact.get("corpus_map", {}).get("per_word", [])
    }
    for word in words:
        mine_rows = corpus.word_rows(word)
        their_rows = [dict(entry) for entry in (stored_words.get(word) or {}).get("labels", [])]
        if len(mine_rows) != len(their_rows):
            problems.append(
                f"the artifact holds {len(their_rows)} labels for the word {word!r}, "
                f"the viewer's map has {len(mine_rows)}"
            )
            continue
        for pair, row in zip(mine_rows, their_rows, strict=False):
            if row.get("label") != pair.label:
                problems.append(
                    f"the artifact's label {row.get('label')!r} for the word {word!r} is "
                    f"not the viewer's {pair.label!r}"
                )
                break
            if abs(float(row.get("log_odds", 0.0)) - pair.log_odds) > VALUE_TOLERANCE:
                problems.append(
                    f"the artifact's log-odds for {word!r}/{pair.label!r} is "
                    f"{row.get('log_odds')}, the viewer's is {pair.log_odds:.6f}"
                )
                break
    return problems


def remove_word(text: str, word: str) -> tuple[str, int]:
    """The text without a word: every occurrence of it, and how many there were.

    The removal is the **causal** one: the word is taken out of the text and the
    model is asked again, so the word's effect is read from a perturbed input.

    Raises:
        ValueError: If the word is not in the text.
    """
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", re.IGNORECASE)
    stripped, count = pattern.subn(" ", text)
    if count == 0:
        raise ValueError(f"the word {word!r} is not in the text")
    return " ".join(stripped.split()), count


def build_decider(
    context: ViewerContext,
    arm_name: str | None,
) -> tuple[Any, DecisionArm, str, dict[str, Any]]:
    """The decision model client, built from the **artifact's own** pinned config.

    The viewer has no model of its own: it reads the very model the map used
    (the id, the pinned digest, the prompt and the prompt version), so the live
    mode is read against the same instrument as the map.

    Returns:
        The decider (the cache in front of the client), the arm, the arm's name
        and the model's provenance block.
    """
    chosen, surrogate = surrogate_block(context.surrogate, arm_name)
    arms = context.surrogate.get("ablation_map", {}).get("arms", {})
    model_info = dict(arms[chosen].get("model", {}))
    models = [dict(model) for model in context.surrogate.get("config", {}).get("models", [])]
    model_dict = next((model for model in models if model.get("name") == chosen), None)
    if model_dict is None:
        if not models:
            raise KeyError(f"the artifact {chosen!r} names no model to read")
        model_dict = models[0]
    model_config = DecisionModelConfig(**model_dict)
    decision = UprDecisionConfig(
        models=(model_config,),
        host=str(model_info.get("host") or ""),
        prompt_version=str(model_info.get("prompt_version") or "v1"),
        max_questions_per_call=DEFAULT_MAX_QUESTIONS,
    )
    analysis = UprAnalysisConfig(decision=decision)
    config = context.config.model_copy(update={"upr_analysis": analysis})
    decider = build_upr_decider(model_config, config)
    arm = DecisionArm(
        name=model_config.name,
        state=model_config.state,
        question_type=model_config.question_type,
        instructions=model_config.instructions,
        options=tuple(model_config.options),
        criteria=dict(model_config.criteria),
        positive_option=model_config.positive_option,
        max_input_tokens=model_config.max_input_tokens,
    )
    return decider, arm, chosen, model_info


def score_grid(
    decider: Any,
    arm: DecisionArm,
    text: str,
    labels: Sequence[str],
    *,
    name_base: int,
    max_questions_per_call: int = DEFAULT_MAX_QUESTIONS,
) -> tuple[np.ndarray, int]:
    """Score one text against every label: the probabilities and the calls' count.

    The call is the **same call the CS-9b grid run made**: the same state, the
    same question specs and the same question names (the grid's own cell index),
    so the val texts' answers come from the decision cache. The names are
    rebuilt from ``name_base`` — the text's row in the grid — which is what
    makes the cache hit.
    """
    cells = build_upr_cells([text], labels, np.zeros((1, len(labels)), dtype=bool))
    calls = plan_calls(
        cells,
        state_shape=arm.state,
        instructions=arm.instructions,
        question_type=arm.question_type,
        options=arm.options,
        criteria=arm.criteria,
        max_questions_per_call=max_questions_per_call,
        max_input_tokens=arm.max_input_tokens,
    )
    probabilities: list[float | None] = [None] * len(cells)
    for call in calls:
        names = tuple(f"c{name_base + index}" for index in call.cell_indexes)
        answers = decider.decide(call.state, dict(zip(names, call.specs, strict=True)))
        for name, index in zip(names, call.cell_indexes, strict=True):
            answer = answers.get(name)
            if answer is None:
                raise RuntimeError(f"the decision call returned no answer for {name!r}")
            probabilities[index] = answer_probability(answer, arm.positive_option)
    missing = [index for index, value in enumerate(probabilities) if value is None]
    if missing:
        raise RuntimeError(f"{len(missing)} cells were not scored by the decision model")
    return np.asarray(probabilities, dtype=np.float64), len(calls)


def _ranking_rows(
    ranking: Sequence[tuple[str, float]],
    limit: int,
) -> list[tuple[int, str, float]]:
    """The ranking's rows: the rank, the label and the value."""
    return [(index + 1, label, value) for index, (label, value) in enumerate(ranking[:limit])]


def _ranking_rows_with_gold(
    ranking: Sequence[tuple[str, float]],
    limit: int,
    gold_labels: Sequence[str],
) -> list[tuple[int, str, float]]:
    """The ranking's rows: the top ones **and** the gold labels it did not show.

    A label that is true for the text must be readable whatever its rank is
    (CS-27), so the gold labels the top-k left out follow it, each with its
    true rank. The order is the rank's, so the list never lies about where a
    label sits.
    """
    rows = _ranking_rows(ranking, limit)
    shown = {label for _, label, _ in rows}
    extras = [
        (rank, label, value)
        for rank, (label, value) in enumerate(ranking, start=1)
        if label in gold_labels and label not in shown
    ]
    return rows + extras


def _cache_line(cache: Mapping[str, int] | None) -> str:
    """The cache's own counters, so "free" is a measurement and not a claim."""
    if not cache:
        return ""
    return (
        f"(the decision cache: {int(cache.get('hits', 0))} hits, "
        f"{int(cache.get('misses', 0))} misses, {int(cache.get('stored', 0))} stored)"
    )


def _cache_counters(decider: Any) -> dict[str, int]:
    """The decision cache's own counters (the hits, the misses, the stored)."""
    return {
        "hits": int(getattr(decider, "cache_hits", 0)),
        "misses": int(getattr(decider, "cache_misses", 0)),
        "stored": int(getattr(decider, "n_stored", 0)),
    }


def _default_focus(record: Record, labels: Sequence[str]) -> tuple[str, str]:
    """The topic the view focuses on when none is asked for.

    The gold label whose **own words the text carries the most** (the map's
    topic-side measure, ``lex_label_coverage``): the topic the text is really
    about, and not the catch-all that happens to be listed first. The tie-break
    is the record's own label order, so the choice is deterministic.
    """
    candidates = [label for label in record.labels if label in labels]
    if not candidates:
        return (labels[0] if labels else ""), "the label space's first label"
    scores = [(label, cell_coverage(record.text, label)) for label in candidates]
    best = max(score for _, score in scores)
    focus = next(label for label, score in scores if score == best)
    return focus, "the gold label whose own words the text carries the most"


def render(
    context: ViewerContext,
    row: int,
    *,
    focus: str,
    focus_rule: str = "",
    picked: str = "",
    top_k: int,
    colour: bool,
    live: np.ndarray | None = None,
    live_calls: int = 0,
    live_info: Mapping[str, Any] | None = None,
    cache: Mapping[str, int] | None = None,
    what_if: Sequence[tuple[str, str, int, np.ndarray, int, Mapping[str, int]]] = (),
    what_if_baseline: np.ndarray | None = None,
    what_if_calls: int = 0,
    min_df: int = 0,
    problems: Sequence[str] = (),
) -> str:
    """The whole screen, as text (the tool's own output).

    Every block names **the question it answers** (CS-25), and every colour
    carries nothing the plain text does not: a colour is a reading of a number
    that is on the screen anyway, so ``--no-colour`` loses no information.
    """
    record = context.records[row]
    text = record.text
    labels = context.labels
    gold_labels = list(record.labels)
    text_ranking = lexical_ranking(text, labels)
    topic_ranking_rows = topic_ranking(text, labels)
    _arm_name, surrogate = surrogate_block(context.surrogate, None)
    coefficients = surrogate_coefficients(surrogate, focus)
    vector = surrogate_full_vector(surrogate)
    rows = word_rows(text, focus)
    maximum = max((float(entry["contribution"]) for entry in rows), default=0.0)
    # the topic's own name words (the topic side of the feature) — the words the
    # screen marks in cyan
    topic_name_words = topic_words(text, focus)
    # the text's own content words, in the text's order (the corpus map's unit)
    text_words: list[str] = []
    for word in content_tokens(text, DEFAULT_STOPWORDS):
        if word not in text_words:
            text_words.append(word)
    surrogate_max = max((abs(value) for value in coefficients.values()), default=0.0)
    corpus = corpus_block(context.corpus_map)
    lines: list[str] = []
    add = lines.append

    def label_cell(label: str, width: int = 0) -> str:
        """A label's name: a **gold** label in bold green, the rest plain.

        The padding is applied **before** the colour, so the columns line up in
        the coloured output as well (and the plain and the coloured screens
        carry the same text).
        """
        cell = f"{label:<{width}}" if width else label
        return paint(cell, GOLD, colour=colour) if label in gold_labels else cell

    add(paint("=" * 96, BOLD, colour=colour))
    add(
        paint(
            "HIT MAP VIEWER — ",
            BOLD,
            colour=colour,
        )
        + paint(f"UPR {context.split}", BOLD + CYAN, colour=colour)
        + paint(
            f" ({len(context.records)} texts, {len(labels)} labels, "
            f"{int(context.gold.sum())} gold cells)",
            BOLD,
            colour=colour,
        )
    )
    artifacts = f"{context.text_map_run}, {context.surrogate_run}"
    if context.corpus_map is not None:
        artifacts += f", {context.corpus_map_run}"
    add(
        paint(
            f"text: {record.id} (row {row} of {len(context.records)}) · picked by {picked}"
            f" · artifacts: {artifacts}",
            DIM,
            colour=colour,
        )
    )
    if problems:
        add(
            paint(
                "!! THE ARTIFACTS AND THE DATA DISAGREE — the numbers below may be stale",
                RED,
                colour=colour,
            )
        )
        for problem in problems:
            add(f"   - {problem}")
    add(paint("=" * 96, BOLD, colour=colour))
    add("")
    add(paint("THE TEXT", BOLD, colour=colour))
    add(f'  "{text}"')
    add("")
    add(paint("WHAT IS TRUE — the gold labels of this text (UPR val)", BOLD, colour=colour))
    add("  " + " · ".join(label_cell(label) for label in gold_labels))
    add("")
    add(paint(f'THE FOCUS TOPIC: "{focus}"', BOLD, colour=colour))
    if focus_rule:
        add(paint(f"  (chosen: {focus_rule})", DIM, colour=colour))
    add(
        "  the topic's own name words: "
        + (paint(", ".join(topic_name_words), CYAN, colour=colour) if topic_name_words else "—")
    )
    add("")

    # -- the feature's own split: the text side ---------------------------------
    add(
        paint(
            "THE FEATURE'S OWN SPLIT — a spelling question: how much of each word is in "
            "the topic's name?",
            BOLD,
            colour=colour,
        )
    )
    add(
        paint(
            "  an attribution: the feature's own arithmetic (lex_text_coverage), never the "
            "model's reading",
            DIM,
            colour=colour,
        )
    )
    add(
        paint(
            f"  {'word':<20}{'n':>2}   {'contribution':>11}   {'bar':<28}",
            DIM,
            colour=colour,
        )
    )
    for entry in rows:
        contribution = float(entry["contribution"])
        word = str(entry["word"])
        code = ramp_code(contribution, maximum, GREEN_RAMP)
        is_topic_word = word in topic_name_words
        add(
            "  "
            + paint(f"{word:<20}", CYAN if is_topic_word else code, colour=colour)
            + f"{int(entry['n_occurrences']):>2}   {contribution:>11.4f}   "
            + paint(f"{_bar(contribution, maximum):<28}", code, colour=colour)
            + paint("  ▲" if is_topic_word else "  ", DIM, colour=colour)
        )
    add(
        paint(
            "  ▲ = the word is one of the topic's own name words; the bar's shade is the "
            "contribution's size (the number says it too).",
            DIM,
            colour=colour,
        )
    )
    add("")

    # -- the model's word habits: the surrogate ---------------------------------
    add(
        paint(
            "THE MODEL'S WORD HABITS — a learned question: does the model react to this "
            "word for this topic?",
            BOLD,
            colour=colour,
        )
    )
    add(
        paint(
            "  an attribution of the model's own probability (a ridge surrogate), never a "
            "causal claim",
            DIM,
            colour=colour,
        )
    )
    add(
        paint(
            f"  {'word':<20}{'the surrogate says':>22}",
            DIM,
            colour=colour,
        )
    )
    for entry in rows:
        word = str(entry["word"])
        surrogate = coefficients.get(word)
        if surrogate is not None:
            # the rounding is at the display's precision, and a value that
            # rounds to zero never shows a sign ("-0.000" would be an artefact)
            shown = round(float(surrogate), 3) + 0.0
            cell = f"{shown:+.3f}"
            code = signed_ramp(shown, surrogate_max)
        elif vector is not None:
            # the word is outside the model's vocabulary (CS-24): a different
            # and more useful statement than "not stored"
            cell = "not in the model's vocabulary"
            code = DIM
        else:
            cell = "—"
            code = DIM
        add(
            "  "
            + paint(f"{word:<20}", DIM, colour=colour)
            + paint(f"{cell:>22}", code, colour=colour)
        )
    if vector is not None:
        add(
            paint(
                "  the surrogate's coefficient is the model's own for every word of its "
                "vocabulary (the train corpus's words with df ≥ 3); a word outside it reads "
                "'not in the model's vocabulary' — never zero.",
                DIM,
                colour=colour,
            )
        )
    else:
        add(
            paint(
                "  the surrogate's coefficient is stored for a topic's top words only: "
                "'—' means 'not stored', never zero.",
                DIM,
                colour=colour,
            )
        )
    add("")

    # -- the feature's own split: the topic side -------------------------------
    add(
        paint(
            "THE TOPIC'S OWN WORDS — a spelling question: how much of the topic's name is "
            "in the text?",
            BOLD,
            colour=colour,
        )
    )
    add(
        paint(
            "  the same feature's other side (lex_label_coverage): the topic's name words "
            "and how much of each the text has",
            DIM,
            colour=colour,
        )
    )

    def _topic_side(label: str, heading: str, *, header: bool = False) -> None:
        """One label's name words and how much of each the text has.

        The block asks the same question of every label it shows, so the
        helper is the block itself: the focus topic first, then the other gold
        labels of the text (CS-28).
        """
        add(paint(heading, DIM, colour=colour))
        if header:
            add(
                paint(
                    f"  {'word':<20}{'coverage':>11}   {'bar':<28}",
                    DIM,
                    colour=colour,
                )
            )
        contributions = list(label_word_contributions(text, label))
        maximum = max((item.coverage for item in contributions), default=0.0)
        for item in contributions:
            code = ramp_code(item.coverage, maximum, GREEN_RAMP)
            add(
                "    "
                + paint(f"{item.word:<20}", CYAN, colour=colour)
                + f"{item.coverage:>11.4f}   "
                + paint(f"{_bar(item.coverage, maximum):<28}", code, colour=colour)
            )

    _topic_side(
        focus,
        f'  the focus topic "{focus}" — the feature\'s value {cell_coverage(text, focus):.4f}',
        header=True,
    )
    other_gold = [label for label in gold_labels if label != focus]
    if other_gold:
        add(paint("  the other gold labels of this text:", DIM, colour=colour))
        for label in other_gold:
            _topic_side(
                label,
                f'  "{label}" — the feature\'s value {cell_coverage(text, label):.4f}',
            )
    add(
        paint(
            "  the topic's name words are in cyan; the bar's shade is the coverage's size "
            "(the number says it too).",
            DIM,
            colour=colour,
        )
    )
    add("")

    add(
        paint(
            "WHO THE WORDS SAY IT IS — a spelling question: which labels do the words point at?",
            BOLD,
            colour=colour,
        )
    )
    add(
        paint(
            "  the text's words against the topic's name (lex_text_coverage)",
            DIM,
            colour=colour,
        )
    )
    for rank, label, value in _ranking_rows_with_gold(text_ranking, top_k, gold_labels):
        mark = " ★" if label in gold_labels else "  "
        add(f"   {rank:>2}  {label_cell(label, 38)}{value:>9.4f}{mark}")
    add(
        paint(
            "  the topic's words against the text (lex_label_coverage)",
            DIM,
            colour=colour,
        )
    )
    for rank, label, value in _ranking_rows_with_gold(topic_ranking_rows, top_k, gold_labels):
        mark = " ★" if label in gold_labels else "  "
        add(f"   {rank:>2}  {label_cell(label, 38)}{value:>9.4f}{mark}")
    add(
        paint(
            "   (★ = a gold label of this text; the gold labels are always shown, even "
            "below the top-5)",
            DIM,
            colour=colour,
        )
    )
    add("")

    if corpus:
        add(
            paint(
                "THE CORPUS'S OWN WORDS FOR THIS TOPIC — a statistical question: which "
                "words go with this label in the training corpus?",
                BOLD,
                colour=colour,
            )
        )
        add(
            paint(
                "  a description of the data (the train split's own counts), never the "
                "model's reading",
                DIM,
                colour=colour,
            )
        )
        corpus_info = corpus.get("corpus", {})
        corpus_texts = corpus_info.get("n_texts", "?")
        corpus_words = corpus_info.get("n_words", "?")
        corpus_pairs = corpus_info.get("n_pairs", "?")
        add(
            paint(
                f"  the corpus: the {corpus_info.get('split', 'train')} split's "
                f"{corpus_texts} texts, {corpus_words} words, "
                f"{corpus_pairs} (label, word) pairs",
                DIM,
                colour=colour,
            )
        )

        def _corpus_side(label: str, heading: str, *, header: bool = False) -> None:
            """One label's corpus table: the words that go with it in the corpus.

            The block asks the same question of every label it shows, so the
            helper is the block itself: the focus topic first, then the other
            gold labels of the text (CS-28).
            """
            add(paint(heading, DIM, colour=colour))
            rows_all = corpus_topic_rows(context.corpus_map, label)
            rows = corpus_topic_rows(context.corpus_map, label, min_df=min_df)
            if not rows_all:
                add(
                    paint(
                        f"    {label!r} has no train example: its table is empty by construction",
                        DIM,
                        colour=colour,
                    )
                )
                return
            if not rows:
                add(
                    paint(
                        f"    the --min-df {min_df} filter hides every row of this label's "
                        f"table ({len(rows_all)} rows in the artifact)",
                        DIM,
                        colour=colour,
                    )
                )
                return
            if header:
                add(
                    paint(
                        f"    {'word':<20}{'n_with':>8}{'df':>6}{'lift':>9}{'log-odds':>11}",
                        DIM,
                        colour=colour,
                    )
                )
            log_odds_max = max((abs(float(row["log_odds"])) for row in rows), default=0.0)
            for row in rows[:top_k]:
                word = str(row["word"])
                in_text = word in text_words
                log_odds = float(row["log_odds"])
                code = signed_ramp(log_odds, log_odds_max)
                add(
                    "    "
                    + paint(
                        f"{word:<20}",
                        # the word that is in this text is bold on top of its
                        # ramp's shade, so the log-odds' size stays readable
                        (BOLD + code) if in_text else code,
                        colour=colour,
                    )
                    + paint(
                        f"{int(row['n_with']):>8}{int(row['df']):>6}"
                        f"{float(row['lift']):>9.2f}{log_odds:>11.3f}",
                        DIM if in_text else code,
                        colour=colour,
                    )
                    + paint("  ★ in this text" if in_text else "", BOLD, colour=colour)
                )

        _corpus_side(
            focus,
            f'  the focus topic "{focus}":',
            header=True,
        )
        corpus_other_gold = [label for label in gold_labels if label != focus]
        if corpus_other_gold:
            add(paint("  the other gold labels of this text:", DIM, colour=colour))
            for label in corpus_other_gold:
                _corpus_side(label, f'  "{label}":')
        add(
            paint(
                "  the rows are the corpus's own (by log-odds); the counts sit next to the "
                "number, so the noise is visible.",
                DIM,
                colour=colour,
            )
        )
        if min_df > 0:
            add(
                paint(
                    f"  the --min-df {min_df} filter is display-time: the artifact keeps "
                    "every pair.",
                    DIM,
                    colour=colour,
                )
            )
        add("")
        add(
            paint(
                "THIS TEXT'S WORDS AND THE LABELS THEY LEAN TO — the same association, "
                "read the other way",
                BOLD,
                colour=colour,
            )
        )
        add(
            paint(
                f"  {'word':<20}{'df':>6}   the labels it leans to (the corpus's log-odds)",
                DIM,
                colour=colour,
            )
        )
        for word in text_words:
            rows_for_word = corpus_word_rows(context.corpus_map, word, limit=CORPUS_WORD_LABELS)
            df = next(
                (
                    int(row.get("df", 0))
                    for row in corpus.get("per_word", [])
                    if row.get("word") == word
                ),
                0,
            )
            if not rows_for_word:
                add(
                    f"  {word:<20}{'—':>6}   "
                    + paint("not in the training corpus", DIM, colour=colour)
                )
                continue
            leans = " · ".join(
                f"{label_cell(str(entry['label']))} {float(entry['log_odds']):+.3f}"
                for entry in rows_for_word
            )
            below = min_df > 0 and df < min_df
            line = f"  {word:<20}{df:>6}   {leans}"
            if below:
                line += paint("  (below the --min-df filter)", DIM, colour=colour)
            add(paint(line, DIM, colour=colour) if below else line)
        add("")

    elif context.corpus_map is None:
        add(
            paint(
                f"THE CORPUS'S OWN WORDS — not shown: the run {context.corpus_map_run!r} "
                "has not been made (the map's block is in its hitmap.json)",
                DIM,
                colour=colour,
            )
        )
        add("")

    if live is not None:
        add(
            paint(
                "WHAT THE MODEL SAYS — the model's own reading: what does the model say "
                "this text is about?",
                BOLD,
                colour=colour,
            )
        )
        name = str((live_info or {}).get("model_id", "?"))
        digest = str((live_info or {}).get("digest", "?"))[:12]
        add(
            paint(
                f"  {name} (digest {digest}…) · the tool's own reading, not an artifact's number",
                DIM,
                colour=colour,
            )
        )
        order = np.argsort(-np.asarray(live, dtype=np.float64))
        live_max = max((float(live[int(column)]) for column in order[:top_k]), default=0.0)
        for rank, column in enumerate(order[:top_k], start=1):
            label = labels[int(column)]
            value = float(live[int(column)])
            mark = " ★" if label in gold_labels else "  "
            code = ramp_code(value, live_max, GREEN_RAMP)
            add(f"   {rank:>2}  {label_cell(label, 38)}{value:>9.4f}{mark}")
        add(paint(f"  the calls: {live_calls} {_cache_line(cache)}", DIM, colour=colour))
        add("")

    add(
        paint(
            "THE THREE SIDE BY SIDE — the comparison: what the words say, what the model "
            "says, what is true",
            BOLD,
            colour=colour,
        )
    )
    add(
        paint(
            f"  {'topic':<38}{'the words say':>15}{'the model says':>16}{'the truth':>10}",
            DIM,
            colour=colour,
        )
    )
    text_rank = {label: index + 1 for index, (label, _) in enumerate(text_ranking)}
    model_rank: dict[str, int] = {}
    if live is not None:
        order = np.argsort(-np.asarray(live, dtype=np.float64))
        model_rank = {labels[int(column)]: index + 1 for index, column in enumerate(order)}
    shown: list[str] = []
    for label in gold_labels:
        shown.append(label)
    for label, _ in text_ranking[:top_k]:
        if label not in shown:
            shown.append(label)
    for label in list(model_rank)[:top_k]:
        if label not in shown:
            shown.append(label)
    for label in shown:
        cells = [
            f"{text_rank[label]:>2} ({text_side_coverage(text, label):.4f})"
            if label in text_rank
            else "—",
            f"{model_rank[label]:>2} ({live[labels.index(label)]:.4f})"
            if live is not None and label in model_rank
            else "—",
            "gold" if label in gold_labels else "",
        ]
        add(f"  {label_cell(label, 38)}{cells[0]:>15}{cells[1]:>16}{cells[2]:>10}")
    add("")

    if what_if:
        add(
            paint(
                "WHAT IF THIS WORD IS REMOVED — the causal one, at word level",
                BOLD,
                colour=colour,
            )
        )
        add(
            paint(
                "  the model is asked again about the text without the word: this is the "
                "causal read (the blocks above are not)",
                DIM,
                colour=colour,
            )
        )
        for word, _variant, count, scores, calls, word_cache in what_if:
            add(
                paint(
                    f'  the word "{word}" occurs {count} time(s); the model is asked about the '
                    "text without it",
                    DIM,
                    colour=colour,
                )
            )
            assert what_if_baseline is not None
            tracked = list(dict.fromkeys([focus, *gold_labels]))
            add(
                paint(
                    f"  {'topic':<38}{'intact':>9}{'without':>10}{'delta':>10}",
                    DIM,
                    colour=colour,
                )
            )
            deltas = [
                float(scores[labels.index(label)]) - float(what_if_baseline[labels.index(label)])
                for label in tracked
            ]
            delta_max = max((abs(delta) for delta in deltas), default=0.0)
            for label, delta in zip(tracked, deltas, strict=True):
                column = labels.index(label)
                intact = float(what_if_baseline[column])
                without = float(scores[column])
                noise = abs(delta) <= NOISE_BAND
                code = DIM if noise else signed_ramp(delta, delta_max)
                add(
                    f"  {label_cell(label, 38)}{intact:>9.4f}{without:>10.4f}"
                    + paint(f"{delta:>10.4f}", code, colour=colour)
                    + paint("  noise" if noise else "", DIM, colour=colour)
                )
            add(
                paint(
                    f"  a delta inside the model's noise band (±{NOISE_BAND:g}) is marked "
                    '"noise": the model twitched, it did not use the word.',
                    DIM,
                    colour=colour,
                )
            )
            intact_order = int(np.argmax(what_if_baseline))
            without_order = int(np.argmax(scores))
            intact_top = labels[intact_order]
            without_top = labels[without_order]
            add(
                f"  the intact top-1: {intact_top} "
                f"({float(what_if_baseline[intact_order]):.4f})"
                f" -> the new top-1: {without_top} ({float(scores[without_order]):.4f})"
            )
            add(paint(f"  the calls: {calls} {_cache_line(word_cache)}", DIM, colour=colour))
            add("")
        add(
            paint(
                "  NOTE: the decision model is NOT deterministic (the CS-7a precedent): the same "
                "call re-run",
                YELLOW,
                colour=colour,
            )
        )
        add(
            paint(
                "  moves the probability by ~0.001-0.003, so a delta of that size is noise and "
                "not a signal.",
                YELLOW,
                colour=colour,
            )
        )
        add(paint(f"  the what-if calls in total: {what_if_calls}", DIM, colour=colour))
        add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """View one text of the split (the CLI's entry point)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="the config with the paths (default: configs/base.yaml)",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="the split to view (default: val)")
    parser.add_argument("--text", default=None, help="the record id of the text to view")
    parser.add_argument(
        "--sample", action="store_true", help="view a seeded random text of the split"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the seed of the drawn text")
    parser.add_argument(
        "--topic", default=None, help="the label to focus on (default: the first gold label)"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="how many labels the rankings show"
    )
    parser.add_argument(
        "--live", action="store_true", help="also show the decision model's own ranking"
    )
    parser.add_argument(
        "--what-if", default=None, help="the words to remove, comma separated (causal)"
    )
    parser.add_argument(
        "--arm", default=None, help="the decision arm to read (default: the artifact's first)"
    )
    parser.add_argument(
        "--max-questions-per-call",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="the questions one call may carry (the API's cap is 64)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="the results folder (default: the config's paths.results)",
    )
    parser.add_argument("--text-map-run", default=TEXT_MAP_RUN, help="the text-side map's run id")
    parser.add_argument("--surrogate-run", default=SURROGATE_RUN, help="the surrogate's run id")
    parser.add_argument(
        "--corpus-map-run",
        default=CORPUS_MAP_RUN,
        help="the corpus map's run id (CS-23; optional — the block is skipped when absent)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=0,
        help="the display-time filter: only the words with at least this df (0 = off)",
    )
    parser.add_argument("--no-colour", action="store_true", help="plain text (no ANSI colours)")
    parser.add_argument(
        "--colour",
        action="store_true",
        help="force the colours on (the default is 'on when stdout is a terminal')",
    )
    args = parser.parse_args(argv)

    colour = args.colour or (not args.no_colour and sys.stdout.isatty())
    try:
        config = load_config(args.config)
        results = args.results if args.results is not None else Path(config.paths.results)
        context = load_context(
            config,
            split=args.split,
            text_map_run=args.text_map_run,
            surrogate_run=args.surrogate_run,
            results=results,
            corpus_map_run=args.corpus_map_run,
        )
        row = pick_row(context.records, text_id=args.text, sample=args.sample, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"hitmap_view: {exc}", file=sys.stderr)
        return 1

    record = context.records[row]
    if args.topic:
        focus = args.topic
        focus_rule = "asked for"
    else:
        focus, focus_rule = _default_focus(record, context.labels)
    if focus not in context.labels:
        print(f"hitmap_view: {focus!r} is not a label of the split", file=sys.stderr)
        return 1

    problems = cross_check(context, row, record.text)
    if context.corpus_map is not None:
        text_words = list(dict.fromkeys(content_tokens(record.text, DEFAULT_STOPWORDS)))
        problems += corpus_cross_check(context, focus, text_words)
    picked = f"--text {args.text}" if args.text else f"--sample --seed {args.seed}"
    live: np.ndarray | None = None
    live_calls = 0
    live_info: dict[str, Any] = {}
    what_if: list[tuple[str, str, int, np.ndarray, int, Mapping[str, int]]] = []
    what_if_calls = 0
    baseline: np.ndarray | None = None
    decider: Any = None
    arm: DecisionArm | None = None
    cache: dict[str, int] = {}

    if args.live or args.what_if:
        try:
            decider, arm, _arm_name, live_info = build_decider(context, args.arm)
            live, live_calls = score_grid(
                decider,
                arm,
                record.text,
                context.labels,
                name_base=row * len(context.labels),
                max_questions_per_call=args.max_questions_per_call,
            )
            baseline = live
            cache = _cache_counters(decider)
        except (RuntimeError, ValueError) as exc:
            print(f"hitmap_view: {exc}", file=sys.stderr)
            return 1

    if args.what_if:
        assert decider is not None and arm is not None
        for word in [item.strip() for item in args.what_if.split(",") if item.strip()]:
            try:
                variant, count = remove_word(record.text, word)
            except ValueError as exc:
                print(f"hitmap_view: {exc}", file=sys.stderr)
                return 1
            before = _cache_counters(decider)
            scores, calls = score_grid(
                decider,
                arm,
                variant,
                context.labels,
                name_base=row * len(context.labels),
                max_questions_per_call=args.max_questions_per_call,
            )
            after = _cache_counters(decider)
            what_if.append(
                (
                    word,
                    variant,
                    count,
                    scores,
                    calls,
                    {key: after[key] - before.get(key, 0) for key in after},
                )
            )
            what_if_calls += calls

    try:
        print(
            render(
                context,
                row,
                focus=focus,
                focus_rule=focus_rule,
                picked=picked,
                top_k=args.top_k,
                colour=colour,
                live=live if args.live else None,
                live_calls=live_calls,
                live_info=live_info,
                cache=cache,
                what_if=what_if,
                what_if_baseline=baseline,
                what_if_calls=what_if_calls,
                min_df=args.min_df,
                problems=problems,
            )
        )
    except KeyError as exc:
        print(f"hitmap_view: the artifact is missing a block: {exc}", file=sys.stderr)
        return 1
    # a stale artifact is a loud failure: the screen is still shown (it says so
    # at the top), but the exit code says the numbers are not to be trusted
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
