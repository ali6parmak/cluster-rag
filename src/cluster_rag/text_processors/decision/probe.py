"""The decision probe's pure core: the cells, the competence control, the measurement.

The track's Stage 3.5 (TEXT_PROCESSING_PLAN.md section 3) asks one question:
**is the reviewed set learnable at all?** Every embedder sits at chance on it
(AUC 0.468 / 0.500 / 0.537) and the reviewed set is **rank-selected** (exp18's
model chose which pairs got reviewed), so those numbers may be measuring the
selection rather than the labels. A **decision model** is the first
*independent* ranker on this benchmark.

This module is the pure side of the probe:

- :func:`build_scored_cells` — the cells to score (the judged pairs and the
  unjudged-proxy control, or the whole same-document grid);
- :func:`build_competence_cells` — the **synthetic competence control**: pairs
  that share a distinctive phrase (positives) against pairs that share none
  (negatives). It separates *"the data is at the ceiling"* from *"the model is
  the wrong shape"*, which a chance-level result on the reviewed set cannot;
- :func:`measure_decision_arm` — the measurement blocks, in the **same shape**
  as the embedding arms' blocks (the pair AUC, the within/cross decomposition,
  the score distributions, the rank percentiles, the hit@k, the top-1
  provenance) plus the decision-specific **calibration** block.

**The two negative subsets are never pooled**: the reviewer-verified hard
negatives (the judged pairs' label 0) and the unjudged-proxy control cells are
separate groups everywhere, and every number names its group.

Pure core (AGENTS.md dependency direction): numpy and the standard library
only — no model client, no I/O, no global state.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.eval.metrics import grouped_auc_decomposition, roc_auc
from cluster_rag.text_processors.calibration import reliability_diagram
from cluster_rag.text_processors.control import percentiles_from_positions, rank_positions
from cluster_rag.text_processors.decision.questions import (
    QuestionSpec,
    build_question_spec,
    render_instructions,
)

# The two state shapes of the prompt grid.
STATE_PAIR = "pair"
STATE_PARAGRAPH = "paragraph"
STATE_SHAPES: tuple[str, ...] = (STATE_PAIR, STATE_PARAGRAPH)

# The cell groups: the reviewed set, the unjudged-proxy control and the
# synthetic competence control.
GROUP_JUDGED = "judged"
GROUP_CONTROL = "control"
GROUP_UNJUDGED = "unjudged"
GROUP_COMPETENCE = "competence"

# The hit@k grid (the same one the embedding arms report).
HIT_KS: tuple[int, ...] = (1, 10)

# The quantiles of the reported distributions (the probe's grid).
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# The competence control's question template: the quoted phrase is the only
# difference between a positive and a negative pair.
COMPETENCE_QUESTION_TEMPLATE = 'Does the document discuss "{phrase}"?'

# The competence control's phrase length (in words) and the shortest token it
# accepts (the shorter words are the glue of a sentence, not a distinctive
# phrase).
COMPETENCE_PHRASE_WORDS = 4
COMPETENCE_MIN_TOKEN_LENGTH = 3

_TOKEN_PATTERN = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class DecisionCell:
    """One cell to score: a (paragraph, question) pair and the cell's identity.

    ``group`` names the cell's subset (see the ``GROUP_*`` constants), ``label``
    is the reviewed set's label (``None`` for the unjudged cells, which are
    *presumed* negatives at an unmeasured rate, never ground truth).
    """

    cell_id: str
    group: str
    document: str
    paragraph_row: int
    question_col: int
    paragraph_text: str
    question_text: str
    label: int | None
    paragraph_number: str = ""
    question_number: int = 0


def _distribution(values: np.ndarray) -> dict[str, Any]:
    """Count + mean/min/max + the quantiles of one score group (empty is safe)."""
    block: dict[str, Any] = {"n": int(values.size)}
    if values.size:
        block.update(
            {
                "mean": float(np.mean(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    f"{p:g}": float(np.quantile(values, p)) for p in DISTRIBUTION_QUANTILES
                },
            }
        )
    return block


def _hit_at_k(ranks: np.ndarray, hit_ks: Sequence[int]) -> dict[str, float]:
    """The share of a group's cells whose question is among the top-k candidates."""
    return {f"hit@{k}": float((ranks < k).mean()) if ranks.size else 0.0 for k in hit_ks}


def _top_quintile_share(percentiles: np.ndarray) -> float | None:
    """The share of a group in the top fifth of its paragraph's candidate list."""
    if not percentiles.size:
        return None
    return float((percentiles < 0.2).mean())


def _macro(values: Mapping[str, float | None]) -> float | None:
    """The macro average of the per-document values (None when none exists)."""
    present = [value for value in values.values() if value is not None]
    return float(np.mean(present)) if present else None


def _tokens(text: str) -> list[str]:
    """The paragraph's alphabetic words of at least ``COMPETENCE_MIN_TOKEN_LENGTH``."""
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) >= COMPETENCE_MIN_TOKEN_LENGTH
    ]


def _phrase_candidates(tokens: Sequence[str], size: int) -> list[tuple[int, tuple[str, ...]]]:
    """The word n-grams of ``tokens`` (the contiguous windows of ``size`` words)."""
    if len(tokens) < size:
        return []
    return [(index, tuple(tokens[index : index + size])) for index in range(len(tokens) - size + 1)]


def distinctive_phrase(
    paragraph: str, token_frequencies: Mapping[str, int], *, size: int = COMPETENCE_PHRASE_WORDS
) -> str | None:
    """A phrase of the paragraph that no other paragraph contains.

    The phrase is the window of ``size`` words whose words are the **rarest** in
    the corpus (the sum of the words' paragraph frequencies is minimal), with
    the earliest window winning a tie. Rarity is what makes a phrase
    *distinctive*: a generic four-word window ("the committee recalls the")
    would make the competence control trivial, a garbled one would make it
    meaningless, so the rarest window is the fairest choice.

    Args:
        paragraph: The paragraph's text.
        token_frequencies: How many paragraphs contain each word.
        size: The phrase's length in words.

    Returns:
        The phrase, or ``None`` when the paragraph is too short.
    """
    tokens = _tokens(paragraph)
    candidates = _phrase_candidates(tokens, size)
    if not candidates:
        return None
    best: tuple[int, int, tuple[str, ...]] | None = None
    for index, phrase in candidates:
        rarity = sum(token_frequencies.get(token, 0) for token in phrase)
        if best is None or rarity < best[0]:
            best = (rarity, index, phrase)
    assert best is not None  # the candidate list is non-empty here
    return " ".join(best[2])


def _token_frequencies(paragraphs: Sequence[str]) -> dict[str, int]:
    """How many paragraphs contain each word (over the corpus's paragraphs)."""
    frequencies: dict[str, int] = {}
    for paragraph in paragraphs:
        for token in set(_tokens(paragraph)):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def build_competence_cells(
    paragraphs: Sequence[str],
    *,
    n_pairs: int,
    document: str = "competence",
) -> tuple[DecisionCell, ...]:
    """The **synthetic competence control**: shared-phrase pairs against unrelated ones.

    For every selected paragraph, the control holds two pairs with the *same*
    question template and the *same* paragraph:

    - the **positive**: the question quotes a phrase of the paragraph (so the
      pair shares a distinctive phrase);
    - the **negative**: the question quotes a phrase of a *different* paragraph
      (so the pair shares none).

    A model that can detect a shared phrase at all separates the two; a model
    that cannot does not. The control therefore answers the question a
    chance-level result on the reviewed set cannot: *is the reviewed set at the
    ceiling, or is the model the wrong shape?*

    The construction is deterministic (the paragraphs are taken in order and
    the phrase is the rarest window), so the same corpus gives the same
    control.

    Args:
        paragraphs: The corpus's paragraph texts (in a stable order).
        n_pairs: How many positive/negative pairs to build.
        document: The document name the cells carry (the control is synthetic).

    Returns:
        The control's cells: the positives first, then the negatives.
    """
    if not paragraphs:
        return ()
    frequencies = _token_frequencies(paragraphs)
    phrases: list[str | None] = [
        distinctive_phrase(paragraph, frequencies) for paragraph in paragraphs
    ]
    cells: list[DecisionCell] = []
    for index, paragraph in enumerate(paragraphs):
        if len(cells) >= 2 * n_pairs:
            break
        phrase = phrases[index]
        if not phrase:
            continue
        other = _other_phrase(paragraph, phrase, paragraphs, phrases, index, frequencies)
        if other is None:
            continue
        cells.append(
            DecisionCell(
                cell_id=f"competence-pos-{index}",
                group=GROUP_COMPETENCE,
                document=document,
                paragraph_row=index,
                question_col=2 * index,
                paragraph_text=paragraph,
                question_text=COMPETENCE_QUESTION_TEMPLATE.format(phrase=phrase),
                label=1,
            )
        )
        cells.append(
            DecisionCell(
                cell_id=f"competence-neg-{index}",
                group=GROUP_COMPETENCE,
                document=document,
                paragraph_row=index,
                question_col=2 * index + 1,
                paragraph_text=paragraph,
                question_text=COMPETENCE_QUESTION_TEMPLATE.format(phrase=other),
                label=0,
            )
        )
    return tuple(cells)


def _contains_sequence(tokens: Sequence[str], needle: Sequence[str]) -> bool:
    """Whether the token sequence ``needle`` occurs inside ``tokens``."""
    if not needle or len(needle) > len(tokens):
        return False
    return any(
        tuple(tokens[index : index + len(needle)]) == tuple(needle)
        for index in range(len(tokens) - len(needle) + 1)
    )


def _other_phrase(
    paragraph: str,
    phrase: str,
    paragraphs: Sequence[str],
    phrases: Sequence[str | None],
    index: int,
    frequencies: Mapping[str, int],
) -> str | None:
    """A phrase of another paragraph that makes the pair **unrelated**.

    Two conditions, so the negative is unrelated in both senses the model can
    read: the phrase must not occur in the paragraph, and the two paragraphs
    must share no **distinctive** word (a word only they contain), so the
    negative is not about the same issue either.
    """
    tokens = _tokens(paragraph)
    mine = {token for token in set(tokens) if frequencies.get(token, 0) == 1}
    for offset, other in enumerate(phrases):
        if offset == index or not other:
            continue
        if _contains_sequence(tokens, _tokens(other)):
            continue
        theirs = {
            token for token in set(_tokens(paragraphs[offset])) if frequencies.get(token, 0) == 1
        }
        if mine & theirs:
            continue
        return other
    return None


@dataclass(frozen=True)
class DecisionCall:
    """One planned call: the state and one question spec per cell.

    ``cell_indexes`` are the cells' positions in the list the plan was built
    from, so the caller can put every answer back where it belongs.
    """

    state: dict[str, str]
    cell_indexes: tuple[int, ...]
    question_names: tuple[str, ...]
    specs: tuple[QuestionSpec, ...]


# The characters-per-token estimate of the input size. The server's limit is
# in tokens, the config's budget is in tokens too, and the planner has only
# the text: 3.5 characters per token is a deliberately **conservative**
# estimate (the measured ratio is 4.1 for laya and 4.8 for tev1 on this
# corpus), so a call the planner accepts fits the server's budget.
CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str) -> int:
    """The estimated token count of a text (the conservative chars/3.5 rule)."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _truncate_to_budget(text: str, budget_chars: int) -> str:
    """Cut a text to the given number of characters (never below empty)."""
    if budget_chars <= 0:
        return ""
    return text[:budget_chars]


def plan_calls(
    cells: Sequence[DecisionCell],
    *,
    state_shape: str,
    instructions: str,
    question_type: str,
    options: tuple[str, ...] = (),
    criteria: Mapping[str, str] | None = None,
    max_questions_per_call: int = 64,
    max_input_tokens: int | None = None,
) -> list[DecisionCall]:
    """Plan the calls of one arm: the state shape and the input budget decide the grouping.

    - the **pair** shape puts the pair in one state (``{"paragraph": ...,
      "question": ...}``) and asks one question per call;
    - the **paragraph** shape puts only the paragraph in the state
      (``{"paragraph": ...}``) and carries the question in the instructions,
      which is what allows many questions about the same paragraph in one
      call (the cheap shape of the grid).

    **The input budget.** A decision model has a hard context (measured:
    laya 512 tokens, tev1 ~2,500), and the corpus's longest paragraph and
    question do not fit in the smaller one. The planner therefore

    - groups the cells into calls whose **estimated** input fits the budget
      (``max_input_tokens``; ``None`` = no limit), and
    - **truncates the state's paragraph** when a single cell does not fit on
      its own — never the question and never the instructions, so the arm
      always asks the same question.

    The truncation is the caller's to record (the state's paragraph is shorter
    than the cell's), exactly like the embedding probe's "paragraphs over the
    context" diagnostic: a shortened input is never silent.

    Raises:
        ValueError: If the state shape is unknown, a call would carry more
            questions than the API accepts, or a cell's question alone exceeds
            the budget.
    """
    if state_shape not in STATE_SHAPES:
        raise ValueError(
            f"unknown decision state shape {state_shape!r}; expected one of {list(STATE_SHAPES)}"
        )
    if max_questions_per_call < 1:
        raise ValueError("a decision call must carry at least one question")

    def _instructions_for(cell: DecisionCell) -> str:
        return render_instructions(instructions, cell.question_text, shape=state_shape)

    def _state_for(cell: DecisionCell, paragraph: str) -> dict[str, str]:
        if state_shape == STATE_PAIR:
            return {"paragraph": paragraph, "question": cell.question_text}
        return {"paragraph": paragraph}

    calls: list[DecisionCall] = []
    if state_shape == STATE_PAIR:
        for index, cell in enumerate(cells):
            rendered = _instructions_for(cell)
            paragraph = cell.paragraph_text
            if max_input_tokens is not None:
                budget = max_input_tokens * CHARS_PER_TOKEN
                # the question and the instructions are never cut: only the
                # state's paragraph gives way
                fixed = len(rendered) + (
                    len(cell.question_text) if state_shape == STATE_PAIR else 0
                )
                paragraph = _truncate_to_budget(paragraph, int(budget - fixed))
            spec = build_question_spec(question_type, rendered, options=options, criteria=criteria)
            calls.append(
                DecisionCall(
                    state=_state_for(cell, paragraph),
                    cell_indexes=(index,),
                    question_names=(f"c{index}",),
                    specs=(spec,),
                )
            )
        return calls

    # the paragraph shape: group the cells by their paragraph, then chunk
    groups: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(cell.paragraph_text, []).append(index)
    for paragraph_text, indexes in groups.items():
        chunk: list[int] = []
        chunk_size = 0
        for index in indexes:
            cell = cells[index]
            rendered = _instructions_for(cell)
            size = estimate_tokens(rendered)
            if chunk and (
                len(chunk) >= max_questions_per_call
                or (
                    max_input_tokens is not None
                    and chunk_size + size + estimate_tokens(paragraph_text) > max_input_tokens
                )
            ):
                calls.append(
                    _paragraph_call(
                        cells,
                        chunk,
                        paragraph_text,
                        instructions,
                        question_type,
                        options,
                        criteria,
                        max_input_tokens,
                    )
                )
                chunk = []
                chunk_size = 0
            chunk.append(index)
            chunk_size += size
        if chunk:
            calls.append(
                _paragraph_call(
                    cells,
                    chunk,
                    paragraph_text,
                    instructions,
                    question_type,
                    options,
                    criteria,
                    max_input_tokens,
                )
            )
    return calls


def _paragraph_call(
    cells: Sequence[DecisionCell],
    chunk: Sequence[int],
    paragraph_text: str,
    instructions: str,
    question_type: str,
    options: tuple[str, ...],
    criteria: Mapping[str, str] | None,
    max_input_tokens: int | None,
) -> DecisionCall:
    """One call of the paragraph shape: the paragraph's state plus the chunk's questions."""
    specs = tuple(
        build_question_spec(
            question_type,
            render_instructions(instructions, cells[index].question_text, shape=STATE_PARAGRAPH),
            options=options,
            criteria=criteria,
        )
        for index in chunk
    )
    paragraph = paragraph_text
    if max_input_tokens is not None:
        budget = max_input_tokens * CHARS_PER_TOKEN
        fixed = sum(len(spec.instructions) for spec in specs)
        paragraph = _truncate_to_budget(paragraph, int(budget - fixed))
    return DecisionCall(
        state={"paragraph": paragraph},
        cell_indexes=tuple(chunk),
        question_names=tuple(f"c{index}" for index in chunk),
        specs=specs,
    )


def _candidate_columns(cells: Sequence[DecisionCell]) -> dict[int, tuple[int, ...]]:
    """Every paragraph's **scored** question columns (the restricted candidate list).

    The decision probe scores a *subset* of the grid, so a cell's rank is
    measured inside the cells the run actually scored for its paragraph — not
    inside the paragraph's full same-document question list (which needs the
    grid scope). The artifact says so wherever the number is reported.
    """
    columns: dict[int, list[int]] = {}
    for cell in cells:
        columns.setdefault(cell.paragraph_row, [])
        if cell.question_col not in columns[cell.paragraph_row]:
            columns[cell.paragraph_row].append(cell.question_col)
    return {row: tuple(sorted(cols)) for row, cols in columns.items()}


def measure_decision_arm(
    cells: Sequence[DecisionCell],
    probabilities: Sequence[float],
    *,
    ordinary_group: str = GROUP_CONTROL,
    bins: int = 10,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """Measure one decision arm: the AUCs, the distributions, the ranks, the calibration.

    The block's shape mirrors the embedding arms' block (``pair_auc``,
    ``pair_auc_decomposition``, ``score_distributions``, ``negative_enrichment``,
    ``hit_at_k``, ``top1_candidate_provenance``) so the numbers sit next to the
    embedding arms' in the same table, and it adds the decision-specific
    ``calibration`` block (the reliability diagram, the ECE and the Brier score
    — tev1's card says its calibration is untested, so it has to be measured).

    Args:
        cells: The scored cells (the judged pairs, the ordinary cells and — for
            the competence control — the synthetic pairs).
        probabilities: The arm's probability per cell, in the same order.
        ordinary_group: The group of the ordinary cells (the unjudged-proxy
            control, or the unjudged grid).
        bins: The reliability diagram's number of bins.
        hit_ks: The hit@k grid.

    Returns:
        The measurement block (JSON-serializable).
    """
    if len(cells) != len(probabilities):
        raise ValueError("the cells and the probabilities must have the same length")
    scores = np.asarray(probabilities, dtype=np.float64)
    groups: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(cell.group, []).append(index)

    judged = [index for index in groups.get(GROUP_JUDGED, []) if cells[index].label is not None]
    ordinary = groups.get(ordinary_group, [])
    labels = np.asarray([cells[index].label for index in judged], dtype=np.int64)
    pair_scores = scores[judged]
    documents = np.asarray([cells[index].document for index in judged])
    ordinary_scores = scores[ordinary] if ordinary else np.zeros(0, dtype=np.float64)

    pair_auc: dict[str, Any] = {"overall": None, "by_document": {}}
    if judged and (labels == 1).any() and (labels == 0).any():
        pair_auc["overall"] = roc_auc(labels.tolist(), pair_scores.tolist())
    for document in sorted(set(documents.tolist())):
        mask = documents == document
        try:
            pair_auc["by_document"][document] = roc_auc(
                labels[mask].tolist(), pair_scores[mask].tolist()
            )
        except ValueError:
            pair_auc["by_document"][document] = None
    pair_auc["macro"] = _macro(pair_auc["by_document"])

    decomposition: dict[str, Any]
    try:
        decomposition = grouped_auc_decomposition(
            labels.tolist(), pair_scores.tolist(), documents.tolist()
        )
    except ValueError as exc:
        decomposition = {"note": str(exc)}

    # the score matrix of the scored cells (the unscored cells stay NaN and are
    # never read: the candidate list is the restricted one)
    matrix = _score_matrix(cells, scores)
    candidate_columns = _candidate_columns(cells)
    # the rank helpers index the candidate list by the **row**, so the
    # restricted list is expanded to one entry per paragraph row
    restricted_by_row = [candidate_columns.get(row, ()) for row in range(matrix.shape[0])]
    rows = [cell.paragraph_row for cell in cells]
    columns = [cell.question_col for cell in cells]
    ranks = rank_positions(matrix, rows, columns, restricted_by_row)
    percentiles = percentiles_from_positions(ranks, rows, restricted_by_row)
    judged_ranks = ranks[judged]
    judged_percentiles = percentiles[judged]
    ordinary_ranks = ranks[ordinary] if ordinary else np.zeros(0, dtype=np.int64)
    ordinary_percentiles = percentiles[ordinary] if ordinary else np.zeros(0, dtype=np.float64)
    positive_mask = labels == 1
    negative_mask = labels == 0

    score_distributions: dict[str, Any] = {
        "label1": _distribution(pair_scores[positive_mask]),
        "label0": _distribution(pair_scores[negative_mask]),
        "control": _distribution(ordinary_scores),
        "by_document": {},
    }
    enrichment_by_document: dict[str, Any] = {}
    for document in sorted(set(documents.tolist())):
        document_mask = documents == document
        document_ordinary = [index for index in ordinary if cells[index].document == document]
        document_ordinary_scores = scores[document_ordinary] if document_ordinary else np.zeros(0)
        document_ordinary_percentiles = (
            percentiles[document_ordinary] if document_ordinary else np.zeros(0)
        )
        score_distributions["by_document"][document] = {
            "label1": _distribution(pair_scores[document_mask & positive_mask]),
            "label0": _distribution(pair_scores[document_mask & negative_mask]),
            "control": _distribution(document_ordinary_scores),
        }
        enrichment_by_document[document] = {
            "rank_percentile": {
                "label1": _distribution(judged_percentiles[document_mask & positive_mask]),
                "label0": _distribution(judged_percentiles[document_mask & negative_mask]),
                "control": _distribution(document_ordinary_percentiles),
            },
            "top_quintile_share": {
                "label1": _top_quintile_share(judged_percentiles[document_mask & positive_mask]),
                "label0": _top_quintile_share(judged_percentiles[document_mask & negative_mask]),
                "control": _top_quintile_share(document_ordinary_percentiles),
            },
        }

    calibration = _calibration_block(pair_scores, labels, documents, bins=bins)

    floor_gap: dict[str, Any] = {
        "positive_mean": float(pair_scores[positive_mask].mean()) if positive_mask.any() else None,
        "ordinary_mean": float(ordinary_scores.mean()) if ordinary_scores.size else None,
        "gap": (
            float(pair_scores[positive_mask].mean() - ordinary_scores.mean())
            if positive_mask.any() and ordinary_scores.size
            else None
        ),
        "ordinary_group": ordinary_group,
        "note": (
            "the verified positives' mean minus the ordinary cells' mean: the "
            "track's falsification statistic (does the method separate the "
            "reviewed positives from the ordinary cells?)"
        ),
    }

    return {
        "pair_auc": pair_auc,
        "pair_auc_decomposition": decomposition,
        "score_distributions": score_distributions,
        "negative_enrichment": {
            "rank_percentile": {
                "label1": _distribution(judged_percentiles[positive_mask]),
                "label0": _distribution(judged_percentiles[negative_mask]),
                "control": _distribution(ordinary_percentiles),
            },
            "by_document": enrichment_by_document,
            "top_quintile_share": {
                "label1": _top_quintile_share(judged_percentiles[positive_mask]),
                "label0": _top_quintile_share(judged_percentiles[negative_mask]),
                "control": _top_quintile_share(ordinary_percentiles),
            },
        },
        "hit_at_k": {
            "label1": _hit_at_k(judged_ranks[positive_mask], hit_ks),
            "label0": _hit_at_k(judged_ranks[negative_mask], hit_ks),
            "control": _hit_at_k(ordinary_ranks, hit_ks),
            "note": (
                "the share of a group's cells whose question is among the top-k "
                "candidates of its own paragraph — inside the **scored** cells of "
                "that paragraph, not its full same-document question list (the "
                "restricted candidate list; the full list needs the grid scope)"
            ),
        },
        "calibration": calibration,
        "floor_gap": floor_gap,
        "top1_candidate_provenance": _top1_provenance(cells, matrix, candidate_columns),
        "ordinary_group": ordinary_group,
        "n_cells": len(cells),
        "n_judged": len(judged),
        "n_ordinary": len(ordinary),
    }


def _score_matrix(cells: Sequence[DecisionCell], scores: np.ndarray) -> np.ndarray:
    """The scored cells' scores as a (paragraphs x questions) matrix (NaN elsewhere)."""
    if not cells:
        return np.zeros((0, 0), dtype=np.float64)
    n_rows = max(cell.paragraph_row for cell in cells) + 1
    n_columns = max(cell.question_col for cell in cells) + 1
    matrix = np.full((n_rows, n_columns), np.nan, dtype=np.float64)
    for index, cell in enumerate(cells):
        matrix[cell.paragraph_row, cell.question_col] = float(scores[index])
    return matrix


def _calibration_block(
    scores: np.ndarray, labels: np.ndarray, documents: np.ndarray, *, bins: int
) -> dict[str, Any]:
    """The reliability diagram of the arm's probability: pooled, per document, the macro."""
    pooled = reliability_diagram(scores, labels, n_bins=bins)
    by_document: dict[str, Any] = {}
    for document in sorted(set(documents.tolist())):
        mask = documents == document
        by_document[document] = reliability_diagram(scores[mask], labels[mask], n_bins=bins)
    macro_ece = _macro({document: block.get("ece") for document, block in by_document.items()})
    return {
        "pooled": pooled,
        "by_document": by_document,
        "macro_ece": macro_ece,
        "note": (
            "the reliability diagram of the arm's probability against the reviewed "
            "set's labels: the pooled ECE flatters every arm (it mixes three "
            "documents' calibrations), so the per-document macro sits beside it"
        ),
    }


def _top1_provenance(
    cells: Sequence[DecisionCell],
    matrix: np.ndarray,
    candidate_columns: Mapping[int, tuple[int, ...]],
) -> dict[str, Any]:
    """Where every paragraph's best **scored** candidate comes from.

    A model-dependent measurement (it says what *this model's* first choice for
    a paragraph is, intersected with the file's coverage), the same shape as
    the embedding arms' block.
    """
    labels_by_cell: dict[tuple[int, int], int] = {}
    for cell in cells:
        if cell.label is not None:
            labels_by_cell[(cell.paragraph_row, cell.question_col)] = cell.label
    counts = {"given_positive": 0, "given_negative": 0, "unjudged": 0}
    for row, candidates in candidate_columns.items():
        if not candidates:
            continue
        best = max(candidates, key=lambda col: float(matrix[row, col]))
        label = labels_by_cell.get((row, best))
        if label == 1:
            counts["given_positive"] += 1
        elif label == 0:
            counts["given_negative"] += 1
        else:
            counts["unjudged"] += 1
    return {
        "n_paragraphs": len(candidate_columns),
        **counts,
        "note": (
            "model-dependent measurement: this model's best scored candidate per "
            "paragraph, intersected with the file's coverage. It is not a dataset "
            "property and says nothing about paragraphs that have no match."
        ),
    }


def measure_competence_control(
    cells: Sequence[DecisionCell], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Measure the synthetic competence control: can the model see a shared phrase?

    The control's AUC is the answer to the question a chance-level result on
    the reviewed set cannot answer: *the data is at the ceiling* or *the model
    is the wrong shape*.
    """
    if len(cells) != len(probabilities):
        raise ValueError("the cells and the probabilities must have the same length")
    if not cells:
        return {"n_cells": 0, "note": "the competence control is empty"}
    labels = np.asarray([cell.label or 0 for cell in cells], dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    auc: float | None
    try:
        auc = roc_auc(labels.tolist(), scores.tolist())
    except ValueError as exc:
        auc = None
        note = str(exc)
    else:
        note = ""
    return {
        "n_cells": len(cells),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "pair_auc": auc,
        "positive": _distribution(positive),
        "negative": _distribution(negative),
        "note": (
            "the synthetic competence control: pairs that share a distinctive "
            "phrase against pairs that share none. A model that can detect a "
            "shared phrase separates the two; a chance-level AUC means the model "
            "is the wrong shape (not that the data is at the ceiling)."
            + (f" ({note})" if note else "")
        ),
    }
