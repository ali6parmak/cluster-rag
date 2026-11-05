"""The question map as a **pair scorer** (the track's CS-49).

The map (CS-48) is a description; this module turns it into a score for one
cell (a paragraph, a question): the sum of the question's map weights over the
paragraph's words. The arms are CS-30's four, ported:

- ``sum_positive`` — the sum of the **positive** log-odds of the paragraph's
  words in the question's table (the "evidence **for**" score, the arm the UPR
  fit side selected in CS-30). It is the pre-registered primary arm here, fixed
  **a priori** by the UPR prior, never selected on this run's numbers;
- ``sum_logodds`` — the signed sum;
- ``max_logodds`` — the strongest single word;
- ``support_count`` — how many of the question's table words the paragraph
  contains (the measure-free control).

**The empty table.** A question with no positive pair in a map has an empty
table: its cells carry the row's floor (the row's smallest seen score minus
one), so they can never be ranked above a question the map speaks about — the
track's rule, and the reason CS-48's availability readout matters.

**The cross-fit.** The folds are over the reviewed pairs (the same scheme and
seed as CS-48); each fold's map is built from the other folds and scores the
**whole grid**, and a judged cell takes its own fold's map's score. The
unjudged cells of the grid (the control's, and the other candidates' rank
readouts) take the **full** reviewed-set map's score — the control never enters
any fit, so the full map is the map that is strictly out of it; the vintage
difference (5/5 of the pairs for the control, 4/5 for the judged cells) is
recorded, not hidden.

**The label-blind counterpart.** The same pipeline with the **questions
permuted across the pairs** (the per-question counts preserved): the map is
built from the permuted labels but the grid is still read over the real
question columns, so an arm that beats the blind one is reading label
information, not the shape of the questionnaire.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.text_processors.features.pair_features import (
    DEFAULT_STOPWORDS,
    content_tokens,
)
from cluster_rag.text_processors.features.question_map import QuestionMap, build_question_map

# CS-30's four scoring functions, in the UPR report's order (the primary arm is
# named explicitly by CS-49's pre-registration; the order is the tie-break).
SCORING_FUNCTIONS: tuple[str, ...] = (
    "sum_logodds",
    "sum_positive",
    "max_logodds",
    "support_count",
    # CS-50's estimator arms: the z-scored weights (the informative prior
    # over the standard error), the signed sum and the "evidence for" sum.
    "sum_zscore",
    "sum_zscore_positive",
)

# The arm the UPR fit side selected in CS-30 (the "evidence for" score): the
# pre-registered primary arm of CS-49.
PRIMARY_SCORING = "sum_positive"

# The **union** arm (CS-53): the word map's z-scored score plus the phrase
# map's own — CS-39's composition, ported. It is not one of the scoring
# functions above (it needs **two** maps), so the caller composes it through
# :func:`union_grids` and the name lives here, beside the others.
UNION_SCORING = "sum_zscore_positive_phrase"

# What each arm does: the aggregation, the measure it reads and whether it
# keeps only the words whose value is **positive**.
_ARMS: dict[str, tuple[str, str, bool]] = {
    "sum_logodds": ("sum", "log_odds", False),
    "sum_positive": ("sum", "log_odds", True),
    "max_logodds": ("max", "log_odds", False),
    "support_count": ("count", "", False),
    "sum_zscore": ("sum", "z_score", False),
    "sum_zscore_positive": ("sum", "z_score", True),
}

# The gap between the empty tables' floor and the row's smallest seen score
# (the same rule as the UPR coefficient's `score_cells`).
FLOOR_GAP = 1.0

# CS-51's class restrictions: the name the config carries -> the classes the
# map keeps. ``nogeneric`` drops the register; ``distinctive`` keeps only the
# topic's own vocabulary.
CLASS_RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "nogeneric": ("distinctive", "shared"),
    "distinctive": ("distinctive",),
}

# CS-52's POS restrictions: the name the config carries -> the grammatical
# classes the map keeps.
POS_RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "noun": ("noun",),
    "verb": ("verb",),
}


def _allowed_classes(restrict: Sequence[str] | None) -> frozenset[str] | None:
    """The classes a restriction keeps (``None`` = every class).

    Raises:
        ValueError: On an unknown restriction or an empty one.
    """
    if restrict is None:
        return None
    if not restrict:
        raise ValueError("a class restriction must keep at least one class")
    unknown = [name for name in restrict if name not in CLASS_RESTRICTIONS]
    if unknown:
        raise ValueError(
            f"Unknown class restriction(s) {unknown}; expected some of {list(CLASS_RESTRICTIONS)}"
        )
    kept: set[str] = set()
    for name in restrict:
        kept.update(CLASS_RESTRICTIONS[name])
    return frozenset(kept)


def _allowed_pos(restrict_pos: Sequence[str] | None) -> frozenset[str] | None:
    """The grammatical classes a POS restriction keeps (``None`` = every class).

    Raises:
        ValueError: On an unknown restriction or an empty one.
    """
    if restrict_pos is None:
        return None
    if not restrict_pos:
        raise ValueError("a POS restriction must keep at least one class")
    unknown = [name for name in restrict_pos if name not in POS_RESTRICTIONS]
    if unknown:
        raise ValueError(
            f"Unknown POS restriction(s) {unknown}; expected some of {list(POS_RESTRICTIONS)}"
        )
    kept: set[str] = set()
    for name in restrict_pos:
        kept.update(POS_RESTRICTIONS[name])
    return frozenset(kept)


def _table_is_empty(
    question_map: QuestionMap,
    question: str,
    allowed: frozenset[str] | None,
    allowed_pos: frozenset[str] | None = None,
) -> bool:
    """Whether a question's table has no row the restriction(s) keep."""
    rows = question_map.per_question.get(question) or ()
    if not rows:
        return True
    if allowed is None and allowed_pos is None:
        return False
    for row in rows:
        if allowed is not None and question_map.word_classes.get(row.word) not in allowed:
            continue
        if allowed_pos is not None and question_map.pos_classes.get(row.word) not in allowed_pos:
            continue
        return False
    return True


def _index_map(values: tuple[str, ...]) -> dict[str, int]:
    """A text -> position map (the grid's rows and columns)."""
    return {value: index for index, value in enumerate(values)}


def score_grid(
    paragraph_texts: np.ndarray | list[str] | tuple[str, ...],
    questions: tuple[str, ...],
    question_map: QuestionMap,
    *,
    scoring: str = PRIMARY_SCORING,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    units: object = None,
    restrict: Sequence[str] | None = None,
    restrict_pos: Sequence[str] | None = None,
) -> np.ndarray:
    """Score every (paragraph, question) cell from the map's own weights.

    Args:
        paragraph_texts: The grid's paragraph rows (the texts being scored).
        questions: The grid's question columns (the map's question space, in
            the grid's own order).
        question_map: The map (built on the reviewed pairs of one fold).
        scoring: Which of the four arms to apply (see :data:`SCORING_FUNCTIONS`).
        stopwords: The tokenizer's stopwords — the map's own, so the words
            match the tables.
        units: The map's unit (the callable that turns a paragraph into the
            units its tables are keyed by; the default is the content-word
            tokenizer).
        restrict: The **class restriction** (CS-51): the restriction names whose
            kept classes only contribute (``None`` keeps every class).
        restrict_pos: The **POS restriction** (CS-52): the same rule on the
            grammatical axis (``None`` keeps every grammatical class).

    Returns:
        A ``(n_paragraphs, n_questions)`` float64 matrix, higher = more
        likely. A question with an **empty** table (under the restriction, if
        one is given) carries the row's floor
        (the row's smallest seen score minus one), so it can never be ranked
        above a question the map speaks about.

    Raises:
        ValueError: On an unknown scoring function.
    """
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring function {scoring!r}; expected one of {SCORING_FUNCTIONS}"
        )
    aggregation, measure, positive_only = _ARMS[scoring]
    allowed = _allowed_classes(restrict)
    allowed_pos = _allowed_pos(restrict_pos)
    units_of = units or (lambda text: content_tokens(text, stopwords))

    columns = tuple(questions)
    column_of = _index_map(columns)
    words = tuple(question_map.per_word)
    word_of = _index_map(words)
    n_paragraphs = len(paragraph_texts)
    n_questions = len(columns)
    weights = np.zeros((len(words), n_questions), dtype=np.float64)
    present = np.zeros((len(words), n_questions), dtype=bool)
    for word_index, word in enumerate(words):
        if allowed is not None and question_map.word_classes.get(word) not in allowed:
            continue
        if allowed_pos is not None and question_map.pos_classes.get(word) not in allowed_pos:
            continue
        for pair in question_map.per_word[word]:
            column = column_of.get(pair.question)
            if column is None:
                continue
            present[word_index, column] = True
            if scoring == "support_count":
                weights[word_index, column] = 1.0
            elif measure == "z_score":
                weights[word_index, column] = pair.z_score
            else:
                weights[word_index, column] = pair.log_odds

    grid = np.zeros((n_paragraphs, n_questions), dtype=np.float64)
    for row, text in enumerate(paragraph_texts):
        # Sorted, not set order: Python's string hashing is randomized per
        # process, so a set's iteration order would change the floating-point
        # summation order (and the last digit with it) between runs. The grid
        # must reproduce field for field.
        indexes = sorted({word_of[token] for token in units_of(text) if token in word_of})
        if not indexes:
            continue
        if aggregation == "count":
            grid[row] = present[indexes].sum(axis=0)
        elif aggregation == "max":
            grid[row] = weights[indexes].max(axis=0)
        else:
            values = weights[indexes]
            if positive_only:
                values = np.where(values > 0.0, values, 0.0)
            grid[row] = values.sum(axis=0)

    empty_columns = [
        column
        for column, question in enumerate(columns)
        if _table_is_empty(question_map, question, allowed, allowed_pos)
    ]
    if empty_columns:
        seen = np.delete(grid, empty_columns, axis=1)
        floor = (
            (seen.min(axis=1) - FLOOR_GAP) if seen.shape[1] else np.full(n_paragraphs, -FLOOR_GAP)
        )
        grid[:, empty_columns] = floor[:, None]
    return grid


def cross_fit_grids(
    paragraph_texts: list[str] | tuple[str, ...],
    question_texts: list[str] | tuple[str, ...],
    labels: list[int] | tuple[int, ...],
    fold_ids: np.ndarray | list[int] | tuple[int, ...],
    *,
    paragraphs: tuple[str, ...] | None = None,
    questions: tuple[str, ...] | None = None,
    scorings: tuple[str, ...] = SCORING_FUNCTIONS,
    stopwords: frozenset[str] = DEFAULT_STOPWORDS,
    units: object = None,
    probe_cells: Sequence[tuple[int, int]] | None = None,
    restrict: Sequence[str] | None = None,
    restrict_pos: Sequence[str] | None = None,
    word_pos: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The out-of-fold score grids: every judged cell from a map that never saw it.

    Args:
        paragraph_texts: Each reviewed pair's paragraph (the grid's row side).
        question_texts: Each reviewed pair's **label side** — the question the
            pair is associated with. For the label-blind counterpart the
            caller passes the **permuted** questions here (the counts
            preserved) while ``questions`` stays the real grid's columns.
        labels: Each reviewed pair's 0/1 label.
        fold_ids: Each pair's fold.
        paragraphs: The grid's paragraph rows (defaults to the pairs' distinct
            paragraphs, in first-occurrence order — the dataset view's own).
        questions: The grid's question columns (defaults to the label side's
            distinct texts; the blind arm must pass the real columns).
        scorings: Which arms to compute.
        stopwords: The tokenizer's stopwords.
        units: The map's unit.
        probe_cells: The unjudged cells whose score level is recorded per fold
            (the control cells: the per-fold mean beside the full-map mean is
            the vintage sensitivity check).
        restrict: The **class restriction** (CS-51), riding into every fold's
            grid and into the floor flags (a table the filter empties carries
            the floor).
        restrict_pos: The **POS restriction** (CS-52), the same rule on the
            grammatical axis. It needs ``word_pos`` (the fold maps are rebuilt
            here, so the classes must ride along or every table would empty).
        word_pos: The words' grammatical classes (CS-52), when the paragraphs
            were tagged; passed to the full map and to every fold's map.

    Returns:
        A dict with the grid's ``paragraphs`` / ``questions``, one assembled
        matrix per scoring (judged cells out-of-fold, the rest from the full
        map), the per-pair ``floor`` flags (the pair's fold map has no table
        for its question), the ``fold_ids`` and (when ``probe_cells`` is given)
        the probe cells' per-fold and full-map means.

    Raises:
        ValueError: On unknown scorings or a fold split with fewer than two
            folds.
    """
    if not (len(paragraph_texts) == len(question_texts) == len(labels) == len(fold_ids)):
        raise ValueError("the paragraphs, questions, labels and fold ids must have equal length")
    unknown = [name for name in scorings if name not in SCORING_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown scoring function(s) {unknown}; expected {SCORING_FUNCTIONS}")
    labels_list = [int(label) for label in labels]
    if any(label not in (0, 1) for label in labels_list):
        raise ValueError("the pair labels must be 0/1")
    fold_values = sorted({int(value) for value in fold_ids})
    if len(fold_values) < 2:
        raise ValueError(f"a cross-fit needs at least two folds, got {fold_values}")

    grid_paragraphs = (
        paragraphs if paragraphs is not None else tuple(dict.fromkeys(paragraph_texts))
    )
    grid_questions = questions if questions is not None else tuple(dict.fromkeys(question_texts))
    paragraph_of = _index_map(grid_paragraphs)
    question_of = _index_map(grid_questions)
    rows = np.asarray([paragraph_of[text] for text in paragraph_texts], dtype=np.int64)
    columns = np.asarray([question_of[text] for text in question_texts], dtype=np.int64)

    full_map = build_question_map(
        list(paragraph_texts),
        list(question_texts),
        labels_list,
        stopwords=stopwords,
        units=units,
        word_pos=word_pos,
    )
    fold_maps: dict[int, QuestionMap] = {}
    for fold in fold_values:
        keep = [index for index in range(len(labels_list)) if int(fold_ids[index]) != fold]
        fold_maps[fold] = build_question_map(
            [paragraph_texts[index] for index in keep],
            [question_texts[index] for index in keep],
            [labels_list[index] for index in keep],
            stopwords=stopwords,
            units=units,
            word_pos=word_pos,
        )

    allowed = _allowed_classes(restrict)
    allowed_pos = _allowed_pos(restrict_pos)
    floor = np.asarray(
        [
            _table_is_empty(
                fold_maps[int(fold_ids[index])],
                question_texts[index],
                allowed,
                allowed_pos,
            )
            for index in range(len(labels_list))
        ],
        dtype=bool,
    )
    matrices: dict[str, np.ndarray] = {}
    probe: dict[str, Any] = {}
    for scoring in scorings:
        full_grid = score_grid(
            grid_paragraphs,
            grid_questions,
            full_map,
            scoring=scoring,
            stopwords=stopwords,
            units=units,
            restrict=restrict,
            restrict_pos=restrict_pos,
        )
        fold_grids = {
            fold: score_grid(
                grid_paragraphs,
                grid_questions,
                fold_maps[fold],
                scoring=scoring,
                stopwords=stopwords,
                units=units,
                restrict=restrict,
                restrict_pos=restrict_pos,
            )
            for fold in fold_values
        }
        assembled = full_grid.copy()
        for index in range(len(labels_list)):
            assembled[rows[index], columns[index]] = fold_grids[int(fold_ids[index])][
                rows[index], columns[index]
            ]
        matrices[scoring] = assembled
        if probe_cells is not None:
            cells = [(int(row), int(column)) for row, column in probe_cells]
            probe[scoring] = {
                "full_map_mean": (
                    float(np.mean([full_grid[row, column] for row, column in cells]))
                    if cells
                    else None
                ),
                "per_fold_mean": {
                    str(fold): (
                        float(np.mean([fold_grids[fold][row, column] for row, column in cells]))
                        if cells
                        else None
                    )
                    for fold in fold_values
                },
            }
    return {
        "paragraphs": grid_paragraphs,
        "questions": grid_questions,
        "scorings": matrices,
        "floor": floor,
        "fold_ids": np.asarray([int(value) for value in fold_ids], dtype=np.int64),
        "probe": probe,
        "note": (
            "each judged cell is scored by the map of the fold that did NOT contain "
            "it; the unjudged cells (the control's and the other candidates') carry "
            "the full reviewed-set map's score, because the control never entered "
            "any fit (the vintage difference is recorded: 4/5 of the pairs for the "
            "judged cells, 5/5 for the unjudged ones)"
        ),
    }


def blind_question_permutation(n_pairs: int, seed: int) -> np.ndarray:
    """The label-blind counterpart's permutation (the questions among the pairs).

    The permutation is drawn from ``default_rng([seed, 1])`` — a stream
    independent of the cross-fit's own ``default_rng(seed)`` — and it is a
    permutation, so every question keeps exactly its number of reviewed pairs
    and the blind map is the same *shape* as the real one.

    Args:
        n_pairs: The number of reviewed pairs.
        seed: The cross-fit's seed (the stream is derived from it).

    Returns:
        The permutation: pair ``i`` carries question ``permutation[i]`` in the
        blind arm.
    """
    return np.random.default_rng([int(seed), 1]).permutation(int(n_pairs))


def union_grids(
    word_grids: Mapping[str, Any],
    phrase_grids: Mapping[str, Any],
    *,
    scoring: str = PRIMARY_SCORING,
) -> dict[str, Any]:
    """The **union** arm (CS-53): the word map's score plus the phrase map's own.

    CS-39's composition, ported cell for cell: both halves are the same scoring
    function (:data:`PRIMARY_SCORING`'s z-scored "evidence for" sum in the
    registered run), each computed from its own map behind the **same folds**,
    and the union is their sum. No rescaling and no re-ranking: the union's
    number is what a reader gets by adding the two arms, which is what makes
    the registered increment test readable.

    Args:
        word_grids: The word map's cross-fit grids (a :func:`cross_fit_grids`
            output).
        phrase_grids: The phrase map's grids (the same output, built with
            ``units=phrase_units``).
        scoring: Which arm the union sums.

    Returns:
        The union's cells, its halves' cells, both halves' floor flags (their
        union is the union arm's floor) and the two count arms when the caller
        computed them.

    Raises:
        ValueError: On an unknown scoring function, or when a grids mapping
            does not carry the requested scoring.
    """
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(
            f"Unknown scoring function {scoring!r}; expected one of {SCORING_FUNCTIONS}"
        )
    for name, grids in (("word", word_grids), ("phrase", phrase_grids)):
        if scoring not in grids["scorings"]:
            raise ValueError(
                f"the {name} grids do not carry the scoring {scoring!r}; "
                f"they carry {sorted(grids['scorings'])}"
            )
    word = np.asarray(word_grids["scorings"][scoring], dtype=np.float64)
    phrase = np.asarray(phrase_grids["scorings"][scoring], dtype=np.float64)
    if word.shape != phrase.shape:
        raise ValueError(
            f"the union's two halves must be the same grid, got {word.shape} and {phrase.shape}"
        )
    word_floor = np.asarray(word_grids["floor"], dtype=bool)
    phrase_floor = np.asarray(phrase_grids["floor"], dtype=bool)
    if word_floor.shape != phrase_floor.shape:
        raise ValueError(
            "the union's two halves must have the same floor shape, got "
            f"{word_floor.shape} and {phrase_floor.shape}"
        )

    def _count(grids: Mapping[str, Any]) -> np.ndarray | None:
        matrix = grids["scorings"].get("support_count")
        return None if matrix is None else np.asarray(matrix, dtype=np.float64)

    return {
        "scoring": scoring,
        "union": word + phrase,
        "word": word,
        "phrase": phrase,
        "word_floor": word_floor,
        "phrase_floor": phrase_floor,
        "word_count": _count(word_grids),
        "phrase_count": _count(phrase_grids),
        "note": (
            "the union is the word map's score plus the phrase map's own, both "
            "out-of-fold behind the same folds (CS-39's composition, ported); "
            "the union's floor is the union of the two halves' floor flags"
        ),
    }
