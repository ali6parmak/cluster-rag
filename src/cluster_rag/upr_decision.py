"""The UPR decision probe's driver (``decide-upr``).

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-9) puts the **decision models** on
the UPR data as a **measurement**, not a pipeline component: a decision model
(Ollama's model *type*: a state plus typed questions, answered with
probabilities in one forward pass) answers the UPR-native question **"is this
text about *<label>*?"** per label, and the probe measures what its
probability buys.

The driver orchestrates:

- the model-independent part (the UPR view: the texts, the label space and the
  gold matrix) — built by :mod:`cluster_rag.upr_analysis`;
- the pure core (the question text, the cells, the **label-blind
  counterpart** and the measurement) — in
  :mod:`cluster_rag.text_processors.decision.upr_probe`, on top of the OMCT
  probe's shape-agnostic pieces (the call plan, the ``DecisionCell`` and the
  synthetic **competence control**);
- the model client (the ``/v1/systemone`` caller behind the
  :class:`~cluster_rag.decision.base.Decider` port) — wired in by the CLI.

**The scope.** The probe measures a **seeded sample** of the evaluated split's
texts by default (``upr_analysis.decision.scope: sample``): the full val grid
is ~44,880 cells, and the decision models cost one call per ~64 cells in the
``paragraph`` shape (so ~1,320 calls per model) and one call per cell in the
``pair`` shape (~44,880 calls). The sample keeps the probe inside a sane
budget; the full grid is the ``grid`` scope, and the artifact records which
scope produced the numbers.

**The honesty rule.** The non-gold labels' true positive rate is unmeasured,
so nothing here is a precision claim: the probe measures a *reach* and a
*cost*, and the **label-blind counterpart** is the yardstick that says how
much of the cost is the model's own label-blind signal.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.decision.base import Decider
from cluster_rag.omct_decision import DecisionArm, score_cells
from cluster_rag.text_processors.decision.probe import (
    GROUP_COMPETENCE,
    build_competence_cells,
    measure_competence_control,
)
from cluster_rag.text_processors.decision.upr_listwise import (
    fuse_rankings,
    generic_criteria,
    plan_listwise_calls,
    top_candidates,
)
from cluster_rag.text_processors.decision.upr_probe import (
    build_upr_cells,
    measure_upr_arm,
    score_matrix,
)
from cluster_rag.text_processors.decision.upr_rule import (
    upr_rule_block,
    upr_shortlist_rerank_block,
)
from cluster_rag.upr_analysis import UprView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UprDecisionSample:
    """The probe's sample: the texts, the label space and the gold matrix.

    The sample is a **seeded** draw of the evaluated split's texts (the same
    texts for every arm, so the arms are paired), and the gold matrix is the
    sample's own (the cells outside it are not scored at all).
    """

    texts: tuple[str, ...]
    labels: tuple[str, ...]
    gold: np.ndarray
    rows: tuple[int, ...]

    @property
    def n_texts(self) -> int:
        """The number of sampled texts."""
        return len(self.texts)

    @property
    def n_labels(self) -> int:
        """The size of the label space."""
        return len(self.labels)

    @property
    def n_cells(self) -> int:
        """The number of scored cells (the texts x the labels)."""
        return self.n_texts * self.n_labels


def sample_texts(
    view: UprView,
    *,
    size: int,
    seed: int,
) -> UprDecisionSample:
    """The probe's sample: a seeded draw of the split's texts.

    Args:
        view: The evaluated split's view.
        size: How many texts to take (the whole split when it is larger).
        seed: The seed of the draw (the same texts for every arm).

    Returns:
        The :class:`UprDecisionSample` of the probe.
    """
    n_texts = view.n_texts
    take = min(int(size), n_texts)
    generator = np.random.default_rng(seed)
    rows = tuple(sorted(int(row) for row in generator.permutation(n_texts)[:take]))
    texts = tuple(view.records[row].text for row in rows)
    gold = view.gold[np.asarray(rows, dtype=np.int64)] if rows else np.zeros((0, 0), dtype=bool)
    return UprDecisionSample(
        texts=texts,
        labels=tuple(view.labels),
        gold=gold,
        rows=rows,
    )


def score_upr_grid(
    sample: UprDecisionSample,
    decider: Decider,
    arm: DecisionArm,
    *,
    max_questions_per_call: int = 64,
    mismatched: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score a grid with one arm: the score matrix and the call plan's cost.

    This is the raw scoring half of :func:`run_upr_decision_arm`, split out
    so the **fit side** (the train split's grid, whose scores fit the rule's
    operating point) can be scored with the very same arm and the very same
    cache keys.

    Args:
        sample: The grid to score (the evaluated split's or the fit side's).
        decider: The decision model (the port; the CLI wires the client in).
        arm: The arm's prompt (the state shape, the question type, the task).
        max_questions_per_call: How many questions one call may carry.
        mismatched: Whether the cells are the **label-blind counterpart**.

    Returns:
        The ``(n_texts, n_labels)`` score matrix and the cost block (the
        cells, the planned calls and the truncation report).
    """
    cells = build_upr_cells(
        sample.texts,
        sample.labels,
        sample.gold,
        mismatched=mismatched,
    )
    probabilities, truncated, n_calls = score_cells(
        decider,
        cells,
        arm,
        max_questions_per_call=max_questions_per_call,
    )
    matrix = score_matrix(cells, probabilities)
    cost = {
        "n_cells": len(cells),
        "n_calls_planned": n_calls,
        "truncation": {
            "n_cells": len(cells),
            "n_truncated": int(sum(truncated)),
            "share": (sum(truncated) / len(cells)) if cells else None,
            "note": (
                "the cells whose text was cut to fit the model's input budget "
                "(the question and the instructions are never cut): a shortened "
                "input is never silent"
            ),
        },
    }
    return matrix, cost


def run_upr_decision_arm(
    sample: UprDecisionSample,
    decider: Decider,
    arm: DecisionArm,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    max_questions_per_call: int = 64,
    bins: int = 10,
    mismatched: bool = False,
    bucket_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one arm: score the sample's cells and measure them.

    Args:
        sample: The probe's sample.
        decider: The decision model (the port; the CLI wires the client in).
        arm: The arm's prompt (the state shape, the question type, the task).
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        max_questions_per_call: How many questions one call may carry.
        bins: The number of the reliability diagram's bins.
        mismatched: Whether the cells are the **label-blind counterpart**.
        bucket_masks: The label buckets' column masks (the per-bucket view).

    Returns:
        The arm's block (the measurement, the call plan's cost and the
        truncation report) and the arm's ``(n_texts, n_labels)`` score matrix
        (the rule and the rerank need it; it never enters the artifact).
    """
    matrix, cost = score_upr_grid(
        sample,
        decider,
        arm,
        max_questions_per_call=max_questions_per_call,
        mismatched=mismatched,
    )
    measurement = measure_upr_arm(
        matrix,
        sample.gold,
        decision=decision,
        ks=ks,
        shortlist_ms=shortlist_ms,
        bins=bins,
        bucket_masks=bucket_masks,
    )
    block = {
        "arm": arm.name,
        "state": arm.state,
        "question_type": arm.question_type,
        "instructions": arm.instructions,
        "options": list(arm.options),
        "criteria": dict(arm.criteria),
        "positive_option": arm.positive_option,
        "max_input_tokens": arm.max_input_tokens,
        "label_blind": mismatched,
        **cost,
        "measurement": measurement,
    }
    return block, matrix


def run_upr_listwise_stage(
    sample: UprDecisionSample,
    decider: Decider,
    *,
    round_one: np.ndarray,
    option_labels: Sequence[str],
    descriptions: Mapping[str, str] | None,
    criteria_source: str,
    n_candidates: int,
    task: str,
    question_text: str,
    max_input_tokens: int | None,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    bucket_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """The **listwise** second stage: one choice question per text (CS-12).

    The stage is a second round on top of the pointwise arm: the round-1
    scores give the **candidates** (the top-N labels per text) and the second
    round asks the model to **order** them ("which of these N topics is the
    text about?"). The arm's final ranking is the **rank fusion** (the
    round-2 order over the candidates, then the round-1 order for the rest),
    and its score matrix is the negative final rank.

    Args:
        sample: The probe's sample (the texts and the gold matrix).
        decider: The decision model (the port; the CLI wires the client in).
        round_one: The round-1 score matrix (the pointwise arm's).
        option_labels: The label name to **present** for every column (the
            label-blind counterpart presents the shifted one).
        descriptions: The catalog's descriptions (the ``criteria`` source);
            ``None`` when the catalog is absent.
        criteria_source: ``"catalog"`` or ``"name"`` (the control variant).
        n_candidates: How many candidates the second round offers.
        task: The listwise task instruction.
        question_text: The listwise question's wording.
        max_input_tokens: The model's input budget.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        bucket_masks: The label buckets' column masks (the per-bucket view).

    Returns:
        The stage's block (the measurement, the call plan's cost and the
        candidates' report) and the fused score matrix.
    """
    candidates = top_candidates(round_one, n_candidates)
    option_sets: list[tuple[str, ...]] = []
    criteria_sets: list[dict[str, str]] = []
    for columns in candidates:
        options = tuple(option_labels[column] for column in columns)
        option_sets.append(options)
        criteria: dict[str, str] = {}
        for option in options:
            if criteria_source == "catalog" and descriptions:
                criteria[option] = descriptions.get(option) or generic_criteria(option)
            else:
                criteria[option] = generic_criteria(option)
        criteria_sets.append(criteria)
    calls = plan_listwise_calls(
        sample.texts,
        option_sets,
        criteria_sets,
        task=task,
        question_text=question_text,
        max_input_tokens=max_input_tokens,
    )
    round_two = np.full(np.asarray(round_one).shape, np.nan, dtype=np.float64)
    truncated = 0
    for call in calls:
        index = call.cell_indexes[0]
        answers = decider.decide(
            call.state, dict(zip(call.question_names, call.specs, strict=True))
        )
        answer = answers.get(call.question_names[0])
        if answer is None:
            raise RuntimeError(f"the listwise call returned no answer for the text {index}")
        probabilities = answer.probabilities
        for column, option in zip(candidates[index], option_sets[index], strict=True):
            if option in probabilities:
                round_two[index, column] = float(probabilities[option])
        truncated += int(len(call.state.get("paragraph", "")) < len(sample.texts[index]))
    fused = fuse_rankings(round_one, round_two)
    measurement = measure_upr_arm(
        fused,
        sample.gold,
        decision=decision,
        ks=ks,
        shortlist_ms=shortlist_ms,
        bucket_masks=bucket_masks,
        probability_scores=False,
    )
    block = {
        "candidates": n_candidates,
        "criteria_source": criteria_source,
        "question_text": question_text,
        "instructions": task,
        "n_calls_planned": len(calls),
        "n_truncated": truncated,
        "measurement": measurement,
        "note": (
            "the listwise second stage: the round-1 top-N candidates per text "
            "are offered to the model in ONE choice question, and the arm's "
            "ranking is the fusion of the round-2 order (over the candidates) "
            "and the round-1 order (for the rest). The score matrix is the "
            "negative final rank, so the calibration and the probability "
            "compression are not applicable."
        ),
    }
    return block, fused


def upr_per_text_block(
    matrix: np.ndarray,
    gold: np.ndarray,
    labels: Sequence[str],
    labels_by_bucket: Mapping[str, Sequence[str]],
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
) -> dict[str, Any]:
    """An arm's **per-text** summary (what a paired comparison needs).

    The block is the compact form of the arm's two views: the decision
    view's per-text (true positives, false positives, false negatives) and the
    ranking view's per-text nDCG. It is what makes a **cross-run** paired
    comparison possible (the two score matrices are not in memory together).

    Args:
        matrix: The arm's score matrix.
        gold: The bool gold matrix (the same shape).
        labels: The label space (the matrix's column order).
        labels_by_bucket: The label buckets' members.
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.

    Returns:
        The per-text block.
    """
    from cluster_rag.eval.multilabel import apply_decision_rule
    from cluster_rag.upr_analysis import per_text_block

    values = np.asarray(matrix, dtype=np.float64)
    predicted = apply_decision_rule(
        values,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    return per_text_block(
        values,
        np.asarray(gold, dtype=bool),
        predicted,
        labels,
        labels_by_bucket,
        ks=ks,
    )


def run_upr_rule_stage(
    sample: UprDecisionSample,
    decider: Decider,
    arm: DecisionArm,
    *,
    arm_matrix: np.ndarray,
    blind_matrix: np.ndarray | None,
    fit_sample: UprDecisionSample | None,
    champion_scores: np.ndarray,
    fixed_cut: float,
    threshold_quantiles: int,
    margin_quantiles: int,
    rerank_ks: Sequence[int],
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    max_questions_per_call: int = 64,
) -> dict[str, Any]:
    """The safe-negative rule and the shortlist rerank (the track's CS-9b).

    The rule's three forms (the fixed cut, the fit-side threshold, the margin
    rule) with the **label-blind counterpart** beside each, and the shortlist
    rerank (the champion's top-K, re-ordered by the decision model's own
    probability). The rule's fit side is the **train** split, so no operating
    point is fitted on the evaluated split.

    Returns:
        The stage's block: the rule and the rerank.
    """
    block: dict[str, Any] = {}
    if fit_sample is not None:
        fit_matrix, fit_cost = score_upr_grid(
            fit_sample,
            decider,
            arm,
            max_questions_per_call=max_questions_per_call,
        )
        rule = build_upr_safe_negative_rule(
            arm_matrix,
            sample.gold,
            fit_scores=fit_matrix,
            fit_gold=fit_sample.gold,
            blind_scores=blind_matrix,
            fixed_cut=fixed_cut,
            threshold_quantiles=threshold_quantiles,
            margin_quantiles=margin_quantiles,
        )
        rule["fit_side_cost"] = fit_cost
        block["safe_negative_rule"] = rule
    block["shortlist_rerank"] = build_upr_shortlist_rerank(
        arm_matrix,
        sample.gold,
        reference_scores=champion_scores,
        ks=rerank_ks,
    )
    return block


def build_upr_safe_negative_rule(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    fit_scores: np.ndarray,
    fit_gold: np.ndarray,
    blind_scores: np.ndarray | None = None,
    fixed_cut: float = 0.5,
    threshold_quantiles: int = 99,
    margin_quantiles: int = 49,
) -> dict[str, Any]:
    """The safe-negative rule on the UPR grid (the pure core's block).

    The rule's three forms (the fixed cut, the fit-side threshold, the margin
    rule) with the **label-blind counterpart** beside each, at the **same**
    operating point (the CS-7b OMCT precedent). The fit side is the **train**
    split's grid, so no operating point is fitted on the evaluated split.
    """
    return upr_rule_block(
        scores,
        gold,
        fit_scores=fit_scores,
        fit_gold=fit_gold,
        blind_scores=blind_scores,
        fixed_cut=fixed_cut,
        threshold_quantiles=threshold_quantiles,
        margin_quantiles=margin_quantiles,
    )


def build_upr_shortlist_rerank(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    reference_scores: np.ndarray,
    ks: Sequence[int],
) -> dict[str, Any]:
    """The shortlist rerank on UPR (the pure core's block).

    The champion's top-K labels per text, re-ordered by the decision model's
    own probability, with the shortlist's own recall as the ceiling.
    """
    return upr_shortlist_rerank_block(
        scores,
        gold,
        reference_scores=reference_scores,
        ks=ks,
    )


def run_upr_competence_control(
    sample: UprDecisionSample,
    decider: Decider,
    arm: DecisionArm,
    *,
    n_pairs: int,
    max_questions_per_call: int = 64,
    bins: int = 10,
) -> dict[str, Any]:
    """The **synthetic competence control** of the UPR probe.

    The control holds two cells per text with the *same* question template and
    the *same* text: the positive quotes a distinctive phrase **of that text**
    (so the pair shares a phrase) and the negative quotes a phrase of a
    *different* text (so it shares none). A model that can detect a shared
    phrase at all separates the two — which is what tells "the data is at the
    ceiling" from "the model is the wrong shape".
    """
    cells = build_competence_cells(list(sample.texts), n_pairs=n_pairs, document="competence")
    if not cells:
        return {"n_pairs": 0, "note": "no competence cell could be built"}
    probabilities, truncated, n_calls = score_cells(
        decider,
        cells,
        arm,
        max_questions_per_call=max_questions_per_call,
    )
    measurement = measure_competence_control(cells, probabilities)
    return {
        "n_pairs": len(cells) // 2,
        "n_cells": len(cells),
        "n_calls_planned": n_calls,
        "n_truncated": int(sum(truncated)),
        "group": GROUP_COMPETENCE,
        "measurement": measurement,
    }


def build_upr_decision_report(
    sample: UprDecisionSample,
    arms: Mapping[str, dict[str, Any]],
    *,
    run_id: str,
    split: str,
    source: str,
    source_sha256: str,
    scope: str,
    seed: int,
    config_block: Mapping[str, Any],
    reference: Mapping[str, Any] | None = None,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    """Assemble the probe's artifact.

    Args:
        sample: The probe's sample.
        arms: The arms' blocks, keyed by the arm's name.
        run_id: The run's id.
        split: The evaluated split (``val``).
        source: The dataset path.
        source_sha256: The dataset's sha256.
        scope: The scope (``sample`` or ``grid``).
        seed: The sample's seed.
        config_block: The probe's configuration (the provenance).
        reference: The pipeline's own scoring of the probe's cells (the
            champion's numbers on the same texts); ``None`` when it was not
            measured.
        total_seconds: The run's wall time.

    Returns:
        The JSON-serializable artifact.
    """
    return {
        "run_id": run_id,
        "command": "decide-upr",
        "split": split,
        "source": source,
        "source_sha256": source_sha256,
        "sample": {
            "scope": scope,
            "seed": int(seed),
            "n_texts": sample.n_texts,
            "n_labels": sample.n_labels,
            "n_cells": sample.n_cells,
            "n_gold_cells": int(sample.gold.sum()),
            "n_background_cells": int((~sample.gold).sum()),
            "rows": list(sample.rows),
            "note": (
                "the probe's sample: a seeded draw of the evaluated split's "
                "texts, the same texts for every arm (so the arms are paired). "
                "The grid scope takes every text of the split."
            ),
        },
        "config": dict(config_block),
        "reference_arm": dict(reference) if reference else None,
        "arms": dict(arms),
        "total_seconds": total_seconds,
    }
