"""The UPR learned arm: the feature pack, the combiner and the three floors.

`TEXT_PROCESSING_PLAN.md` section 7.1 (CS-8d) asks for the OMCT track's
learned pair scorer on the **UPR** data: the feature pack, the logistic
combiner and the three sanity floors, all measured on the UPR harness's own
metrics.

The protocol is the main pipeline's (AGENTS.md: the indexes and the fit side
come from **train**, val is for tuning, test is touched once):

- the **fit side** is the train split's whole grid — the train gold cells
  (weight 1) plus the train non-gold cells as **presumed negatives** (weight
  ``control_weight``, 0.3 as on OMCT: their true positive rate is unmeasured,
  so they must not dominate the fit);
- the **evaluation** is the val split's whole grid, scored by the fitted
  combiner, and reported with the UPR harness's own metrics (the decision
  view and the ranking view, per label bucket and per length bucket) beside
  the whole-text reference arm's numbers.

**The three floors** are measured on the same statistic as the headline (not
on an AUC, as on OMCT — UPR has no pairwise AUC):

- the **chance floor** (the label-permutation floor): the fit side's whole
  label vector is permuted, so the class counts and the weights stay exactly
  as they are and only the assignment changes. The model has no label
  information at all.
- the **control-role floor**: the fit side's gold cells' positive labels are
  relocated into the non-gold space — the non-gold cells keep their
  presumed-negative role as the negative class and the original gold cells
  become negatives. The gap to the chance floor is what the training design's
  own prior ("the ordinary cells are negative") buys. On UPR the grid has no
  separate "judged vs control" group (the non-gold cells are the gold cells'
  complement), so this floor is structurally close to the chance floor; the
  measurement says by how much.
- the **random-vector floor**: the pack's **embedding sides** are replaced
  with seeded random unit vectors, so the lexical features stay real and the
  embedding geometry is gone. It says how much of the arm's number is the
  embedding's and how much is the vocabulary's.

Driver level (AGENTS.md's dependency direction): the module orchestrates the
track's pure core (the pack, the combiners) and the UPR harness's metrics; it
never imports a model client — the CLI wires the backends in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    decision_f1_metrics,
    ranking_metrics,
    shortlist_recall,
)
from cluster_rag.text_processors.base import Combiner
from cluster_rag.text_processors.features.upr_features import UprFeatureExtractor

logger = logging.getLogger(__name__)

# The name of the learned arm (the artifact's arm key).
LEARNED_ARM = "learned"


@dataclass(frozen=True)
class UprLearningResult:
    """The learned arms' score matrices and the artifact's ``learned`` block.

    Attributes:
        scores: The arms' ``(n_texts, n_labels)`` score matrices, keyed by the
            arm's name.
        meta: The artifact's ``learned`` block (the pack, the combiner, the
            training's shape and the three floors).
    """

    scores: dict[str, np.ndarray]
    meta: dict[str, Any]


def _floor_metric_block(
    scores: np.ndarray,
    gold: np.ndarray,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
) -> dict[str, float]:
    """The headline statistic of one floor draw (the same numbers as the arm)."""
    predicted = apply_decision_rule(
        scores,
        rule=decision["rule"],
        k=decision.get("k"),
        threshold=decision.get("threshold"),
        min_score=decision.get("min_score"),
    )
    f1 = decision_f1_metrics(predicted, gold)
    ranking = ranking_metrics(scores, gold, tuple(ks))
    shortlist = shortlist_recall(scores, gold, tuple(shortlist_ms))
    block = {
        "micro_f1": float(f1["micro_f1"]),
        "macro_f1": float(f1["macro_f1"]),
        "sample_f1": float(f1["sample_f1"]),
        "coverage": float(f1["n_predicted_cells"]) / float(max(1, f1["n_gold_cells"])),
    }
    for k in ks:
        block[f"precision@{k}"] = float(ranking.get(f"precision@{k}", 0.0))
        block[f"recall@{k}"] = float(ranking.get(f"recall@{k}", 0.0))
        block[f"ndcg@{k}"] = float(ranking.get(f"ndcg@{k}", 0.0))
    for m in shortlist_ms:
        block[f"shortlist_recall@{m}"] = float(shortlist.get(f"recall@{m}", 0.0))
    return block


def _summarize_draws(draws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The mean / min / max of a floor's draws (per metric)."""
    if not draws:
        return {"n_draws": 0}
    keys = sorted(draws[0].keys())
    summary: dict[str, Any] = {"n_draws": len(draws)}
    for key in keys:
        values = np.asarray([float(draw[key]) for draw in draws], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _fit_rows(
    features: np.ndarray,
    gold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The fit side's rows and labels (the grid, flattened)."""
    rows = features.reshape(-1, features.shape[-1])
    labels = np.asarray(gold, dtype=bool).reshape(-1).astype(np.int64)
    return rows, labels


def _weights(labels: np.ndarray, control_weight: float) -> np.ndarray:
    """The training weights: the gold cells at 1, the presumed negatives at ``control_weight``."""
    return np.where(labels == 1, 1.0, float(control_weight))


def _permuted_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The **chance floor**'s labels: the whole label vector, permuted."""
    permuted = labels.copy()
    rng.shuffle(permuted)
    return permuted


def _control_role_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The **control-role floor**'s labels.

    The gold cells' positive labels are relocated into the non-gold space: the
    non-gold cells keep their presumed-negative role as the negative class,
    the original gold cells become negatives, and the class counts and the
    training weights stay exactly as they are.
    """
    gold_rows = np.flatnonzero(labels == 1)
    non_gold_rows = np.flatnonzero(labels == 0)
    permuted = np.zeros_like(labels)
    if gold_rows.size == 0 or non_gold_rows.size == 0:
        return permuted
    chosen = rng.choice(non_gold_rows, size=gold_rows.size, replace=False)
    permuted[chosen] = 1
    return permuted


def score_upr_learned_arms(
    *,
    train_texts: Sequence[str],
    train_gold: np.ndarray,
    eval_texts: Sequence[str],
    eval_gold: np.ndarray,
    extractor: UprFeatureExtractor,
    combiner_factory: Callable[[], Combiner],
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    control_weight: float = 0.3,
    permutation_resamples: int = 5,
    random_extractor: UprFeatureExtractor | None = None,
    seed: int = 42,
    arm_name: str = LEARNED_ARM,
    reporting: Mapping[str, Any] | None = None,
    bucket_masks: Mapping[str, np.ndarray] | None = None,
    length_masks: Mapping[str, np.ndarray] | None = None,
    decision_train: np.ndarray | None = None,
    decision_eval: np.ndarray | None = None,
    rerank_train: np.ndarray | None = None,
    rerank_eval: np.ndarray | None = None,
) -> UprLearningResult:
    """Fit the learned arm on train and score it (and its floors) on val.

    Args:
        train_texts: The train split's texts (the fit side).
        train_gold: The train split's bool gold matrix.
        eval_texts: The evaluated split's texts (val).
        eval_gold: The evaluated split's bool gold matrix.
        extractor: The feature pack (the real embedding sides).
        combiner_factory: A callable that builds a fresh combiner (one per
            fit: the floors must not share a model).
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        control_weight: The training weight of the non-gold cells.
        permutation_resamples: The number of label permutations per floor.
        random_extractor: The feature pack with the **random** embedding sides
            (the random-vector floor). ``None`` disables that floor.
        seed: The seed of the permutations and of the random draw.
        arm_name: The arm's name (the artifact's key).
        reporting: The reporting block's settings (``None`` computes none, so
            every earlier run's artifact keeps its exact content).
        bucket_masks: The label buckets' column masks (the calibration's
            per-bucket view).
        length_masks: The text-length buckets' row masks (the calibration's
            per-length-bucket view).
        decision_train: The decision model's probability matrix for the **fit
            side**'s cells (the ``decision`` group's column; the CS-13
            feature).
        decision_eval: The decision model's probability matrix for the
            evaluated split's cells.
        rerank_train: The reranker's score matrix for the **fit side**'s
            cells (the ``rerank`` group's column; the CS-16 feature).
        rerank_eval: The reranker's score matrix for the evaluated split's
            cells.

    Returns:
        The :class:`UprLearningResult` (the arms' score matrices and the
        artifact's ``learned`` block).
    """
    train_features = extractor.extract(train_texts, decision_train, rerank_train)
    eval_features = extractor.extract(eval_texts, decision_eval, rerank_eval)
    train_rows, train_labels = _fit_rows(train_features, train_gold)
    eval_rows = eval_features.reshape(-1, eval_features.shape[-1])
    weights = _weights(train_labels, control_weight)

    combiner = combiner_factory().fit(train_rows, train_labels, weights)
    arm_scores = np.asarray(combiner.score(eval_rows), dtype=np.float64).reshape(
        eval_features.shape[0], eval_features.shape[1]
    )
    scores: dict[str, np.ndarray] = {arm_name: arm_scores}
    rng = np.random.default_rng(seed)
    floors: dict[str, Any] = {}

    # -- the chance floor (the label-permutation floor) --
    chance_draws: list[dict[str, Any]] = []
    for _ in range(permutation_resamples):
        permuted = combiner_factory().fit(train_rows, _permuted_labels(train_labels, rng), weights)
        draw = np.asarray(permuted.score(eval_rows), dtype=np.float64).reshape(
            eval_features.shape[0], eval_features.shape[1]
        )
        chance_draws.append(
            _floor_metric_block(
                draw, eval_gold, decision=decision, ks=ks, shortlist_ms=shortlist_ms
            )
        )
    if chance_draws:
        floors["chance"] = {
            "summary": _summarize_draws(chance_draws),
            "draws": chance_draws,
            "note": (
                "the label-permutation floor: the fit side's whole label vector "
                "is permuted (the class counts and the training weights stay "
                "exactly as they are, only the assignment changes), so the "
                "model has no label information at all"
            ),
        }

    # -- the control-role floor --
    control_draws: list[dict[str, Any]] = []
    for _ in range(permutation_resamples):
        permuted = combiner_factory().fit(
            train_rows, _control_role_labels(train_labels, rng), weights
        )
        draw = np.asarray(permuted.score(eval_rows), dtype=np.float64).reshape(
            eval_features.shape[0], eval_features.shape[1]
        )
        control_draws.append(
            _floor_metric_block(
                draw, eval_gold, decision=decision, ks=ks, shortlist_ms=shortlist_ms
            )
        )
    if control_draws:
        floors["control_role"] = {
            "summary": _summarize_draws(control_draws),
            "draws": control_draws,
            "note": (
                "the control-role floor: the fit side's gold cells' positive "
                "labels are relocated into the non-gold space (the non-gold "
                "cells keep their presumed-negative role and the original gold "
                "cells become negatives). The gap to the chance floor is what "
                "the training design's own prior buys."
            ),
        }

    # -- the random-vector floor --
    if random_extractor is not None:
        random_train = random_extractor.extract(train_texts, decision_train, rerank_train)
        random_eval = random_extractor.extract(eval_texts, decision_eval, rerank_eval)
        random_rows, random_labels = _fit_rows(random_train, train_gold)
        random_eval_rows = random_eval.reshape(-1, random_eval.shape[-1])
        random_combiner = combiner_factory().fit(random_rows, random_labels, weights)
        random_scores = np.asarray(
            random_combiner.score(random_eval_rows), dtype=np.float64
        ).reshape(random_eval.shape[0], random_eval.shape[1])
        scores[f"{arm_name}_random"] = random_scores
        floors["random_vectors"] = {
            "summary": _summarize_draws(
                [
                    _floor_metric_block(
                        random_scores,
                        eval_gold,
                        decision=decision,
                        ks=ks,
                        shortlist_ms=shortlist_ms,
                    )
                ]
            ),
            "draws": [
                _floor_metric_block(
                    random_scores,
                    eval_gold,
                    decision=decision,
                    ks=ks,
                    shortlist_ms=shortlist_ms,
                )
            ],
            "note": (
                "the random-vector floor: the feature pack's embedding sides "
                "are seeded random unit vectors, so the lexical features stay "
                "real and the embedding geometry is gone. It says how much of "
                "the arm's number is the vocabulary's and how much is the "
                "embedding's."
            ),
        }

    meta: dict[str, Any] = {
        "arm": arm_name,
        "pack": extractor.describe(),
        "combiner": _describe_combiner(combiner),
        "coefficients": _named_coefficients(combiner, extractor.feature_names),
        "feature_diagnostics": _feature_diagnostics(
            train_features=train_features,
            train_gold=np.asarray(train_gold, dtype=bool),
            train_texts=train_texts,
            eval_features=eval_features,
            eval_gold=np.asarray(eval_gold, dtype=bool),
            eval_texts=eval_texts,
            feature_names=extractor.feature_names,
        ),
        "training": {
            "fit_side": "train",
            "evaluation": "the evaluated split (val)",
            "n_train_rows": int(train_rows.shape[0]),
            "n_train_gold": int((train_labels == 1).sum()),
            "n_train_non_gold": int((train_labels == 0).sum()),
            "n_eval_rows": int(eval_rows.shape[0]),
            "control_weight": float(control_weight),
            "fit_weight_mass": {
                "gold": float(weights[train_labels == 1].sum()),
                "non_gold": float(weights[train_labels == 0].sum()),
            },
            "leakage_rules": [
                "the fit side is the train split only; the evaluated split "
                "never reaches the combiner",
                "the features are cell-local: no corpus statistic, no "
                "document-identity feature and no length feature",
                "the scaler of the combiner is fitted on the fit side's rows only",
            ],
        },
        "floors": floors,
        "permutation_resamples": int(permutation_resamples),
        "seed": int(seed),
    }
    # The reporting blocks (CS-8f): the calibration and the decision-rule
    # comparison of the arm. The threshold is chosen on the fit side, so the
    # fit side's scores are needed here.
    if reporting and reporting.get("enabled"):
        train_scores = np.asarray(combiner.score(train_rows), dtype=np.float64).reshape(
            train_features.shape[0], train_features.shape[1]
        )
        meta["reporting"] = _reporting_block(
            scores=arm_scores,
            train_scores=train_scores,
            gold=np.asarray(eval_gold, dtype=bool),
            train_gold=np.asarray(train_gold, dtype=bool),
            decision=decision,
            bins=int(reporting.get("bins", 10)),
            threshold_quantiles=int(reporting.get("threshold_quantiles", 99)),
            bucket_masks=dict(bucket_masks or {}),
            length_masks=dict(length_masks or {}),
        )
    return UprLearningResult(scores=scores, meta=meta)


def _reporting_block(
    *,
    scores: np.ndarray,
    train_scores: np.ndarray,
    gold: np.ndarray,
    train_gold: np.ndarray,
    decision: Mapping[str, Any],
    bins: int,
    threshold_quantiles: int,
    bucket_masks: Mapping[str, np.ndarray],
    length_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """The reporting blocks of one arm (CS-8f).

    Two blocks:

    - the **calibration**: the reliability diagram of the arm's probability
      against the gold cells, pooled and per label bucket (a column mask) and
      per text-length bucket (a row mask);
    - the **decision-rule comparison**: the three rules on the same arm. The
      threshold rules' cut is chosen on the **fit side** (the train split) by
      the Youden separation, so the evaluated split never picks it.
    """
    from cluster_rag.text_processors.calibration import (
        reliability_diagram,
        threshold_candidates,
        youden_curve,
    )

    truth = np.asarray(gold, dtype=bool)
    fit_truth = np.asarray(train_gold, dtype=bool)
    calibration: dict[str, Any] = {
        "pooled": reliability_diagram(
            scores.reshape(-1), truth.reshape(-1).astype(int), n_bins=bins
        ),
        "per_label_bucket": {
            name: reliability_diagram(
                scores[:, mask].reshape(-1),
                truth[:, mask].reshape(-1).astype(int),
                n_bins=bins,
            )
            for name, mask in bucket_masks.items()
        },
        "per_length_bucket": {
            name: reliability_diagram(
                scores[mask].reshape(-1),
                truth[mask].reshape(-1).astype(int),
                n_bins=bins,
            )
            for name, mask in length_masks.items()
        },
        "note": (
            "the reliability diagram of the arm's probability against the gold "
            "cells: the expected calibration error, the Brier score and the "
            "per-bin records. The per-label-bucket view restricts the columns, "
            "the per-length-bucket view the rows."
        ),
    }

    # the fit side's threshold (the Youden separation on the train split)
    fit_scores = train_scores.reshape(-1)
    fit_labels = fit_truth.reshape(-1)
    candidates = threshold_candidates(fit_scores, n_quantiles=threshold_quantiles)
    curve = youden_curve(
        fit_scores,
        positives=fit_labels,
        negatives=~fit_labels,
        candidates=candidates,
    )
    best = max(curve, key=lambda entry: entry["value"])
    threshold = float(best["threshold"])
    k = decision.get("k")
    rules: dict[str, Any] = {}
    for rule, parameters in (
        ("top_k", {"k": k}),
        ("score_threshold", {"threshold": threshold}),
        ("top_k_min_score", {"k": k, "min_score": threshold}),
    ):
        predicted = apply_decision_rule(scores, rule=rule, **parameters)
        metrics = decision_f1_metrics(predicted, truth)
        rules[rule] = {
            "parameters": parameters,
            "micro_f1": float(metrics["micro_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "sample_f1": float(metrics["sample_f1"]),
            "coverage": float(metrics["n_predicted_cells"])
            / float(max(1, metrics["n_gold_cells"])),
            "n_predicted_cells": int(metrics["n_predicted_cells"]),
            "n_gold_cells": int(metrics["n_gold_cells"]),
        }
    return {
        "calibration": calibration,
        "decision_rules": {
            "rules": rules,
            "threshold": {
                "value": threshold,
                "criterion": "youden (TPR - FPR) on the fit side (the train split)",
                "fit_side_tpr": best["tpr"],
                "fit_side_fpr": best["fpr"],
                "fit_side_value": best["value"],
                "n_candidates": int(len(candidates)),
                "note": (
                    "the threshold is chosen on the fit side only, so the "
                    "evaluated split never picks the operating point"
                ),
            },
            "note": (
                "the three decision rules on the same arm: the top-k (the run's "
                "k), the score threshold (the fit side's cut) and the top-k with "
                "the same cut as a floor"
            ),
        },
    }


def _feature_diagnostics(
    *,
    train_features: np.ndarray,
    train_gold: np.ndarray,
    train_texts: Sequence[str],
    eval_features: np.ndarray,
    eval_gold: np.ndarray,
    eval_texts: Sequence[str],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """The per-feature diagnostics of the evaluated and the fit side.

    The CS-6c precedent, adapted to the UPR grid: every feature's own ranking
    quality, its background-side counterpart and its length correlation. The
    evaluated split's block is the headline; the fit side's AUC is reported
    beside it (a feature that only works on the fit side is a fit-side
    artefact).
    """
    from cluster_rag.text_processors.upr_diagnostics import feature_diagnostics

    evaluated = feature_diagnostics(
        eval_features,
        eval_gold,
        [len(text) for text in eval_texts],
        feature_names,
    )
    fitted = feature_diagnostics(
        train_features,
        train_gold,
        [len(text) for text in train_texts],
        feature_names,
    )
    for name, block in evaluated["features"].items():
        block["auc_train"] = fitted["features"][name]["auc"]
    evaluated["fit_side"] = {
        "n_cells": fitted["n_cells"],
        "n_texts": fitted["n_texts"],
        "note": "the fit side's own numbers (the AUC of every feature there)",
    }
    return evaluated


def _describe_combiner(combiner: Combiner) -> dict[str, Any]:
    """The combiner's own description (a combiner without one says so)."""
    describe = getattr(combiner, "describe", None)
    if callable(describe):
        described = describe()
        if isinstance(described, dict):
            return dict(described)
    return {"name": type(combiner).__name__, "described": False}


def _named_coefficients(combiner: Combiner, feature_names: Sequence[str]) -> dict[str, Any]:
    """The combiner's coefficients, keyed by the feature's name.

    A combiner that exposes no coefficients (a tree ensemble) has none: the
    empty mapping says so.
    """
    coefficients = getattr(combiner, "coefficients", None)
    if not callable(coefficients):
        return {}
    values = coefficients()
    named: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        key = str(index)
        if key in values:
            named[name] = float(values[key])
    return named
