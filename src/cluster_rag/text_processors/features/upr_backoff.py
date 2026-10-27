"""The **sibling-max backoff**: the catalog's parent channel (the track's CS-11).

The label catalog (CS-11) carries a **parent** per label — the label's place
in the human-rights taxonomy. The reviewer's mechanism for the labels the
train split never saw is a **sibling-max backoff**: a label with **no
fit-side support** inherits the score of a taxonomy **sibling**, so

    score'(text, L) = max(score(text, L), lambda * max over the siblings S of score(text, S))

The backoff is a **fallback**, not a better scorer: it can only raise a
label's score, and it raises it only where a sibling scores higher.

**The siblings are the supported ones.** A sibling with no fit-side support
of its own has no fitted score to inherit, so the sibling set of a label is
its taxonomy siblings **that have at least one gold cell on the fit side**
(the fit side decides this — never the evaluated split).

**The control is the point.** The same rule applied to a label that *has*
fit-side support must **not** help it: a backoff that helps every label is
not a fallback but "another correlated signal" (the reviewer's wording), and
the honest claim changes with it. The control's labels are chosen on the fit
side (the rarest supported labels, matched in rarity to the unsupported
ones), so the comparison is fair.

Pure core: numpy + the standard library only (the track's pure-core rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def fit_side_counts(fit_gold: np.ndarray) -> np.ndarray:
    """The fit side's gold-cell count per label (the labels' support)."""
    truth = np.asarray(fit_gold, dtype=bool)
    if truth.ndim != 2:
        raise ValueError(f"the fit side's gold matrix must be 2-dimensional, got {truth.shape}")
    return truth.sum(axis=0).astype(np.int64)


def unsupported_labels(
    fit_gold: np.ndarray,
    labels: Sequence[str],
) -> tuple[str, ...]:
    """The labels with **no fit-side support** (zero gold cells on the fit side).

    Args:
        fit_gold: The fit side's bool gold matrix.
        labels: The label space (the matrix's column order).

    Returns:
        The unsupported labels, in the label space's order.
    """
    counts = fit_side_counts(fit_gold)
    return tuple(label for label, count in zip(labels, counts, strict=True) if count == 0)


def control_labels(
    fit_gold: np.ndarray,
    labels: Sequence[str],
    *,
    n: int,
) -> tuple[str, ...]:
    """The **control's** labels: the ``n`` rarest *supported* labels.

    The control asks whether the backoff helps a label that has its own
    fitted support. The fairest such label is the **rarest** supported one
    (the labels the backoff would help most if it were a general
    improvement), so the control's labels are chosen on the **fit side** by
    the fit side's own counts, with the label name as the tie-break (the
    choice is deterministic).

    Args:
        fit_gold: The fit side's bool gold matrix.
        labels: The label space (the matrix's column order).
        n: How many control labels to take.

    Returns:
        The control labels, rarest first.
    """
    counts = fit_side_counts(fit_gold)
    supported = [
        (int(count), label) for label, count in zip(labels, counts, strict=True) if count > 0
    ]
    supported.sort()
    return tuple(label for _, label in supported[:n])


def sibling_sets(
    labels: Sequence[str],
    parents: Mapping[str, str],
    *,
    supported: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Every label's **supported** siblings (its taxonomy siblings, minus itself).

    A label whose parent is unknown (the catalog does not cover it) has no
    siblings, and so has no backoff — a measured null by construction.

    Args:
        labels: The label space.
        parents: The label -> parent map (the catalog's field).
        supported: The labels with fit-side support (the ones a sibling
            score may come from).

    Returns:
        The sibling set of every label.
    """
    supported_set = set(supported)
    by_parent: dict[str, list[str]] = {}
    for label in labels:
        parent = parents.get(label)
        if not parent:
            continue
        by_parent.setdefault(parent, []).append(label)
    siblings: dict[str, tuple[str, ...]] = {}
    for label in labels:
        parent = parents.get(label)
        if not parent:
            siblings[label] = ()
            continue
        siblings[label] = tuple(
            other
            for other in by_parent.get(parent, ())
            if other != label and other in supported_set
        )
    return siblings


def sibling_backoff(
    scores: np.ndarray,
    labels: Sequence[str],
    parents: Mapping[str, str],
    fit_gold: np.ndarray,
    *,
    targets: Sequence[str],
    lam: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The backoff'd score matrix: a target label inherits its siblings' best score.

    Args:
        scores: The ``(n_texts, n_labels)`` score matrix of the arm.
        labels: The label space (the matrix's column order).
        parents: The label -> parent map (the catalog's field).
        fit_gold: The fit side's bool gold matrix (which labels have support).
        targets: The labels the backoff applies to.
        lam: The mixing weight (the backoff's score is ``lam`` times the
            siblings' best, and the label's own score is the floor).

    Returns:
        The backoff'd score matrix (the non-target labels are untouched) and
        the report: every target's siblings, the share of the cells where the
        backoff changed the score and the mean lift.

    Raises:
        ValueError: If the shapes disagree or a target is not in the label
            space.
    """
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(fit_gold, dtype=bool)
    if values.shape[1] != len(labels):
        raise ValueError(f"the score matrix has {values.shape[1]} columns for {len(labels)} labels")
    if truth.shape[1] != len(labels):
        raise ValueError(
            f"the fit side's gold matrix has {truth.shape[1]} columns for {len(labels)} labels"
        )
    if lam < 0:
        raise ValueError(f"the backoff's weight must be >= 0, got {lam}")
    index = {label: position for position, label in enumerate(labels)}
    unknown = [label for label in targets if label not in index]
    if unknown:
        raise ValueError(f"the backoff's targets {unknown} are not in the label space")
    counts = fit_side_counts(truth)
    supported = [label for label, count in zip(labels, counts, strict=True) if count > 0]
    siblings = sibling_sets(labels, parents, supported=supported)

    backoff = values.copy()
    per_target: dict[str, Any] = {}
    for label in targets:
        column = index[label]
        family = siblings.get(label, ())
        if not family:
            per_target[label] = {
                "n_siblings": 0,
                "siblings": [],
                "share_changed": 0.0,
                "mean_lift": 0.0,
                "note": (
                    "no supported sibling: the backoff cannot act on this label "
                    "(a measured null by construction)"
                ),
            }
            continue
        sibling_columns = [index[other] for other in family]
        inherited = values[:, sibling_columns].max(axis=1) * float(lam)
        own = values[:, column]
        lifted = np.maximum(own, inherited)
        changed = lifted > own
        backoff[:, column] = lifted
        per_target[label] = {
            "n_siblings": len(family),
            "siblings": list(family),
            "share_changed": float(changed.mean()),
            "mean_lift": float((lifted - own).mean()),
            "max_lift": float((lifted - own).max()),
        }
    return backoff, {
        "lam": float(lam),
        "targets": list(targets),
        "per_target": per_target,
        "note": (
            "the sibling-max backoff: a target label's score is raised to the "
            "siblings' best (times the weight) when that is higher. The "
            "siblings are the taxonomy siblings WITH fit-side support; a label "
            "without one is a measured null."
        ),
    }
