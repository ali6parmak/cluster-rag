"""Zero-shot quota insertion for the EURLEX-4K shortlist (menu item 5).

A seen-label k-NN can never rank a label with no train example: its score is 0
for every document.  The title embeddings give those labels a signal, but the
raw cosine does not survive the full-ranking competition (TC-4/TC-6).  The
quota takes the other route: reserve the **bottom N slots of the top-50** for
the N zero-shot labels with the highest title cosine, displacing the base
ranking's weakest shortlist entries.

Inserting at the bottom of the shortlist is the minimal-cost placement: the
reserved labels join the top-50 but stay behind the base's ranked entries, so
they cannot disturb R-precision (``k = |gold|``) or nDCG@10, and only the
documents whose gold sat in the displaced positions pay for the reservation.

Pure numpy; the caller supplies the base order and the candidates' order.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DEPTH = 50


def quota_insertion(
    base_order: np.ndarray,
    candidate_order: np.ndarray,
    *,
    quota: int,
    depth: int = DEFAULT_DEPTH,
) -> np.ndarray:
    """Insert up to ``quota`` reserved candidates at the bottom of the top-``depth``.

    Args:
        base_order: ``(n_documents, n_labels)`` label indices, best first.
        candidate_order: ``(n_documents, n_candidates)`` reserved-label columns,
            best first (e.g. the zero-shot labels by title cosine).
        quota: How many slots to reserve; ``0`` returns ``base_order`` unchanged.
        depth: The shortlist depth (the top-``depth`` is the shortlist).

    Returns:
        A new ``(n_documents, n_labels)`` int32 order. The reserved candidates
        occupy the last ``quota`` slots of the top-``depth``; the entries they
        displace move to just after the shortlist. A candidate already ranked
        inside the top-``depth`` is skipped, so fewer than ``quota`` labels can
        be inserted.

    Raises:
        ValueError: On non-2-D inputs, mismatched label counts, a negative
            quota, a non-positive depth, or a quota larger than the depth.
    """
    base = np.asarray(base_order)
    candidates = np.asarray(candidate_order)
    if base.ndim != 2 or candidates.ndim != 2:
        raise ValueError(
            f"expected 2-D orders; got base {base.shape} and candidates {candidates.shape}"
        )
    if base.shape[0] != candidates.shape[0]:
        raise ValueError(
            f"document counts differ: base {base.shape[0]}, candidates {candidates.shape[0]}"
        )
    if candidates.size and (candidates.min() < 0 or candidates.max() >= base.shape[1]):
        raise ValueError(
            f"candidate label columns must lie in [0, {base.shape[1]}); got "
            f"[{int(candidates.min())}, {int(candidates.max())}]"
        )
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
        raise ValueError(f"quota must be a non-negative integer, got {quota!r}")
    if quota > depth:
        raise ValueError(f"quota {quota} exceeds the shortlist depth {depth}")
    if quota == 0:
        return base.astype(np.int32, copy=True)

    n_documents, n_labels = base.shape
    out = np.empty((n_documents, n_labels), dtype=np.int32)
    for row in range(n_documents):
        head = base[row, :depth]
        tail = base[row, depth:]
        head_set = set(head.tolist())
        chosen: list[int] = []
        chosen_mask = np.zeros(n_labels, dtype=bool)
        for label in candidates[row].tolist():
            if len(chosen) == quota:
                break
            label = int(label)
            if label in head_set or chosen_mask[label]:
                continue
            chosen.append(label)
            chosen_mask[label] = True
        chosen_array = np.asarray(chosen, dtype=np.int32)
        keep = depth - len(chosen)
        displaced = head[keep:]
        used = chosen_mask.copy()
        used[displaced] = True
        rest = tail[~used[tail]]
        out[row] = np.concatenate([head[:keep], chosen_array, displaced, rest])
    return out
