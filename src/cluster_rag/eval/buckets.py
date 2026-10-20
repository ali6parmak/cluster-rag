"""Label frequency buckets for evaluation (Phase 3): head / torso / tail / unseen.

Membership uses the **full-dataset** label counts, so the buckets are identical
to the Phase 1 ``stats`` artifact (thresholds live in the stats config:
head count >= ``head_min``, torso count >= ``torso_min``, tail below).
A label absent from train is ``unseen`` — regardless of its size bucket;
unseen takes precedence, so held-out labels are always evaluated as unseen
(the zero-shot discovery test), never mixed into their size bucket.
"""

from __future__ import annotations

from collections.abc import Mapping

BUCKET_NAMES: tuple[str, str, str, str] = ("head", "torso", "tail", "unseen")


def assign_label_buckets(
    label_counts: Mapping[str, int],
    train_counts: Mapping[str, int],
    *,
    head_min: int,
    torso_min: int,
) -> dict[str, str]:
    """Map every label to its evaluation bucket name.

    Args:
        label_counts: Full-dataset example count per label (the Phase 1
            ``stats`` view; this fixes the size buckets).
        train_counts: Train-split example count per label; a count of 0 marks
            the label unseen (train artifacts must never contain it).
        head_min: Minimum count for the head bucket (from the stats config).
        torso_min: Minimum count for the torso bucket (below is tail).

    Returns:
        ``{label: "head" | "torso" | "tail" | "unseen"}`` covering every label
        in ``label_counts``.

    Raises:
        ValueError: For non-positive thresholds, or a train count larger than
            the full-dataset count (an inconsistent bookkeeping).
    """
    if head_min < 1 or torso_min < 1:
        raise ValueError("bucket thresholds head_min/torso_min must be positive")
    if head_min <= torso_min:
        raise ValueError("head_min must be strictly greater than torso_min")
    buckets: dict[str, str] = {}
    for label, count in label_counts.items():
        train_count = train_counts.get(label, 0)
        if train_count > count:
            raise ValueError(
                f"label {label!r} has more train examples ({train_count}) than "
                f"dataset examples ({count})"
            )
        if train_count == 0:
            buckets[label] = "unseen"
        elif count >= head_min:
            buckets[label] = "head"
        elif count >= torso_min:
            buckets[label] = "torso"
        else:
            buckets[label] = "tail"
    return buckets


def labels_by_bucket(buckets: Mapping[str, str]) -> dict[str, list[str]]:
    """Group labels by bucket, in BUCKET_NAMES order, alphabetical within each.

    Labels not present in the mapping are ignored (the caller validated the
    mapping against the label space already).
    """
    grouped: dict[str, list[str]] = {name: [] for name in BUCKET_NAMES}
    for label in sorted(buckets):
        bucket = buckets[label]
        if bucket in grouped:
            grouped[bucket].append(label)
    return grouped
