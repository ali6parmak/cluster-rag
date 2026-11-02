"""The **label-blind counterpart** (the track's own yardstick).

A control on the *labels*, not on the words: the train records' label sets are
permuted among the records, so the **per-label counts are preserved** (only the
assignment changes) and every word↔label association the map learns is
destroyed. Whatever a channel buys on the blind counterpart, it bought on
noise.

The permutation is the track's own: CS-32's blind corpus-map feature and
CS-36's blind classifier arm are the **same** control, so the two are read
against each other.

Pure core: numpy and the standard library only (no model client, no I/O).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def permuted_label_sets(
    label_sets: Sequence[Sequence[str]],
    seed: int,
) -> list[list[str]]:
    """The records' label sets permuted among the records.

    The per-label counts are preserved (only the assignment changes), so the
    map's *construction* is the same and only the word↔label association is
    destroyed.

    Args:
        label_sets: Each record's labels.
        seed: The permutation's seed (deterministic).

    Returns:
        The permuted label sets, in the records' own order.
    """
    order = np.random.default_rng(seed).permutation(len(label_sets))
    return [list(label_sets[int(index)]) for index in order]
