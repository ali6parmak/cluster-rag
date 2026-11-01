"""The corpus map's **estimator** (CS-31): the measures beside the log-odds.

CS-23's map carries one measure — the 0.5-smoothed signed log-odds — and it
is a *size*: "this word goes with this label, this much". This module adds the
measures that turn a size into a **statement**:

- the **informative-prior log-odds** — the log-odds with the word's **global
  document frequency** as the pseudo-count, in the form of the reference
  implementation (the ``logodds`` package on PyPI, MIT, no dependencies —
  Monroe, Colaresi & Quinn 2008's Dirichlet prior, "Fightin' Words"), adapted
  to the **document level**: this map's unit is the document that contains the
  word, not the token.
- its **standard error** and the **z-score** (δ over the standard error);
- the **significance** on the same 2×2 table: the likelihood-ratio **G²**
  statistic (1 degree of freedom) and its p-value;
- the **conditional smoothing**: the 0.5 added only where a cell is actually
  zero — the variant the statistics literature criticises our fixed 0.5 for
  (Jackson et al., *JRSS-M* 2021; Agresti).

**The reference implementations are oracles, not dependencies.** The formulas
below follow the published method; no package is installed (the track's pure
package may import numpy and the standard library only, and CS-31's cost line
says *no new dependency*), so the implementation is pinned by hand-computed
unit tests instead.

Pure core: the standard library only (no numpy, no model client, no I/O).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

# The reference implementation's floor: "a floor so no word gets a zero prior".
ALPHA = 0.01


@dataclass(frozen=True)
class PairMeasures:
    """The estimator's measures for one (label, word) pair.

    Attributes:
        log_odds_informative: δ — the log-odds with the word's global
            document frequency as the pseudo-count.
        standard_error: σ — the standard error of δ.
        z_score: δ / σ — the plan's "δ over the standard error".
        g2: the likelihood-ratio G² statistic on the same 2×2 table (1 df).
        p_value: the G² statistic's p-value (the chi-square 1-df survival
            function).
        log_odds_conditional: the log-odds with the 0.5 added only where a
            cell is actually zero.
    """

    log_odds_informative: float
    standard_error: float
    z_score: float
    g2: float
    p_value: float
    log_odds_conditional: float


def prior_total(document_frequencies: Iterable[int], alpha: float = ALPHA) -> float:
    """``α₀`` — the sum of the words' priors over the map's vocabulary.

    The reference implementation's ``α₀ = Σ_w α_w``: the corpus's own total
    (text, word) incidences plus the floor's contribution. It is a **map-level**
    constant (the same for every pair), computed once per map.

    Args:
        document_frequencies: Each word's document frequency.
        alpha: The pseudo-count's floor.

    Returns:
        The prior's total.
    """
    return float(sum(df + alpha for df in document_frequencies))


def conditional_log_odds(n_with: int, df: int, n_label: int, n_texts: int) -> float:
    """The log-odds with the 0.5 added **only to the cells that are zero**.

    The variant the statistics literature criticises our fixed 0.5 for: a cell
    that is already non-zero needs no pseudo-count (Jackson et al., *JRSS-M*
    2021; Agresti).

    Args:
        n_with: The corpus's texts that have both the word and the label.
        df: The corpus's texts that have the word at all.
        n_label: The corpus's texts that carry the label.
        n_texts: The corpus's size.

    Returns:
        The conditional-smoothing log-odds.
    """
    a, b, c, d = _table(n_with, df, n_label, n_texts)

    def smooth(value: int) -> float:
        return 0.5 if value == 0 else 0.0

    return math.log((a + smooth(a)) / (b + smooth(b))) + math.log((d + smooth(d)) / (c + smooth(c)))


def g2_statistic(n_with: int, df: int, n_label: int, n_texts: int) -> float:
    """The likelihood-ratio **G²** statistic on the 2×2 table (1 degree of freedom).

    ``G² = 2 Σ O ln(O/E)`` over the four cells, with ``E`` the expected counts
    under independence. It is the statistic the `quanteda.textstats::
    textstat_keyness()` family reports, on the same table the map already has.

    Args:
        n_with: The corpus's texts that have both the word and the label.
        df: The corpus's texts that have the word at all.
        n_label: The corpus's texts that carry the label.
        n_texts: The corpus's size.

    Returns:
        The G² statistic (0 for a degenerate table).
    """
    a, b, c, d = _table(n_with, df, n_label, n_texts)
    total = a + b + c + d
    if not total:
        return 0.0
    row_one = a + b
    row_two = c + d
    col_one = a + c
    col_two = b + d
    expected = (
        row_one * col_one / total,
        row_one * col_two / total,
        row_two * col_one / total,
        row_two * col_two / total,
    )

    def term(observed: int, expectation: float) -> float:
        if observed == 0:
            return 0.0
        return observed * math.log(observed / expectation)

    return 2.0 * (
        term(a, expected[0]) + term(b, expected[1]) + term(c, expected[2]) + term(d, expected[3])
    )


def g2_p_value(g2: float) -> float:
    """The G² statistic's p-value (the chi-square 1-df survival function).

    ``p = erfc(sqrt(G²/2))`` — the standard library computes it exactly, so
    the track's package needs no scipy (the pure-core rule).

    Args:
        g2: The G² statistic.

    Returns:
        The p-value, in [0, 1].
    """
    if g2 <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(g2 / 2.0))


def pair_measures(
    n_with: int,
    df: int,
    n_label: int,
    n_texts: int,
    *,
    alpha0: float,
    alpha: float = ALPHA,
) -> PairMeasures:
    """Every new measure for one (label, word) pair, from the same 2×2 table.

    Args:
        n_with: The corpus's texts that have both the word and the label.
        df: The corpus's texts that have the word at all.
        n_label: The corpus's texts that carry the label.
        n_texts: The corpus's size.
        alpha0: The map's prior total (see :func:`prior_total`).
        alpha: The pseudo-count's floor.

    Returns:
        The pair's measures.

    Raises:
        ValueError: If the counts do not form a 2×2 table.
    """
    a, b, c, d = _table(n_with, df, n_label, n_texts)
    n_other = n_texts - n_label
    prior = df + alpha
    delta_label = math.log((a + prior) / (n_label + alpha0 - a - prior))
    delta_other = math.log((b + prior) / (n_other + alpha0 - b - prior))
    delta = delta_label - delta_other
    sigma = math.sqrt(1.0 / (a + prior) + 1.0 / (b + prior))
    g2 = g2_statistic(n_with, df, n_label, n_texts)
    return PairMeasures(
        log_odds_informative=delta,
        standard_error=sigma,
        z_score=delta / sigma if sigma else 0.0,
        g2=g2,
        p_value=g2_p_value(g2),
        log_odds_conditional=conditional_log_odds(n_with, df, n_label, n_texts),
    )


def _table(n_with: int, df: int, n_label: int, n_texts: int) -> tuple[int, int, int, int]:
    """The 2×2 table of one (label, word) pair.

    ``a`` = the texts with the word **and** the label, ``b`` = the texts with
    the word but not the label, ``c`` = the texts with the label but not the
    word, ``d`` = the texts with neither.

    Raises:
        ValueError: If the counts do not form a 2×2 table.
    """
    a = n_with
    b = df - n_with
    c = n_label - n_with
    d = n_texts - n_label - b
    if min(a, b, c, d) < 0:
        raise ValueError(
            "the counts do not form a 2x2 table: "
            f"n_with={n_with}, df={df}, n_label={n_label}, n_texts={n_texts}"
        )
    return a, b, c, d
