"""Positional entropy, full entropy decompositions, and Gibbs-corrected unions."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from quantum_foundations.entropic_causets.automorphism import aut_order
from quantum_foundations.entropic_causets.causet import GameCauset, twin_classes, twin_quotient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EntropyDecomposition:
    """Entropy decomposition of a causet at three coarse-graining levels.

    For each level we report the algebraic (``S^B = ln|Aut|``) and Boltzmann
    (``S^A = ln(n!) - ln|Aut|``) conventions, plus orbit-loss terms between
    levels.

    Attributes:
        aut_order_pos: ``|Aut(C)|`` (rank + past-preserving DAG automorphisms).
        aut_order_twin: ``|Aut(C / ~_twin)|`` on the twin quotient.
        rank_factorial_product: ``\\prod_r n_r!`` (size of the rank-only group).
        log_aut_pos: ``\\ln|Aut(C)|`` (S_B at pos level).
        log_aut_twin: ``\\ln|Aut(C/\\sim_{twin})|`` (S_B at twin level).
        log_rank_factorial: ``\\sum_r \\ln n_r!`` (S_B at rank / upper bound).
        log_n_factorial: ``\\ln n!`` (universal anchor).
        s_a_pos: ``\\ln n! - \\ln|Aut(C)|`` (S_A, pos).
        s_a_twin: ``\\ln n! - \\ln|Aut(C/\\sim)|`` (S_A, twin).
        s_a_rank: ``\\ln n! - \\sum_r \\ln n_r!`` (S_A, rank / labeling within layers).
        delta_max_pos: ``S_{rank}^B - S_{pos}^B`` (orbit loss, rank to pos).
        delta_pos_twin: ``S_{pos}^B - S_{twin}^B`` (orbit loss, pos to twin).
        twin_class_sizes: Class sizes, sorted descending (includes singletons).
        n: Event count of the original causet.
    """

    aut_order_pos: int
    aut_order_twin: int
    rank_factorial_product: int
    log_aut_pos: float
    log_aut_twin: float
    log_rank_factorial: float
    log_n_factorial: float
    s_a_pos: float
    s_a_twin: float
    s_a_rank: float
    delta_max_pos: float
    delta_pos_twin: float
    twin_class_sizes: tuple[int, ...]
    n: int


def _rank_factorial_product(causet: GameCauset) -> int:
    """``\\prod_r n_r!`` as an exact ``int`` (``1`` for ``n=0``)."""
    c = Counter(causet.rank)
    p = 1
    for _r, nr in c.items():
        for k in range(2, nr + 1):
            p *= k
    return p


def _twin_class_sizes(causet: GameCauset) -> tuple[int, ...]:
    if causet.n == 0:
        return tuple()
    return tuple(sorted((len(t) for t in twin_classes(causet)), reverse=True))


def _log_n_factorial(n: int) -> float:
    if n <= 0:
        return 0.0
    return float(math.lgamma(float(n) + 1.0))


def _log_int_order(order: int) -> float:
    if order <= 1:
        return 0.0
    return float(math.log(float(order)))


def entropy_decomposition(causet: GameCauset, *, backend: str = "auto") -> EntropyDecomposition:
    """Compute three coarse-graining levels in both sign conventions and orbit terms.

    Args:
        causet: Input game causet.
        backend: Forwarded to :func:`aut_order` (``"auto"``, ``"pynauty"``,
            ``"networkx"``).

    Returns:
        A frozen :class:`EntropyDecomposition` for *causet*.
    """
    n = causet.n
    o_pos = aut_order(causet, backend=backend)
    o_twin = aut_order(twin_quotient(causet), backend=backend)
    rfp = _rank_factorial_product(causet)
    by_rank = Counter(causet.rank)
    log_rk: float = sum(float(math.lgamma(float(nr) + 1.0)) for nr in by_rank.values())
    log_n = _log_n_factorial(n)
    log_p = _log_int_order(o_pos)
    log_t = _log_int_order(o_twin)
    s_a_p = log_n - log_p
    s_a_tw = log_n - log_t
    s_a_rk = log_n - log_rk
    d_mp = log_rk - log_p
    d_pt = log_p - log_t
    t_sizes = _twin_class_sizes(causet)
    return EntropyDecomposition(
        aut_order_pos=o_pos,
        aut_order_twin=o_twin,
        rank_factorial_product=rfp,
        log_aut_pos=log_p,
        log_aut_twin=log_t,
        log_rank_factorial=log_rk,
        log_n_factorial=log_n,
        s_a_pos=s_a_p,
        s_a_twin=s_a_tw,
        s_a_rank=s_a_rk,
        delta_max_pos=d_mp,
        delta_pos_twin=d_pt,
        twin_class_sizes=t_sizes,
        n=n,
    )


def entropy_position(causet: GameCauset, *, backend: str = "auto") -> float:
    """``ln|Aut|`` for rank- and cover-preserving automorphisms of *causet* (S_B, pos)."""
    return entropy_decomposition(causet, backend=backend).log_aut_pos


def _causet_signature(causet: GameCauset) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    return causet.rank, tuple(tuple(sorted(p)) for p in causet.past)


def entropy_with_gibbs_correction(causets: Sequence[GameCauset]) -> float:
    r"""Sum of ``\ln|Aut(C_i)|`` plus the Gibbs term ``\ln\prod_\tau N_\tau!``.

    Isomorphic components in the sequence are counted with factorial multiplicity
    for permuting indistinguishable copies, matching the standard Gibbs bookkeeping.

    Args:
        causets: List of (possibly isomorphic) components.

    Returns:
        Total natural-log entropy.

    Note:
        For singleton isomorphism classes, ``\ln(1!) = 0``; the loop uses
        :func:`math.lgamma` as ``\ln\Gamma(N_\tau+1) = \ln(N_\tau!)``.
    """
    n = len(causets)
    logger.debug("entropy_with_gibbs | n_causets=%d", n)
    base: float = sum(entropy_position(c) for c in causets)
    class_counts = Counter(_causet_signature(c) for c in causets)
    gibbs_term: float = 0.0
    for sz in class_counts.values():
        gibbs_term += float(math.lgamma(float(sz) + 1.0))
    if any(sz > 1 for sz in class_counts.values()):
        logger.info(
            "Gibbs correction active | iso_class_sizes=%s | extra_from_factorials=%.4f",
            sorted(class_counts.values(), reverse=True),
            gibbs_term,
        )
    return base + gibbs_term
