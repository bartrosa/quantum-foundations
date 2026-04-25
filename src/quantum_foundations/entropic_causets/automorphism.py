"""Rank-preserving automorphism order ``|Aut(C)|`` for Hasse DAGs of causets.

Backends: optional ``pynauty`` (C/nauty) with vertex colours by rank, or
``networkx`` :class:`DiGraphMatcher` exact enumeration. Library code only logs; no
``basicConfig`` here.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable

import networkx as nx  # type: ignore[import-untyped]

from quantum_foundations.entropic_causets.causet import GameCauset

logger = logging.getLogger(__name__)

try:
    import pynauty  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    pynauty = None

_PYNAUTY_AVAILABLE: bool = pynauty is not None


def _graph_for_networkx(causet: GameCauset) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node, rank in enumerate(causet.rank):
        graph.add_node(node, rank=rank)
    graph.add_edges_from(causet.hasse_edges())
    return graph


def _aut_order_networkx(causet: GameCauset) -> int:
    graph = _graph_for_networkx(causet)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        graph,
        graph,
        node_match=lambda a, b: a["rank"] == b["rank"],
    )
    return sum(1 for _ in matcher.isomorphisms_iter())


def _aut_order_pynauty(causet: GameCauset) -> int:
    if pynauty is None:
        raise RuntimeError("pynauty backend requested but pynauty is not installed")

    adjacency: dict[int, set[int]] = {i: set() for i in range(causet.n)}
    for src, dst in causet.hasse_edges():
        adjacency[src].add(dst)

    by_rank: dict[int, set[int]] = defaultdict(set)
    for i, rank in enumerate(causet.rank):
        by_rank[rank].add(i)

    graph = pynauty.Graph(
        number_of_vertices=causet.n,
        directed=True,
        adjacency_dict=adjacency,
        vertex_coloring=list(by_rank.values()),
    )
    _gens, grpsize1, grpsize2, _orbits, _num_orbits = pynauty.autgrp(graph)
    return int(grpsize1 * (10**grpsize2))


_AUT_BACKENDS: dict[str, Callable[[GameCauset], int]] = {
    "networkx": _aut_order_networkx,
    "pynauty": _aut_order_pynauty,
}


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "pynauty", "networkx"):
        raise ValueError(f"backend must be 'auto', 'pynauty', or 'networkx', got {backend!r}")
    if backend == "auto":
        if _PYNAUTY_AVAILABLE:
            return "pynauty"
        logger.info("pynauty not installed; falling back to networkx")
        return "networkx"
    if backend == "pynauty" and not _PYNAUTY_AVAILABLE:
        raise RuntimeError("pynauty backend requested but pynauty is not installed")
    return backend


def aut_order(causet: GameCauset, *, backend: str = "auto") -> int:
    r"""Count the rank-preserving automorphisms of a causet's Hasse DAG.

    An automorphism is a bijection :math:`\sigma` on vertices that preserves
    the rank at each node and the immediate-past relation (covering edges of the
    Hasse diagram). This equals automorphisms of the coloured DAG with colours
    :attr:`GameCauset.rank`.

    Args:
        causet: Stratified game causet to count automorphisms of.
        backend: One of

            * ``"auto"`` — use pynauty if installed, else networkx;
            * ``"pynauty"`` — require nauty (raises if missing);
            * ``"networkx"`` — exact VF2-style self-isomorphism count (slow for large n).

    Returns:
        Group order in ``int`` (always :math:`\geq 1`).

    Raises:
        RuntimeError: If ``"pynauty"`` is selected but the package is not installed.
        ValueError: If *backend* is not one of the allowed names.

    Note:
        The antichain on *N* nodes at rank 0 has order *N!*; Python ``int`` avoids overflow.

    Example:
        >>> from quantum_foundations.entropic_causets.causet import make_antichain
        >>> aut_order(make_antichain(4), backend="networkx")
        24
    """
    if causet.n == 0:
        return 1
    chosen = _resolve_backend(backend)
    logger.debug(
        "aut_order | n=%d backend_requested=%s backend_used=%s",
        causet.n,
        backend,
        chosen,
    )
    t0 = time.perf_counter()
    try:
        order = _AUT_BACKENDS[chosen](causet)
    except Exception:
        logger.exception("aut_order failed | n=%d backend=%s", causet.n, chosen)
        raise
    elapsed = time.perf_counter() - t0
    logger.debug(
        "aut_order ok | n=%d backend=%s order=%d elapsed_s=%.4f",
        causet.n,
        chosen,
        order,
        elapsed,
    )
    if elapsed > 5.0:
        logger.warning(
            "aut_order slow | n=%d backend=%s elapsed_s=%.2f — consider pynauty",
            causet.n,
            chosen,
            elapsed,
        )
    return order


def log_aut_order(causet: GameCauset, *, backend: str = "auto") -> float:
    """Return ``log(|Aut|)`` with the same *backend* semantics as :func:`aut_order`."""
    return math.log(aut_order(causet, backend=backend))


def pynauty_available() -> bool:
    """Return whether the optional pynauty (nauty) Python bindings are importable."""
    return _PYNAUTY_AVAILABLE
