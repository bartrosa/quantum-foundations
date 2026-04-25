"""Edge-colored automorphism order for Conway causets."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

import networkx as nx  # type: ignore[import-untyped]

from quantum_foundations.conway_games.causet import ConwayCauset

logger = logging.getLogger(__name__)

try:
    import pynauty  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    pynauty = None

_PYNAUTY_AVAILABLE = pynauty is not None


def _graph_for_networkx(causet: ConwayCauset) -> nx.DiGraph:
    g = nx.DiGraph()
    for i, rk in enumerate(causet.rank):
        g.add_node(i, rank=rk)
    for u, v, c in causet.hasse_edges_with_color():
        g.add_edge(u, v, color=c)
    return g


def _aut_order_networkx(causet: ConwayCauset) -> int:
    g = _graph_for_networkx(causet)
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        g,
        g,
        node_match=lambda a, b: a["rank"] == b["rank"],
        edge_match=lambda a, b: a["color"] == b["color"],
    )
    return sum(1 for _ in matcher.isomorphisms_iter())


def _pynauty_subdivided_graph(causet: ConwayCauset) -> object:
    """Build subdivided, vertex-colored directed graph preserving edge colors."""
    if pynauty is None:
        raise RuntimeError("pynauty backend requested but pynauty is not installed")
    edges = causet.hasse_edges_with_color()
    n0 = causet.n
    total = n0 + len(edges)
    adjacency: dict[int, set[int]] = {i: set() for i in range(total)}
    max_rank = max(causet.rank) if causet.rank else 0
    marker_base = max_rank + 1

    by_color: dict[int, set[int]] = defaultdict(set)
    for i, rk in enumerate(causet.rank):
        by_color[rk].add(i)

    for k, (u, v, c) in enumerate(edges):
        marker = n0 + k
        if c == "L":
            by_color[marker_base + 0].add(marker)
        elif c == "R":
            by_color[marker_base + 1].add(marker)
        else:
            by_color[marker_base + 2].add(marker)
        adjacency[u].add(marker)
        adjacency[marker].add(v)

    return pynauty.Graph(
        number_of_vertices=total,
        directed=True,
        adjacency_dict=adjacency,
        vertex_coloring=list(by_color.values()),
    )


def _aut_order_pynauty(causet: ConwayCauset) -> int:
    g = _pynauty_subdivided_graph(causet)
    _gens, s1, s2, _orbits, _num_orbits = pynauty.autgrp(g)
    return int(s1 * (10**s2))


def _resolve_backend(backend: str) -> str:
    """Resolve backend name for Conway automorphism counting.

    The Conway extension uses edge-colored subdivision graphs where ``pynauty``
    can hit severe backtracking cases; use ``networkx`` as default resolver
    for ``auto``.
    """
    if backend not in ("auto", "pynauty", "networkx"):
        raise ValueError(f"backend must be 'auto', 'pynauty', or 'networkx', got {backend!r}")
    if backend == "auto":
        if _PYNAUTY_AVAILABLE:
            logger.debug("Conway aut_order auto backend resolved to networkx")
        else:
            logger.warning("pynauty missing: Conway aut_order auto backend uses networkx")
        return "networkx"
    if backend == "pynauty" and not _PYNAUTY_AVAILABLE:
        raise RuntimeError("pynauty backend requested but pynauty is not installed")
    return backend


_AUT_BACKENDS: dict[str, Callable[[ConwayCauset], int]] = {
    "networkx": _aut_order_networkx,
    "pynauty": _aut_order_pynauty,
}


def aut_order_conway(causet: ConwayCauset, *, backend: str = "networkx") -> int:
    """Order of the rank- and edge-color-preserving automorphism group.

    Backends:
        ``networkx`` (default): VF2 isomorphism with rank node-match and
            edge-color match. This is the recommended default for Conway
            edge-colored causets.
        ``pynauty``: edge subdivision + vertex coloring encoded for nauty.
            Kept for parity checks against ``networkx``.
        ``auto``: resolves to ``networkx``.
    """
    if causet.n == 0:
        return 1
    chosen = _resolve_backend(backend)
    return _AUT_BACKENDS[chosen](causet)


def pynauty_available_conway() -> bool:
    """Return whether optional ``pynauty`` bindings are available."""
    return _PYNAUTY_AVAILABLE
