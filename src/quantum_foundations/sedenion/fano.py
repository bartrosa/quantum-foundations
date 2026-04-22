"""Fano lines from quaternion triples in the chosen octonion table."""

from __future__ import annotations

from quantum_foundations.sedenion.algebra import OCT_TAB


def compute_fano_lines() -> frozenset[frozenset[int]]:
    triples: set[frozenset[int]] = set()
    for a in range(1, 8):
        for b in range(a + 1, 8):
            _sg, k = OCT_TAB[a][b]
            triples.add(frozenset({a, b, k}))
    if len(triples) != 7:
        raise RuntimeError(f"expected 7 Fano lines, got {len(triples)}")
    return frozenset(triples)


FANO_LINES = compute_fano_lines()


def is_fano_line(triple: frozenset[int]) -> bool:
    return triple in FANO_LINES
