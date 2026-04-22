"""Associator norm matrices, characteristic polynomial classes (paper §3)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

from quantum_foundations.sedenion.algebra import basis_s, s_assoc, s_norm_sq
from quantum_foundations.sedenion.fano import FANO_LINES, is_fano_line


def classify_case(m: int, a: int, b: int) -> str:
    """Case label (A/B/C/D1/D2) for entry A^(8+m)_{pq} (Lemma 3.5 case analysis)."""
    if m == 0:
        return "A"
    if m == a:
        return "B"
    if m == b:
        return "C"
    if is_fano_line(frozenset({m, a, b})):
        return "D1"
    return "D2"


def predict_entry(m: int, a: int, b: int) -> int:
    return 0 if classify_case(m, a, b) == "D2" else 2


def predict_class(m: int, l0: frozenset[int], triple: Sequence[int]) -> str:
    comp = sorted(set(range(1, 8)) - set(l0))
    gset = frozenset(triple)
    rest = set(comp) - set(gset)
    assert len(rest) == 1
    g4 = next(iter(rest))
    if m == 0:
        return "Democratic"
    if m in l0:
        return "Edge"
    if m in gset:
        return "Hub"
    if m == g4:
        return "Zero"
    raise AssertionError(f"impossible m={m}")


def char_poly_traceless_3x3(a: Sequence[Sequence[int]]) -> tuple[int, int]:
    """det(λI - A) = λ³ + c1 λ + c0 for trace-zero 3×3."""
    a00, a01, a02 = a[0]
    a10, a11, a12 = a[1]
    a20, a21, a22 = a[2]
    assert int(a00 + a11 + a22) == 0
    m0 = int(a11 * a22 - a12 * a21)
    m1 = int(a00 * a22 - a02 * a20)
    m2 = int(a00 * a11 - a01 * a10)
    c1 = m0 + m1 + m2
    det = int(
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    return c1, -det


CLASS_SIGS: dict[str, tuple[int, int]] = {
    "Democratic": (-12, -16),
    "Edge": (-4, 0),
    "Hub": (-8, 0),
    "Zero": (0, 0),
}


def render_canonical_table_tex() -> str:
    """Canonical row: L0={1,2,3}, g=(4,5,6), g4=7 — LaTeX tabular (Appendix-style)."""
    g = (4, 5, 6)
    lines = [
        r"\begin{tabular}{r|ccc}",
        r"\hline",
        r"$k\setminus (p,q)$ & $(1,2)$ & $(1,3)$ & $(2,3)$ \\",
        r"\hline",
    ]
    for k in range(8, 16):
        m = k - 8
        row_tex = [str(k)]
        for p, q in ((0, 1), (0, 2), (1, 2)):
            gi, gj = g[p], g[q]
            assoc = s_assoc(basis_s(gi), basis_s(gj), basis_s(k))
            n = int(math.isqrt(s_norm_sq(assoc)))
            lab = classify_case(m, gi, gj)
            row_tex.append(f"{n} ({lab})")
        lines.append(" & ".join(row_tex) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


@dataclass(frozen=True)
class ProbeVerificationResult:
    mismatches: list[str]
    config_count: int
    class_histogram: dict[str, int]
    canonical_table_tex: str


def verify_probe_classification() -> ProbeVerificationResult:
    """
    Exhaustive check over 7 Fano lines L0 and 4 complementary triples each (28 configs).

    Matches claims in paper.tex: Lemma 3.2 (entries in {0,2}), Theorems 3.4–3.6
    (characteristic polynomial classes), Table with (c1,c0) for Democratic/Edge/Hub/Zero.
    """
    mismatches: list[str] = []
    class_counts: dict[str, int] = {}
    configs = 0

    for l0 in sorted(FANO_LINES, key=lambda s: tuple(sorted(s))):
        comp = sorted(set(range(1, 8)) - set(l0))
        for trip in itertools.combinations(comp, 3):
            g = tuple(sorted(trip))
            configs += 1
            for k in range(8, 16):
                m = k - 8
                cls = predict_class(m, l0, g)
                class_counts[cls] = class_counts.get(cls, 0) + 1

                matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                for p in range(3):
                    for q in range(3):
                        if p == q:
                            continue
                        gi, gj = g[p], g[q]
                        assoc = s_assoc(basis_s(gi), basis_s(gj), basis_s(k))
                        entry = int(math.isqrt(s_norm_sq(assoc)))
                        if entry * entry != s_norm_sq(assoc):
                            mismatches.append(f"non-square norm L0={sorted(l0)} g={g} k={k}")
                        pred = predict_entry(m, gi, gj)
                        lab = classify_case(m, gi, gj)
                        if entry != pred:
                            mismatches.append(
                                f"ENTRY L0={sorted(l0)} g={g} k={k} ({p},{q}) "
                                f"case={lab} pred={pred} got={entry}"
                            )
                        matrix[p][q] = entry

                c1, c0 = char_poly_traceless_3x3(matrix)
                exp = CLASS_SIGS[cls]
                if (c1, c0) != exp:
                    mismatches.append(
                        f"POLY L0={sorted(l0)} g={g} k={k} cls={cls} got ({c1},{c0}) want {exp}"
                    )
                tr = sum(matrix[i][i] for i in range(3))
                if tr != 0:
                    mismatches.append(f"trace nonzero k={k} L0={sorted(l0)} g={g}")

    return ProbeVerificationResult(
        mismatches=mismatches,
        config_count=configs,
        class_histogram=dict(sorted(class_counts.items())),
        canonical_table_tex=render_canonical_table_tex(),
    )
