"""Theorem edge-to-Fano bijection (paper Theorem `thm:edge-fano-bijection`)."""

from __future__ import annotations

import itertools

from quantum_foundations.sedenion.fano import FANO_LINES, is_fano_line


def phi_for_l(l0: frozenset[int], g: tuple[int, ...], ell: int) -> frozenset[int]:
    """Unique pair {p,q} ⊂ {1,2,3} with {ell, g_p, g_q} a Fano line."""
    pairs: list[tuple[int, int]] = []
    for p, q in ((0, 1), (0, 2), (1, 2)):
        gp, gq = g[p], g[q]
        if is_fano_line(frozenset({ell, gp, gq})):
            pairs.append((p + 1, q + 1))
    if len(pairs) != 1:
        raise RuntimeError(f"expected one pair for ell={ell}, got {pairs}")
    a, b = pairs[0]
    return frozenset({min(a, b), max(a, b)})


def latex_tabular_cell(edge: frozenset[int]) -> str:
    i, j = sorted(edge)
    return f"$\\{{{i},{j}\\}}$"


def verify_edge_fano_bijection() -> list[str]:
    """Return mismatch messages; empty list means all 28 rows satisfy the bijection."""
    mismatches: list[str] = []

    for l0 in sorted(FANO_LINES, key=lambda s: tuple(sorted(s))):
        comp = sorted(set(range(1, 8)) - set(l0))
        for trip in itertools.combinations(comp, 3):
            g = tuple(sorted(trip))
            mapped: list[frozenset[int]] = []
            for ell in sorted(l0):
                mapped.append(phi_for_l(l0, g, ell))
            target = frozenset(
                {
                    frozenset({1, 2}),
                    frozenset({1, 3}),
                    frozenset({2, 3}),
                }
            )
            if frozenset(mapped) != target:
                mismatches.append(f"bijection fail L0={sorted(l0)} g={g} maps={mapped}")

    return mismatches


def render_edge_fano_table_tex() -> str:
    """Full LaTeX tabular for the 28 configurations (for pasting into paper)."""
    rows_tex: list[str] = []
    for l0 in sorted(FANO_LINES, key=lambda s: tuple(sorted(s))):
        comp = sorted(set(range(1, 8)) - set(l0))
        for trip in itertools.combinations(comp, 3):
            g = tuple(sorted(trip))
            mapped = [phi_for_l(l0, g, ell) for ell in sorted(l0)]
            row = (
                f"${{{','.join(map(str, sorted(l0)))}}}$ & "
                f"${{{','.join(map(str, g))}}}$ & "
                + " & ".join(latex_tabular_cell(e) for e in mapped)
                + r" \\"
            )
            rows_tex.append(row)

    header = (
        r"\begin{tabular}{l|l|ccc}"
        "\n"
        r"\hline"
        "\n"
        r"$L_0$ & $(g_1,g_2,g_3)$ & $\varphi(\mathrm{1st})$ & "
        r"$\varphi(\mathrm{2nd})$ & $\varphi(\mathrm{3rd})$ \\"
        "\n"
        r"\hline"
    )
    body = "\n".join(rows_tex)
    footer = "\n" + r"\hline" + "\n" + r"\end{tabular}"
    return header + "\n" + body + footer
