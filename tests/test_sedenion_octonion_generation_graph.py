"""
Regression tests for the sedenion associator probe paper (sit draft → publication).

Numeric/thesis alignment with paper.tex:
- Abstract / §3: 28 configurations (7 Fano lines × C(4,3)); multiplicities 1+3+3+1 per probe block.
- Table (Democratic / Edge / Hub / Zero): (c1, c0) = (-12,-16), (-4,0), (-8,0), (0,0).
- Theorem edge–Fano: φ maps the three vertices of L0 to the three edges of K3 on labels {1,2,3}.
"""

from __future__ import annotations

import math

from quantum_foundations.sedenion.algebra import OCT_TAB, basis_s, s_assoc, s_norm_sq
from quantum_foundations.sedenion.edge_fano import verify_edge_fano_bijection
from quantum_foundations.sedenion.fano import FANO_LINES
from quantum_foundations.sedenion.probe_verification import (
    CLASS_SIGS,
    char_poly_traceless_3x3,
    verify_probe_classification,
)


def test_basis_associator_norms_are_perfect_squares() -> None:
    """Associator norms in ℤ are squares (entries 0 or 2 as √n in paper)."""
    for i in range(8):
        for j in range(8):
            for k in range(8):
                a, b, c = basis_s(i), basis_s(j), basis_s(k)
                n = s_norm_sq(s_assoc(a, b, c))
                r = int(math.isqrt(n))
                assert r * r == n


def test_fano_lines_count_and_incidence() -> None:
    assert len(FANO_LINES) == 7
    # each of 7 points lies on 3 lines (PG(2,2))
    for p in range(1, 8):
        on = sum(1 for ln in FANO_LINES if p in ln)
        assert on == 3


def test_CLASS_SIGS_match_paper_table() -> None:
    """paper.tex rows: Democratic (-12,-16); Edge (-4,0); Hub (-8,0); Zero (0,0)."""
    assert CLASS_SIGS["Democratic"] == (-12, -16)
    assert CLASS_SIGS["Edge"] == (-4, 0)
    assert CLASS_SIGS["Hub"] == (-8, 0)
    assert CLASS_SIGS["Zero"] == (0, 0)


def test_democratic_charpoly_matches_lambda_cube_minus_12_lambda_minus_16() -> None:
    """Paper: Democratic → λ³ - 12λ - 16 i.e. (c1,c0)=(-12,-16)."""
    # A = 2(J-I) on 3 vertices: diagonal 0, off-diagonal 2
    a_demo = [[0, 2, 2], [2, 0, 2], [2, 2, 0]]
    c1, c0 = char_poly_traceless_3x3(a_demo)
    assert (c1, c0) == (-12, -16)


def test_edge_hub_charpolys_match_paper() -> None:
    """Paper: Edge λ³-4λ; Hub λ³-8λ → (c1,c0)=(-4,0) and (-8,0)."""
    # One edge of weight 2 on 3 vertices (path P3 scaled)
    a_edge = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]
    assert char_poly_traceless_3x3(a_edge) == (-4, 0)
    # Star at vertex 0
    a_hub = [[0, 2, 2], [2, 0, 0], [2, 0, 0]]
    assert char_poly_traceless_3x3(a_hub) == (-8, 0)


def test_probe_verification_exhaustive_no_mismatches() -> None:
    result = verify_probe_classification()
    assert result.config_count == 28
    assert result.mismatches == []
    assert result.class_histogram == {
        "Democratic": 28,
        "Edge": 84,
        "Hub": 84,
        "Zero": 28,
    }


def test_histogram_matches_paper_multiplicities_1_plus_3_plus_3_plus_1() -> None:
    """Per configuration: 8 probes split 1+3+3+1; over 28 configs multiply by 28."""
    result = verify_probe_classification()
    total = sum(result.class_histogram.values())
    assert total == 28 * 8 == 224
    assert result.class_histogram["Democratic"] == 28 * 1
    assert result.class_histogram["Edge"] == 28 * 3
    assert result.class_histogram["Hub"] == 28 * 3
    assert result.class_histogram["Zero"] == 28 * 1


def test_edge_fano_bijection_all_configurations() -> None:
    assert verify_edge_fano_bijection() == []


def test_oct_tab_agrees_with_imaginary_row_in_code() -> None:
    """Row 1: e1*e0=e1, e1*e1=-e0, e1*e2=e3, ... (must match `paper.tex` table)."""
    assert OCT_TAB[1][0] == (1, 1)
    assert OCT_TAB[1][1] == (-1, 0)
    assert OCT_TAB[1][2] == (1, 3)
    assert OCT_TAB[1][3] == (-1, 2)


def test_canonical_table_contains_expected_case_labels() -> None:
    tex = verify_probe_classification().canonical_table_tex
    assert "(A)" in tex or " (A)" in tex
    assert "(D1)" in tex or "(D2)" in tex
