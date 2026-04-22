"""Sedenion / octonion probe machinery (associator norms, Fano incidence)."""

from quantum_foundations.sedenion.algebra import (
    OCT_TAB,
    basis_o,
    basis_s,
    o_add,
    o_conj,
    o_mul,
    o_norm_sq,
    o_sub,
    s_assoc,
    s_mul,
    s_norm_sq,
    s_sub,
)
from quantum_foundations.sedenion.edge_fano import (
    latex_tabular_cell,
    phi_for_l,
    verify_edge_fano_bijection,
)
from quantum_foundations.sedenion.fano import FANO_LINES, compute_fano_lines, is_fano_line
from quantum_foundations.sedenion.probe_verification import (
    CLASS_SIGS,
    char_poly_traceless_3x3,
    classify_case,
    predict_class,
    predict_entry,
    render_canonical_table_tex,
    verify_probe_classification,
)

__all__ = [
    "CLASS_SIGS",
    "FANO_LINES",
    "OCT_TAB",
    "basis_o",
    "basis_s",
    "char_poly_traceless_3x3",
    "classify_case",
    "compute_fano_lines",
    "is_fano_line",
    "latex_tabular_cell",
    "o_add",
    "o_conj",
    "o_mul",
    "o_norm_sq",
    "o_sub",
    "phi_for_l",
    "predict_class",
    "predict_entry",
    "render_canonical_table_tex",
    "s_assoc",
    "s_mul",
    "s_norm_sq",
    "s_sub",
    "verify_edge_fano_bijection",
    "verify_probe_classification",
]
