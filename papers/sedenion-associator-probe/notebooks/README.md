# Verification notebooks (sedenion associator probe)

| Notebook | Paper references |
|----------|-------------------|
| [`01_octonion_table_and_fano.ipynb`](01_octonion_table_and_fano.ipynb) | Appendix `tab:oct`; Fano incidence / quaternion triples |
| [`02_probe_matrices_and_classification.ipynb`](02_probe_matrices_and_classification.ipynb) | §3 — 28 configs, entries $\\{0,2\\}$, Democratic/Edge/Hub/Zero, `(c₁,c₀)` |
| [`03_edge_fano_bijection.ipynb`](03_edge_fano_bijection.ipynb) | Theorem edge–Fano bijection |

From repository root (so `quantum_foundations` is importable):

```bash
uv sync --all-extras
uv run jupyter notebook papers/sedenion-associator-probe/notebooks/
```

Headless check (optional):

```bash
uv run jupyter nbconvert --execute --inplace papers/sedenion-associator-probe/notebooks/*.ipynb
```
