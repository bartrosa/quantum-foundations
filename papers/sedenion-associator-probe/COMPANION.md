# Companion: Sedenion associator probe / octonion generation graph paper

Exact checks that accompany the publication draft (`sit`, then arXiv/journal).

## Claims checked in code

| Paper (draft `paper.tex`) | Implementation |
|---------------------------|----------------|
| §3, Lemma entry quantization (`{0,2}` norms); Lemma 3.5 cases A/B/C/D1/D2 | `quantum_foundations.sedenion.probe_verification:classify_case`, `predict_entry` |
| §3, Theorems 3.4–3.6: characteristic polynomial classes Democratic / Edge / Hub / Zero with `(c₁,c₀)=(-12,-16),(-4,0),(-8,0),(0,0)` | `CLASS_SIGS`, `char_poly_traceless_3x3`, `verify_probe_classification` |
| Abstract / §3: **28** configurations `(L₀,(g₁,g₂,g₃))`; multiplicities **1+3+3+1** over eight probes | `verify_probe_classification` → `config_count==28`, histogram `{Democratic:28, Edge:84, Hub:84, Zero:28}` over all probe instances |
| Theorem edge–Fano bijection (`thm:edge-fano-bijection`) | `verify_edge_fano_bijection` |
| Appendix octonion multiplication table | `quantum_foundations.sedenion.algebra.OCT_TAB` (rows match `paper.tex` Table `tab:oct`) |

## How to reproduce

From repo root:

```bash
uv sync --all-extras
uv run pytest -q
uv run qf-verify-probe       # exhaustive §3-style matrix checks + canonical LaTeX table
uv run qf-verify-edge-fano    # bijection theorem
```

Or `make sync && make check && make verify-probe`.

### Jupyter notebooks

Interactive checks mirroring the main theorems: [`notebooks/README.md`](notebooks/README.md). Start Jupyter from repo root after `uv sync --all-extras`.

### LaTeX

Sources: [`paper.tex`](paper.tex), [`references.bib`](references.bib). Build (requires `pdflatex`, `bibtex`):

```bash
cd papers/sedenion-associator-probe && chmod +x build.sh && ./build.sh paper
```

## Canonical configuration

The printed LaTeX tabular (`render_canonical_table_tex` / CLI output) uses the fixed illustrative choice `L₀={1,2,3}`, complementary triple `(g₁,g₂,g₃)=(4,5,6)`, `g₄=7`, matching the narrative in §3 proofs.
