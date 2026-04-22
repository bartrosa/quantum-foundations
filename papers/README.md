# Papers

Each subdirectory is **one publication**: **`COMPANION.md`**, LaTeX sources (`paper.tex`, …), and optional **`notebooks/`** that re-run the same checks as `pytest` / CLI with section references.

## Index

| Slug | Short description |
|------|-------------------|
| [`sedenion-associator-probe/`](sedenion-associator-probe/) | Sedenion associator probe / octonion generation graph |

## Adding a new paper

1. Create `papers/<slug>/`.
2. Add `COMPANION.md` (map sections ↔ modules + `uv run …` commands).
3. Drop `main.tex` / `paper.tex`, bibliography, figures here when ready.

**Slug:** lowercase, hyphen-separated, stable (e.g. keep after the final title changes).

## Build artifacts

Prefer not committing LaTeX noise (`*.aux`, `*.log`, `*.pdf`, …). Build locally or in CI; ship PDF via release if needed.
