# quantum-foundations

Reproducible **exact** Python verification for mathematical claims in publications (starting with sedenion/octonion probe matrices).

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) for environments and locking

## Setup

```bash
git clone https://github.com/bartrosa/quantum-foundations.git
cd quantum-foundations
uv sync --all-extras      # installs package in editable mode + dev tools (pytest, ruff, mypy)
```

Developers typically run checks before committing:

```bash
make check                # ruff + pytest + mypy
```

## Papers

Manuscript companions, LaTeX, and verification notebooks live under [`papers/`](papers/) — see [`papers/README.md`](papers/README.md). Example: [`papers/sedenion-associator-probe/notebooks/`](papers/sedenion-associator-probe/notebooks/).

## Sedenion / octonion generation graph paper

- **Paper folder** (companion + future LaTeX): [`papers/sedenion-associator-probe/`](papers/sedenion-associator-probe/)
- **Companion** (sections ↔ modules, reproduction commands): [`papers/sedenion-associator-probe/COMPANION.md`](papers/sedenion-associator-probe/COMPANION.md)
- **Library API**: `quantum_foundations.sedenion` — algebra, Fano lines, probe verification, edge–Fano checks.
- **CLI** (after `uv sync`):

  | Command | Meaning |
  |---------|---------|
  | `uv run qf-verify-probe` | All 28 configurations; entries + characteristic polynomial signatures + canonical LaTeX snippet |
  | `uv run qf-verify-edge-fano` | All 28 edge–Fano bijection checks |

## Running tests only

```bash
uv run pytest
```

## Makefile shortcuts

| Target | Command |
|--------|---------|
| `make sync` | `uv sync --all-extras` |
| `make test` | `uv run pytest` |
| `make lint` | `uv run ruff check src tests` |
| `make fmt` | `uv run ruff format src tests` |
| `make check` | lint + test + mypy |
| `make verify-probe` | `uv run qf-verify-probe` |
| `make verify-edge-fano` | `uv run qf-verify-edge-fano` |
| `make notebooks-execute` | headless Jupyter on `papers/sedenion-associator-probe/notebooks/*.ipynb` |
| `make paper` | `pdflatex`+`bibtex` for `papers/sedenion-associator-probe/paper.tex` |

If your GitHub username or repo name differs from `bartrosa/quantum-foundations`, update the clone URL here and the `\url{...}` in the paper acknowledgments.

## License

MIT — see `LICENSE`.
