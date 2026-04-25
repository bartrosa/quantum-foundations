# Notebooks

Interactive views for entropic-causet experiments:

- `01_aut_scaling.ipynb`
- `02_gibbs_test.ipynb`
- `03_verlinde_distance.ipynb`

From the repository root, sync dependencies and start Jupyter (uses the same environment as `uv run`):

```bash
uv sync
uv run jupyter lab
```

In the notebook UI, select the **Python 3** kernel for this project (the one backed by the `.venv` that `uv` created). The package root lists `pandas` and `matplotlib`, so `import pandas` works after a plain `uv sync` without extra optional groups.
