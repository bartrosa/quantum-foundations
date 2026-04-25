# Companion: entropic causets

Companion to the entropic-causet experiment suite built on rank-stratified game causets.

## Claims checked in code

| Hypothesis / docs section (MATH / PHYSICS) | Implementation | CLI |
|---|---|---|
| Gibbs theorem (MATH §4) | `quantum_foundations.entropic_causets.entropy:entropy_with_gibbs_correction` + `tests/test_entropic_causets/test_entropy.py` | `qf-run-e2` |
| \|Aut\| scaling (PHYSICS §5, MATH §7) | `quantum_foundations.entropic_causets.experiments.e1_aut_scaling` | `qf-run-e1` |
| Verlinde-style ΔS(r) (PHYSICS §4, MATH §8) | `quantum_foundations.entropic_causets.experiments.e3_verlinde` | `qf-run-e3` |

## How to reproduce

```bash
uv sync
uv sync --all-extras   # if you also want dev tools (ruff, pytest, notebook) in one go
uv pip install -e ".[fast-aut]"   # optional faster |Aut| (pynauty)
make run-all
# Outputs: papers/entropic-causets/results/*.csv and *.json
```

`pandas` and `matplotlib` are project dependencies, so a plain `uv sync` is enough for the notebook imports.
