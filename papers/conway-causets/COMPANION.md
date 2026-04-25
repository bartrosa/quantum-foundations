# Companion: conway causets

Companion for the Conway extension (`e = {P_L | P_R}`) with edge-colored automorphisms.

## Claims checked in code

| Hypothesis / docs section | Implementation | CLI |
|---|---|---|
| Conway colored automorphism scaling (E4) | `quantum_foundations.conway_games.experiments.e4_conway_aut_scaling` | `qf-run-e4` |
| Four-valued outcome distribution (E5) | `quantum_foundations.conway_games.outcome` + `...e5_outcome_distribution` | `qf-run-e5` |
| Hierarchy collapse rates (E6) | `quantum_foundations.conway_games.entropy` + `...e6_hierarchy_collapse` | `qf-run-e6` |
| Gibbs identities on Conway unions (E7) | `quantum_foundations.conway_games.experiments.e7_conway_gibbs` | `qf-run-e7` |

## Reproduce

```bash
uv sync --all-extras
uv pip install -e ".[fast-aut]"   # optional but recommended
make run-conway
```
