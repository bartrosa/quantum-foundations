"""Entropic causets: graph entropy and Verlinde-style scaling experiments.

This package implements three computational experiments on stratified **game**
causets. Entropy is reported in **both** conventions: Boltzmann ``S_A = ln n! - ln|Aut|``
(``pos``-level) and algebraic ``S_B = ln|Aut|``, with additional **twin** and
**rank** (upper-bound) decompositions; see ``docs/MATH.md`` §5. Experiments
export expanded CSV/JSON for data-driven checks (including Verlinde-style ``ΔS`` in E3).

Modules:
    causet: Frozen-dataclass representation of game causets, builder,
        disjoint union, and structural-difference helpers.
    automorphism: Computation of |Aut(C)| via two interchangeable backends
        (``pynauty``, ``networkx``).
    entropy: Position entropy, Gibbs-corrected entropy of unions of causets.
    parallel: Process-pool helpers, deterministic seed spawning,
        diagnostic-collection types for cross-process logging.
    logging_setup: Two-channel (file + stdout) logger configuration.
    experiments: Three experiment drivers (E1: |Aut| scaling; E2: Gibbs
        paradox test; E3: Verlinde distance scaling).

The package is imported as ``quantum_foundations.entropic_causets``. The public
API is re-exported here for convenience; full documentation lives in the
respective module docstrings.

References:
    * ``docs/PHYSICS.md`` — physical motivation and interpretation of results.
    * ``docs/MATH.md`` — formal definitions, Gibbs theorem, fitting protocols.
    * ``papers/entropic-causets/COMPANION.md`` — paper-side mapping of claims to
      code.
"""

from quantum_foundations.entropic_causets.automorphism import (
    aut_order,
    log_aut_order,
    pynauty_available,
)
from quantum_foundations.entropic_causets.causet import (
    GameCauset,
    StructuralDiff,
    build_stratified_game_causet,
    disjoint_union,
    make_antichain,
    make_chain,
    make_diamond,
    structural_diff,
    twin_classes,
    twin_quotient,
)
from quantum_foundations.entropic_causets.entropy import (
    EntropyDecomposition,
    entropy_decomposition,
    entropy_position,
    entropy_with_gibbs_correction,
)
from quantum_foundations.entropic_causets.logging_setup import configure_experiment_logging
from quantum_foundations.entropic_causets.parallel import TaskDiagnostic

__all__ = [
    "EntropyDecomposition",
    "GameCauset",
    "StructuralDiff",
    "TaskDiagnostic",
    "aut_order",
    "build_stratified_game_causet",
    "configure_experiment_logging",
    "disjoint_union",
    "entropy_decomposition",
    "entropy_position",
    "entropy_with_gibbs_correction",
    "log_aut_order",
    "make_antichain",
    "make_chain",
    "make_diamond",
    "pynauty_available",
    "structural_diff",
    "twin_classes",
    "twin_quotient",
]
