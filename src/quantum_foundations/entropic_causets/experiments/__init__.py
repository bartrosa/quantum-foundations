"""Experiment drivers: E1 (|Aut| scaling), E2 (Gibbs checks), E3 (Verlinde distance).

Call :func:`run` on ``e1_aut_scaling``, ``e2_gibbs``, or ``e3_verlinde`` after configuring
``quantum_foundations`` logging in the main process. Entry points: ``qf-run-e1``,
``qf-run-e2``, ``qf-run-e3`` in :mod:`quantum_foundations.cli`.
"""

from quantum_foundations.entropic_causets.experiments.e1_aut_scaling import (
    E1LinearFit,
    E1Result,
    E1Row,
)
from quantum_foundations.entropic_causets.experiments.e2_gibbs import E2Result, E2Row, E2Summary
from quantum_foundations.entropic_causets.experiments.e3_verlinde import E3Fit, E3Result, E3Row

__all__ = [
    "E1LinearFit",
    "E1Result",
    "E1Row",
    "E2Result",
    "E2Row",
    "E2Summary",
    "E3Fit",
    "E3Result",
    "E3Row",
]
