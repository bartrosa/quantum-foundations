"""Console entry points for publication-side reproducibility."""

from __future__ import annotations

import sys

from quantum_foundations.sedenion.edge_fano import verify_edge_fano_bijection
from quantum_foundations.sedenion.probe_verification import verify_probe_classification


def main_verify_probe() -> None:
    result = verify_probe_classification()
    print(f"Configurations checked: {result.config_count} (expect 28)")
    print(f"Probe-class histogram: {result.class_histogram}")
    if result.mismatches:
        print("FAIL")
        for line in result.mismatches[:50]:
            print(line)
        if len(result.mismatches) > 50:
            print(f"... {len(result.mismatches)} total")
        sys.exit(1)
    print(result.canonical_table_tex)
    print(
        "All 28 configurations verified: classification matches analytic proof "
        "(Lemma 3.2 + Theorems 3.4, 3.6)."
    )


def main_verify_edge_fano() -> None:
    mismatches = verify_edge_fano_bijection()
    if mismatches:
        print("FAIL")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print("All 28 bijection checks: PASS")
