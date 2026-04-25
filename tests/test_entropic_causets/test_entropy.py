from __future__ import annotations

from quantum_foundations.entropic_causets.automorphism import aut_order
from quantum_foundations.entropic_causets.causet import build_stratified_game_causet, disjoint_union


def test_gibbs_identity_for_identical_pairs() -> None:
    for seed in range(5):
        causet = build_stratified_game_causet(n_target=10, max_rank=6, sample_rate=0.2, seed=seed)
        union = disjoint_union(causet, causet)
        assert aut_order(union) == 2 * aut_order(causet) * aut_order(causet)
