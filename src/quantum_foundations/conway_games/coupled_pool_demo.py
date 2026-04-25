"""Demo: coupled-pool Conway causet builder vs pool_overlap regimes."""

from __future__ import annotations

from quantum_foundations.conway_games.causet import build_coupled_pool_conway_causet
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway


def main() -> None:
    n_target = 20
    max_rank = 4
    n_seeds = 30

    for pool_overlap in (0.0, 0.5, 1.0):
        log_aut_sum = 0.0
        lr_edge_sum = 0
        for seed in range(n_seeds):
            c = build_coupled_pool_conway_causet(
                n_target=n_target,
                max_rank=max_rank,
                pool_overlap=pool_overlap,
                sample_rate=0.4,
                seed=seed,
            )
            d = entropy_decomposition_conway(c)
            log_aut_sum += d.log_aut_pos
            lr_edge_sum += sum(len(c.past_left[i] & c.past_right[i]) for i in range(c.n))
        print(
            f"pool_overlap={pool_overlap:.1f}  "
            f"mean_log_aut={log_aut_sum / n_seeds:.4f}  "
            f"mean_lr_edges={lr_edge_sum / n_seeds:.2f}"
        )


if __name__ == "__main__":
    main()
