from __future__ import annotations

from pathlib import Path

from quantum_foundations.conway_games.experiments import e8_generator_comparison


def test_e8_enumerate_tasks_count() -> None:
    """Verify default task enumeration has the expected size on a small grid."""
    tasks = e8_generator_comparison.enumerate_tasks(n_values=(15, 20), n_seeds=2)
    # Stochastic: 5 generators × 2 N × 2 seeds = 20
    # Deterministic: 4 + 4 + 3 + 3 + 4 = 18
    expected_stochastic = 5 * 2 * 2
    expected_deterministic = 4 + 4 + 3 + 3 + 4
    assert len(tasks) == expected_stochastic + expected_deterministic
    assert len(tasks) == 20 + 18


def test_e8_smoke(tmp_path: Path) -> None:
    """End-to-end on a tiny config; CSV/JSON and deterministic |Aut| baseline."""
    result = e8_generator_comparison.run(
        n_workers=2,
        seeds=1,
        max_n=15,
        log_queue=None,
        task_timeout_s=120.0,
    )
    # n_values (15) only, 5 stochastic + 18 deterministic
    assert len(result.rows) == 23
    e8_generator_comparison.write_outputs(result, tmp_path)
    csv_path = tmp_path / "e8_generator_comparison.csv"
    assert csv_path.is_file()
    json_path = tmp_path / "e8_generator_comparison.json"
    assert json_path.is_file()

    deterministic = {
        "nimber",
        "integer",
        "balanced_tree",
        "up",
        "down",
        "star",
        "switch",
    }
    for row in result.rows:
        if row.generator in deterministic:
            assert row.log_aut_pos == 0.0, row.generator
    # At least one task per generator family
    assert {r.generator for r in result.rows} == {
        "random_default",
        "random_overlap_05",
        "coupled_pool_00",
        "coupled_pool_05",
        "coupled_pool_10",
    } | deterministic
