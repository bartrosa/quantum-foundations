"""Outcome classifiers for Conway causet events."""

from __future__ import annotations

from quantum_foundations.conway_games.causet import ConwayCauset


def shallow_outcome(event_index: int, causet: ConwayCauset) -> str:
    """Return local (non-recursive) outcome class in ``{'L','R','=','||'}``."""
    left = causet.past_left[event_index]
    right = causet.past_right[event_index]
    if not left and not right:
        return "="
    if left and not right:
        return "L"
    if right and not left:
        return "R"
    return "||"


def recursive_outcome(
    event_index: int,
    causet: ConwayCauset,
    *,
    cache: dict[int, str] | None = None,
    max_depth: int | None = None,
) -> str:
    """Compute recursive Conway-like outcome with memoization."""
    if max_depth is None:
        max_depth = causet.n
    if cache is None:
        cache = {}
    if max_depth <= 0:
        return shallow_outcome(event_index, causet)
    if event_index in cache:
        return cache[event_index]

    left_opts = causet.past_left[event_index]
    right_opts = causet.past_right[event_index]
    if not left_opts and not right_opts:
        cache[event_index] = "="
        return "="

    left_can_move_to_non_R = any(
        recursive_outcome(x, causet, cache=cache, max_depth=max_depth - 1) != "R" for x in left_opts
    )
    right_can_move_to_non_L = any(
        recursive_outcome(x, causet, cache=cache, max_depth=max_depth - 1) != "L"
        for x in right_opts
    )
    if left_can_move_to_non_R and not right_can_move_to_non_L:
        out = "L"
    elif right_can_move_to_non_L and not left_can_move_to_non_R:
        out = "R"
    elif not left_can_move_to_non_R and not right_can_move_to_non_L:
        out = "="
    else:
        out = "||"
    cache[event_index] = out
    return out


def outcome_histogram(causet: ConwayCauset, *, recursive: bool = False) -> dict[str, int]:
    """Count events by outcome class: keys ``L``, ``R``, ``=``, ``||``."""
    out = {"L": 0, "R": 0, "=": 0, "||": 0}
    cache: dict[int, str] | None = {} if recursive else None
    for i in range(causet.n):
        key = recursive_outcome(i, causet, cache=cache) if recursive else shallow_outcome(i, causet)
        out[key] += 1
    return out
