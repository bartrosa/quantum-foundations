"""Process-pool helpers, deterministic seed spawning, and cross-process diagnostics.

Worker processes must not attach logging handlers: they return :class:`TaskDiagnostic`
bundles so the main process can emit ordered log records.
"""
from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np

T = TypeVar("T")
R_co = TypeVar("R_co")


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Optional bundle of default worker count and parent seed (for documentation)."""

    n_workers: int = max(1, (os.cpu_count() or 1) - 1)
    global_seed: int = 12345


@dataclass(frozen=True, slots=True)
class TaskDiagnostic:
    """Structured diagnostic data returned from a worker task.

    Workers cannot use the configured logger handlers directly (they run in
    child processes). They accumulate ``(level, message)`` pairs here; the
    driver in the main process re-emits them through the experiment logger.
    """

    task_id: str
    status: str
    elapsed_s: float
    messages: tuple[tuple[int, str], ...]


def spawn_task_seeds(global_seed: int, n_tasks: int) -> tuple[int, ...]:
    """Return one integer seed per task, derived from a single ``SeedSequence``."""
    seed_sequence = np.random.SeedSequence(global_seed)
    return tuple(int(child.generate_state(1)[0]) for child in seed_sequence.spawn(n_tasks))


def maybe_tqdm(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str | None = None,
) -> Iterable[T]:
    """Wrap *iterable* in ``tqdm`` when the package is installed."""
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        return cast(Iterable[T], tqdm(iterable, total=total, desc=desc))
    except Exception:
        return iterable


def run_in_pool(func: Callable[[T], R_co], items: Iterable[T], *, n_workers: int) -> list[R_co]:
    """Map *func* over *items* in process pool order."""
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(func, items))


def iter_pool_unordered(
    func: Callable[[T], tuple[R_co, TaskDiagnostic]],
    items: list[T],
    *,
    n_workers: int,
) -> Iterator[tuple[R_co, TaskDiagnostic]]:
    """Apply *func* in a process pool; yield ``(result, TaskDiagnostic)`` as futures complete."""
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [pool.submit(func, item) for item in items]
        for future in maybe_tqdm(futures, total=len(futures), desc="tasks"):
            yield future.result()
