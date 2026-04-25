from __future__ import annotations

from collections.abc import Callable

import pytest

from quantum_foundations.conway_games.automorphism import aut_order_conway, pynauty_available_conway
from quantum_foundations.conway_games.canonical_games import (
    make_balanced_binary_tree,
    make_down,
    make_integer,
    make_negative_integer,
    make_nimber,
    make_star,
    make_switch,
    make_up,
    make_zero,
)
from quantum_foundations.conway_games.causet import ConwayCauset
from quantum_foundations.conway_games.entropy import entropy_decomposition_conway


def test_zero() -> None:
    c = make_zero()
    assert c.n == 1
    assert aut_order_conway(c, backend="networkx") == 1


def test_star() -> None:
    c = make_star()
    assert c.n == 2
    assert c.rank == (0, 1)
    assert aut_order_conway(c, backend="networkx") == 1


def test_nimber_k0() -> None:
    assert aut_order_conway(make_nimber(0), backend="networkx") == 1


def test_nimber_k1() -> None:
    c = make_nimber(1)
    assert c.n == 2
    assert aut_order_conway(c, backend="networkx") == 1


def test_nimber_k3() -> None:
    c = make_nimber(3)
    assert c.n == 4
    assert aut_order_conway(c, backend="networkx") == 1


def test_integer_5() -> None:
    c = make_integer(5)
    assert c.n == 6
    assert aut_order_conway(c, backend="networkx") == 1


def test_negative_integer_3() -> None:
    c = make_negative_integer(3)
    assert c.n == 4
    assert aut_order_conway(c, backend="networkx") == 1


def test_up() -> None:
    c = make_up()
    assert c.n == 3
    assert aut_order_conway(c, backend="networkx") == 1


def test_down() -> None:
    c = make_down()
    assert c.n == 3
    assert aut_order_conway(c, backend="networkx") == 1


def test_switch_1_0() -> None:
    c = make_switch(1, 0)
    assert c.n == 3
    assert aut_order_conway(c, backend="networkx") == 1


def test_balanced_binary_tree_depth1() -> None:
    c = make_balanced_binary_tree(1)
    assert c.n == 3
    assert aut_order_conway(c, backend="networkx") == 1


def test_balanced_binary_tree_depth2() -> None:
    c = make_balanced_binary_tree(2)
    assert c.n == 7
    assert aut_order_conway(c, backend="networkx") == 1


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: make_zero(),
        lambda: make_star(),
        lambda: make_nimber(2),
        lambda: make_nimber(3),
        lambda: make_integer(3),
        lambda: make_negative_integer(2),
        lambda: make_up(),
        lambda: make_down(),
        lambda: make_switch(1, 0),
        lambda: make_switch(2, 0),
        lambda: make_balanced_binary_tree(2),
    ],
)
def test_cross_check_backends(constructor: Callable[[], ConwayCauset]) -> None:
    if not pynauty_available_conway():
        pytest.skip("pynauty missing")
    c = constructor()
    assert aut_order_conway(c, backend="networkx") == aut_order_conway(c, backend="pynauty")


def test_entropy_decomposition_runs_on_canonical() -> None:
    constructors: list[Callable[[], ConwayCauset]] = [
        make_zero,
        make_star,
        lambda: make_nimber(3),
        lambda: make_integer(3),
        make_up,
        lambda: make_balanced_binary_tree(2),
    ]
    for constructor in constructors:
        c = constructor()
        d = entropy_decomposition_conway(c)
        assert d.log_aut_pos >= 0
        assert d.log_aut_twin_canonical >= 0
        assert d.log_aut_twin_local >= 0
