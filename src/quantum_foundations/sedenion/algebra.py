"""Octonion / sedenion Cayley–Dickson operations in the paper's ℤ-basis."""

from __future__ import annotations

from typing import TypeAlias

Oct: TypeAlias = tuple[int, ...]
Sen: TypeAlias = tuple[Oct, Oct]


def _build_oct_tab() -> list[list[tuple[int, int]]]:
    """tab[i][j] = (sign, k) with e_i e_j = sign * e_k in ℤ-basis."""
    tab: list[list[tuple[int, int]]] = [[(0, 0)] * 8 for _ in range(8)]
    for k in range(8):
        tab[0][k] = (1, k)
        tab[k][0] = (1, k)
    tab[1] = [(1, 1), (-1, 0), (1, 3), (-1, 2), (1, 5), (-1, 4), (-1, 7), (1, 6)]
    tab[2] = [(1, 2), (-1, 3), (-1, 0), (1, 1), (1, 6), (1, 7), (-1, 4), (-1, 5)]
    tab[3] = [(1, 3), (1, 2), (-1, 1), (-1, 0), (1, 7), (-1, 6), (1, 5), (-1, 4)]
    tab[4] = [(1, 4), (-1, 5), (-1, 6), (-1, 7), (-1, 0), (1, 1), (1, 2), (1, 3)]
    tab[5] = [(1, 5), (1, 4), (-1, 7), (1, 6), (-1, 1), (-1, 0), (-1, 3), (1, 2)]
    tab[6] = [(1, 6), (1, 7), (1, 4), (-1, 5), (-1, 2), (1, 3), (-1, 0), (-1, 1)]
    tab[7] = [(1, 7), (-1, 6), (1, 5), (1, 4), (-1, 3), (-1, 2), (1, 1), (-1, 0)]
    return tab


OCT_TAB = _build_oct_tab()


def o_conj(o: Oct) -> Oct:
    return (o[0],) + tuple(-x for x in o[1:])


def o_add(a: Oct, b: Oct) -> Oct:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def o_sub(a: Oct, b: Oct) -> Oct:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def o_mul(a: Oct, b: Oct) -> Oct:
    acc = [0] * 8
    for i in range(8):
        if a[i] == 0:
            continue
        for j in range(8):
            if b[j] == 0:
                continue
            sgn, k = OCT_TAB[i][j]
            acc[k] += a[i] * b[j] * sgn
    return tuple(acc)


def o_norm_sq(o: Oct) -> int:
    return sum(x * x for x in o)


def s_mul(x: Sen, y: Sen) -> Sen:
    a, b = x
    c, d = y
    left = o_sub(o_mul(a, c), o_mul(o_conj(d), b))
    right = o_add(o_mul(d, a), o_mul(b, o_conj(c)))
    return (left, right)


def s_sub(x: Sen, y: Sen) -> Sen:
    return (o_sub(x[0], y[0]), o_sub(x[1], y[1]))


def s_assoc(x: Sen, y: Sen, z: Sen) -> Sen:
    return s_sub(s_mul(s_mul(x, y), z), s_mul(x, s_mul(y, z)))


def s_norm_sq(x: Sen) -> int:
    return o_norm_sq(x[0]) + o_norm_sq(x[1])


def basis_o(k: int) -> Oct:
    t = [0] * 8
    t[k] = 1
    return tuple(t)


def basis_s(k: int) -> Sen:
    if k < 8:
        return (basis_o(k), tuple([0] * 8))
    return (tuple([0] * 8), basis_o(k - 8))
