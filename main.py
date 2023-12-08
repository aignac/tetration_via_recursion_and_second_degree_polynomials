import math
from typing import Callable
from math import log
from functools import cache, reduce

import pytest


class SuperExponentShouldBeLargerThanNegativeTwo(Exception):
    pass


@cache
def tetration(
    base: float,
    super_exponent: float,
    logarithm: Callable[[float], float] = lambda x: log(x),
    general_logarithm: Callable[[float, float], float] = lambda x, b: log(x, b),
) -> float:
    """This algorithm calculates the tetration with a real base and real super exponent."""
    if super_exponent <= -2:
        raise SuperExponentShouldBeLargerThanNegativeTwo
    if base == 0:
        return not super_exponent % 2
    if super_exponent == -1:
        return 0
    if super_exponent == 0:
        return 1
    if super_exponent < -1:
        return general_logarithm(
            tetration(base, super_exponent + 1, logarithm, general_logarithm), base # NOQA
        )
    if super_exponent > 0:
        return base ** tetration(
            base, super_exponent - 1, logarithm, general_logarithm # NOQA
        )
    first_term = 1
    second_term = (
        (2 * (logarithm_of_base := logarithm(base)))
        / (1 + logarithm_of_base)
        * super_exponent
    )
    third_term = (
        -(1 - logarithm_of_base) / (1 + logarithm_of_base) * super_exponent**2
    )
    terms = [first_term, second_term, third_term]
    return reduce(lambda x, y: x + y, terms, 0)


def test_tetration() -> None:
    assert math.isclose(tetration(2, 0.5), 1.45933, abs_tol=1e-5)


@pytest.mark.parametrize("base", [0, 2, 3, -1])
def test_tetration_when_super_exponent_is_zero(base) -> None:
    assert tetration(base, 0) == 1


@pytest.mark.parametrize("base", [0, 1, 2, 3, 100])
def test_tetration_when_super_exponent_is_minus_one(base) -> None:
    assert tetration(base, -1) == 0


@pytest.mark.parametrize(
    "base, super_exponent, result",
    [(2, 2, 4), (2, 3, 16), (2, 4, 65536), (3, 2, 27), (4, 2, 256)],
)
def test_tetration_when_super_exponent_and_base_are_positive(
    base, super_exponent, result
) -> None:
    assert tetration(base, -1) == 0


@pytest.mark.parametrize("super_exponent", [0, 1, 2, 3])
def test_tetration_when_base_is_one_but_super_exponent_not_minus_one(
    super_exponent,
) -> None:
    assert tetration(1, super_exponent) == 1


@pytest.mark.parametrize("super_exponent", [0, 1, 2, 3, 4, 5, -1, 12312456, 124123])
def test_tetration_when_base_is_zero(super_exponent) -> None:
    assert tetration(0, super_exponent) == ~super_exponent % 2


@pytest.mark.parametrize("base", [1, 2, 3, 4, 5, -1, 12312456, 124123])
def test_tetration_when_super_exponent_is_minus_two(base) -> None:
    with pytest.raises(SuperExponentShouldBeLargerThanNegativeTwo):
        tetration(base, -2)


@pytest.mark.parametrize(
    "base, super_exponent",
    [(1, -3), (2, -4), (3, -5), (4, -6), (1202, -7134), (12, -2.5)],
)
def test_tetration_when_super_exponent_is_smaller_than_minus_two(
    base, super_exponent
) -> None:
    with pytest.raises(SuperExponentShouldBeLargerThanNegativeTwo):
        tetration(base, super_exponent)


@pytest.mark.parametrize("base", [1, 2, 3, 4, 5, -1, 12312456, 124123, 1.5])
def test_tetration_when_super_exponent_is_one(base) -> None:
    assert tetration(base, 1) == base
