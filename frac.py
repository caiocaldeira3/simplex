from __future__ import annotations

import math

from typing import Union

class Fraction:
    numerator: int
    denominator: int

    def __init__ (self, numerator: int, denominator: int = 1) -> None:
        if isinstance(numerator, float):
            self.numerator, self.denominator = numerator.as_integer_ratio()

        elif isinstance(numerator, int):
            self.numerator = numerator
            self.denominator = 1

        else:
            raise TypeError

        if isinstance(denominator, float):
            denom_fraction = Fraction(*denominator.as_integer_ratio())

            self.numerator *= denom_fraction.denominator
            self.denominator *= denom_fraction.numerator

        elif isinstance(denominator, int):
            self.denominator *= denominator

        else:
            raise TypeError

        self.__simplify__()

    def __simplify__ (self) -> None:
        gcd = math.gcd(abs(self.numerator), abs(self.denominator))
        self.numerator //= gcd
        self.denominator //= gcd

    def __add__ (self, y: Union[Fraction, int]) -> Fraction:
        if not isinstance(y, Fraction):
            y = Fraction(y, 1)

        lcm = math.lcm(self.denominator, y.denominator)
        return Fraction(
            self.numerator * lcm // self.denominator + y.numerator * lcm // y.denominator,
            lcm
        )

    def __sub__ (self, y: Union[Fraction, float]) -> Fraction:
        if not isinstance(y, Fraction):
            y = Fraction(y, 1)

        lcm = math.lcm(self.denominator, y.denominator)
        return Fraction(
            self.numerator * lcm // self.denominator - y.numerator * lcm // y.denominator,
            lcm
        )

    def __truediv__ (self, y: Union[Fraction, float]) -> Fraction:
        if not isinstance(y, Fraction):
            y = Fraction(y, 1)

        return Fraction(
            self.numerator * y.denominator,
            self.denominator * y.numerator
        )

    def __rtruediv__ (self, y: Union[Fraction, float]) -> Fraction:
        if not isinstance(y, Fraction):
            y = Fraction(y, 1)

        return Fraction(
            self.denominator * y.numerator,
            self.numerator * y.denominator
        )

    def __mul__ (self, y: Union[Fraction, float]) -> Fraction:
        if not isinstance(y, Fraction):
            y = Fraction(y, 1)

        return Fraction(
            self.numerator * y.numerator,
            self.denominator * y.denominator
        )

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__

    def to_float (self) -> float:
        return self.numerator / self.denominator

    def __repr__(self) -> str:
        return f"{self.numerator}/{self.denominator}"