from __future__ import annotations
from decimal import DivisionByZero

import math
import numpy as np

from typing import Union

class Fraction:
    numerator: int
    denominator: int

    def __init__ (self, numerator: int, denominator: int = 1) -> None:
        if np.issubdtype(type(numerator), np.float):
            self.numerator, self.denominator = numerator.as_integer_ratio()

        elif np.issubdtype(type(numerator), np.integer):
            self.numerator = numerator
            self.denominator = 1

        else:
            print(numerator)
            raise TypeError

        if np.issubdtype(type(denominator), np.float):
            if np.isclose(denominator, 0):
                raise DivisionByZero

            elif denominator < 0:
                denominator = -denominator
                self.numerator = -self.numerator

            denom_fraction = Fraction(*denominator.as_integer_ratio())

            self.numerator *= denom_fraction.denominator
            self.denominator *= denom_fraction.numerator

        elif np.issubdtype(type(denominator), np.integer):
            if denominator == 0:
                raise DivisionByZero

            elif denominator < 0:
                denominator = -denominator
                self.numerator = -self.numerator

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

    def __neg__ (self) -> Fraction:
        return -1 * self

    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__

    def to_float (self) -> float:
        return self.numerator / self.denominator

    def __repr__(self) -> str:
        return f"{self.numerator:+}/{self.denominator}"