import numpy as np

from enum import Enum
from typing import Union

from frac import Fraction

Rational = Union[float, Fraction]

class RestrictionType(Enum):
    LEQ = -1
    EQ = 0
    BEQ = 1

    def __repr__ (self) -> str:
        if self is RestrictionType.LEQ:
            return "<="

        elif self is RestrictionType.EQ:
            return "="

        elif self is RestrictionType.BEQ:
            return ">="

    def __str__ (self) -> str:
        if self.value == -1:
            return "<="

        elif self.value == 0:
            return " ="

        elif self.value == 1:
            return ">="


class ProblemClass(Enum):
    MIN = -1
    MAX = 1

    def __repr__ (self) -> str:
        if self is ProblemClass.MIN:
            return "min"

        elif self is ProblemClass.MAX:
            return "max"

    def __str__ (self) -> str:
        if self.value == -1:
            return "min"

        elif self.value == 1:
            return "max"

def padronize (x: Rational, flag: bool) -> Rational:
    return fractionize(x) if flag else floatize(x)

def fractionize (x: Rational) -> Rational:
    if isinstance(x, Fraction):
        return x

    return Fraction(x)

def floatize (x: Rational) -> Rational:
    if isinstance(x, Fraction):
        return x.to_float()

    return float(x)

def eq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        pass

    return np.isclose(x, y)

def leq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        pass

    else:
        return x < y or eq(x, y)

def beq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        pass

    else:
        return x > y or eq(x, y)

def bt (x: Rational, y: Rational) -> bool:
    return not leq(x, y)

def lt (x: Rational, y: Rational) -> bool:
    return not beq(x, y)

def str_ratio (x: Rational) -> str:
    if isinstance(x, Fraction):
        return str(x)

    else:
        return f"{x:>+7.3f}"

class InvalidPL(Exception):
    pass