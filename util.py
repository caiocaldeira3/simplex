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


def div (x: Rational, y: Rational, flag: bool = False) -> Rational:
    if not flag:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return (x / y).to_float()

        return x / y

    elif isinstance(x, Fraction) or isinstance(y, Fraction):
        return x / y

    else:
        return Fraction(x) / y

def add (x: Rational, y: Rational, flag: bool = False) -> Rational:
    if not flag:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return (x + y).to_float()

        return x + y

    elif isinstance(x, Fraction) or isinstance(y, Fraction):
        return x + y

    else:
        return Fraction(x + y)

def sub (x: Rational, y: Rational, flag: bool = False) -> Rational:
    if not flag:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return (x - y).to_float()

        return x - y

    elif isinstance(x, Fraction) or isinstance(y, Fraction):
        return x - y

    else:
        return Fraction(x - y)

def mult (x: Rational, y: Rational, flag: bool = False) -> Rational:
    if not flag:
        if isinstance(x, Fraction) or isinstance(y, Fraction):
            return (x * y).to_float()

        return x * y

    elif isinstance(x, Fraction) or isinstance(y, Fraction):
        return x * y

    else:
        return Fraction(x * y)

def str_ratio (x: Rational) -> str:
    if isinstance(x, Fraction):
        return str(x)

    else:
        return f"{x:>+7.3f}"