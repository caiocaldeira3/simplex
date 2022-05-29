from __future__ import annotations

import numpy as np
import dataclasses as dc

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

class PLType (Enum):
    INVALID = -1
    LIMITED = 0
    ILIMITED = 1

    def __repr__ (self) -> str:
        if self is PLType.INVALID:
            return "Inválida"

        elif self is PLType.LIMITED:
            return "Limitada"

        elif self is PLType.ILIMITED:
            return "Ilimitada"

    def __str__ (self) -> str:
        if self is PLType.INVALID:
            return "Inválida"

        elif self is PLType.LIMITED:
            return "Limitada"

        elif self is PLType.ILIMITED:
            return "Ilimitada"

@dc.dataclass()
class Result:
    pl_type: PLType = dc.field(init=True)
    certificate: np.ndarray = dc.field(init=True)
    opt_x: np.ndarray = dc.field(init=True, default=None)

def clean_tableau (func: function) -> function:
    def wrapper (pl) -> Result:
        aux_matrix = pl.tableau.copy()

        res = func(pl)
        pl.tableau, aux_matrix = aux_matrix, pl.tableau

        return res

    return wrapper

def padronize (x: Rational, flag: bool) -> Rational:
    return fractionize(x) if flag else floatize(x)

def fractionize (x: Rational) -> Rational:
    if isinstance(x, Fraction):
        return x

    elif isinstance(x, np.ndarray):
        return np.array([ fractionize(num) for num in x ])

    return Fraction(x)

def floatize (x: Rational) -> Rational:
    if isinstance(x, Fraction):
        return x.to_float()

    elif isinstance(x, np.ndarray):
        return np.array([ floatize(num) for num in x ])

    return float(x)

def eq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        return x == y

    elif np.issubdtype(type(x), np.ndarray) and isinstance(x[0], Fraction):
        return x == y

    return np.isclose(x, y)

def leq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        return x <= y

    elif np.issubdtype(type(x), np.ndarray) and isinstance(x[0], Fraction):
        return x <= y

    return np.all(x < y or eq(x, y))

def beq (x: Rational, y: Rational) -> bool:
    if isinstance(x, Fraction):
        return x >= y

    elif np.issubdtype(type(x), np.ndarray) and isinstance(x[0], Fraction):
        return x >= y

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

class NotBasic (Exception):
    pass

class WrongAnswer (Exception):
    pass

class InvalidPLType (Exception):
    pass