from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from util import (
    PLType,
    Rational,
    Result,
    leq, beq, lt, bt, eq, padronize, str_ratio, clean_tableau,
    ProblemClass, RestrictionType
)

class PL:
    num_vars: int
    num_res: int

    problem_class: ProblemClass

    c_vec: np.ndarray[Rational]
    b_vec: np.ndarray[Rational]
    a_matrix: np.ndarray[np.ndarray[Rational]]

    restrictions_type: np.ndarray[RestrictionType]
    fraction: bool

    def __init__ (
        self, problem_class: ProblemClass, c_vec: np.ndarray[Rational],
        restrictions: np.ndarray[np.ndarray[Rational]],
        restrictions_type: np.ndarray[RestrictionType], fraction: bool = False
    ) -> None:
        self.fraction = fraction
        self.problem_class = ProblemClass(problem_class)
        self.restrictions_type = np.array(
            [ RestrictionType(res_id) for res_id in restrictions_type ]
        )

        self.num_vars = len(c_vec)
        self.num_res = restrictions.shape[0]

        self.a_matrix = restrictions[ : , : -1 ]
        self.b_vec = restrictions[ : , -1 ]
        self.c_vec = c_vec

        self.__padronize__()

    def __padronize__ (self):
        self.a_matrix = np.array([
            np.vectorize(padronize)(row, self.fraction) for row in self.a_matrix
        ])
        self.b_vec = np.vectorize(padronize)(self.b_vec, self.fraction)
        self.c_vec = np.vectorize(padronize)(self.c_vec, self.fraction)

    def solve (self) -> Rational:
        return FPI(self).solve()

    def compute (self, x: np.ndarray) -> Rational:
        return self.c_vec @ padronize(x)

    def __format_cvec__ (self) -> range:
        for idx in range(self.num_vars):
            yield f"{str_ratio(self.c_vec[idx])}*x{idx}"

    def __repr__ (self) -> str:
        c_str = " ".join(self.__format_cvec__())
        matrix_str = [
            f"| {' '.join([ str_ratio(res) for res in row ] )} | x {res_type} {str_ratio(b)}"
            for row, res_type, b in zip(self.a_matrix, self.restrictions_type, self.b_vec)
        ]

        return f"{str(self.problem_class)} {c_str}\n" + "\n".join(matrix_str)

class FPI(PL):
    num_aux: int
    original_class: ProblemClass
    tableau: np.ndarray[np.ndarray[Rational]]

    def __init__ (self, pl: PL) -> None:
        self.fraction = pl.fraction
        self.problem_class = ProblemClass.MAX
        self.original_class = pl.problem_class
        self.restrictions_type = np.full(pl.restrictions_type.shape, RestrictionType.EQ)

        self.num_res = pl.num_res
        aux_vars = self.__aux_vars__(pl)
        self.num_vars = pl.num_vars + self.num_aux

        self.a_matrix = np.empty((self.num_res, self.num_vars), dtype=pl.a_matrix.dtype)
        self.a_matrix[ : , : -self.num_aux ] = pl.a_matrix
        self.a_matrix[ : , -self.num_aux : ] = aux_vars

        self.b_vec = pl.b_vec
        self.__init_c__(pl)
        self.__padronize__()

        self.__init_tableau__()

    @property
    def matrix_slice (self) -> Tuple[slice, slice]:
        return (slice(1, self.num_res + 1), slice(self.num_res, self.num_res + self.num_vars))

    @property
    def aux_matrix_slice (self) -> Tuple[slice, slice]:
        return (slice(1, self.num_res + 1), slice(self.num_res))

    @property
    def b_slice (self) -> Tuple[slice, slice]:
        return (slice(1, self.num_res + 1), -1)

    @property
    def c_slice (self) -> Tuple[slice, slice]:
        return (0, slice(self.num_res, self.num_res + self.num_vars))

    @property
    def certificate_slice (self) -> Tuple[slice, slice]:
        return (0, slice(self.num_res))

    @property
    def fpi_slice (self) -> None:
        return (slice(1, self.num_res + 1), slice(0, self.num_res + self.num_vars))

    def __init_tableau__ (self) -> None:
        self.tableau = np.zeros(
            (self.num_res + 1, self.num_res + self.num_vars + 1), dtype=self.a_matrix.dtype
        )
        self.tableau[ self.aux_matrix_slice ] = np.eye(self.num_res)
        self.tableau[ self.c_slice ] = -self.c_vec
        self.tableau[ self.matrix_slice ] = self.a_matrix
        self.tableau[ self.b_slice ] = self.b_vec.T

        self.tableau[ 0, -1 ] = padronize(self.tableau[ 0, -1 ], self.fraction)

    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars, dtype=pl.c_vec.dtype)
        self.c_vec[ : -self.num_aux ] = (
            pl.c_vec if self.original_class is ProblemClass.MAX else -pl.c_vec
        )

    def __aux_vars__ (self, pl: PL) -> np.ndarray[np.ndarray[int]]:
        inequalities = []
        for idx in range(self.num_res):
            if pl.restrictions_type[idx] is not RestrictionType.EQ:
                inequalities.append((idx, len(inequalities), pl.restrictions_type[idx]))

        aux_vars = np.zeros((self.num_res, len(inequalities)), dtype=pl.a_matrix.dtype)
        self.num_aux = len(inequalities)
        for row, column, res_type in inequalities:
            aux_vars[row, column] = 1 if res_type is RestrictionType.LEQ else -1

        return aux_vars

    def __get_t__ (self, row: int, column: int) -> Rational:
        if leq(self.tableau[row, column], 0):
            return padronize(np.inf, self.fraction)

        return self.tableau[row, -1] / self.tableau[row, column]

    def __form_solution__ (self) -> np.ndarray:
        solution = np.zeros(self.num_vars, self.tableau.dtype)
        for column in range(self.num_res, self.num_res + self.num_vars):
            one = np.where(eq(self.tableau[ : , column ], 1))[0]
            zeros = np.where(eq(self.tableau[ : , column ], 0))[0]

            if len(one) == 1 and len(zeros) == self.num_res:
                row = one[0]
                solution[column - self.num_res] = self.tableau[row, -1]

        return np.vectorize(padronize)(solution, self.fraction)

    def stagger_column (self, column: int) -> Optional[Result]:
        row_t = min(range(1, self.num_res + 1), key=lambda r: self.__get_t__(r, column))
        if lt(self.tableau[row_t, column], 0):
            return Result(PLType.ILIMITED, self.tableau[ 0, : self.num_res])

        self.debug(row_t, column)

        self.tableau[row_t] /= self.tableau[row_t, column]
        for row in range(self.num_res + 1):
            if row == row_t:
                continue

            ratio = self.tableau[row, column] / self.tableau[row_t, column]
            self.tableau[row] -= self.tableau[row_t] * ratio

    def __optmize__ (self) -> None:
        flag = True
        while flag:
            flag = False
            for column in range(self.num_res, self.tableau.shape[1] - 1):
                if lt(self.tableau[ 0, column ], 0):
                    flag = True
                    self.stagger_column(column)
                    break

    @clean_tableau
    def solve (self) -> Rational:
        aux_tableau, aux_res = AuxPL(self).solve()
        if aux_res.pl_type is PLType.INVALID:
            return aux_res

        self.tableau[self.fpi_slice] = aux_tableau[self.fpi_slice]
        self.tableau[self.b_slice] = aux_tableau[self.b_slice]

        self.debug()

        for column in range(self.num_vars):
            if bt(aux_res.opt_x[column], 0) and bt(self.tableau[ 0, column ], 0):
                self.stagger_column(column)

        self.__optmize__()

        self.debug()
        solution = self.__form_solution__()

        return Result(PLType.LIMITED, self.tableau[ self.certificate_slice ], solution)

    def str_tableau (self, row_t: int = -1, column_t: int = -1) -> str:
        if row_t != -1 and column_t != -1:
            min_t = self.__get_t__(row_t, column_t)

        elif row_t != -1 or column_t != -1:
            raise IndexError

        st = "| " + " ".join([
            str_ratio(self.tableau[ 0, column ]) + ("*" if column == column_t else "")
            for column in range(self.tableau.shape[1] - 1)
        ]) + f" | {str_ratio(self.tableau[ 0, -1 ])} | \n"

        return st + "\n".join([
            f"| {' '.join([ str_ratio(res) for res in self.tableau[ row, : -1 ] ] )} |" +
            f" {str_ratio(self.tableau[ row, -1 ])} | " + (str_ratio(min_t) if row_t == row else "")
            for row in range(1, self.num_res + 1)
        ])

    def debug (self, *args) -> None:
        print(self.str_tableau(*args))
        print("-" * 100)

class AuxPL(FPI):
    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars)
        self.c_vec[ -self.num_aux : ] = -1

    def __aux_vars__ (self, pl: PL) -> tuple(np.ndarray[np.ndarray[int]], int):
        self.num_aux = pl.num_res
        return np.eye(self.num_aux)

    def __form_basic_solution__ (self) -> np.ndarray:
        basic_solution = np.zeros(self.num_vars - self.num_aux, self.tableau.dtype)
        for column in range(self.num_res, self.num_res + self.num_vars - self.num_aux):
            one = np.where(eq(self.tableau[ : , column ], 1))[0]
            zeros = np.where(eq(self.tableau[ : , column ], 0))[0]

            if len(one) == 1 and len(zeros) == self.num_res:
                row = one[0]
                basic_solution[column - self.num_res] = self.tableau[row, -1]

        return np.vectorize(padronize)(basic_solution, self.fraction)

    @clean_tableau
    def solve (self) -> Result:
        self.debug()
        for aux_idx in range(1, self.num_aux + 1):
            self.tableau[0] -= self.tableau[aux_idx]

        self.__optmize__()

        self.debug()

        if lt(self.tableau[ 0, -1 ], 0):
            return Result(PLType.INVALID, self.tableau[ self.certificate_slice ])

        basic_solution = self.__form_basic_solution__()
        return (
            self.tableau,
            Result(PLType.LIMITED, self.tableau[ self.certificate_slice ], basic_solution)
        )