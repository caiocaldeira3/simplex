from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from util import (
    PLType, ProblemClass, RestrictionType, Rational,
    Result,
    leq, lt, eq, padronize, str_ratio, clean_tableau,
    InvalidPLType, NotBasic, WrongAnswer

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
    debug_mode: bool

    def __init__ (
        self, c_vec: np.ndarray[Rational],
        restrictions: np.ndarray[np.ndarray[Rational]], problem_class: ProblemClass = None,
        restrictions_type: np.ndarray[RestrictionType] = None, fraction: bool = False,
        debug_mode: bool = False
    ) -> None:
        self.fraction = fraction
        self.debug_mode = debug_mode
        self.problem_class = problem_class if problem_class is not None else ProblemClass.MAX
        self.num_vars = len(c_vec)
        self.num_res = restrictions.shape[0]

        if restrictions_type is None:
            self.restrictions_type = np.full((self.num_res,), RestrictionType.LEQ)

        else:
            self.restrictions_type = np.array(
                [ RestrictionType(res_id) for res_id in restrictions_type ]
            )

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

    def __fraction_verify__ (self, result: Result) -> bool:
        if result.pl_type is PLType.ILIMITED:
            if not np.all(self.a_matrix @ result.certificate <= 0):
                return False

            if not np.all(self.c_vec @ result.certificate > 0):
                return False

            return True

        elif result.pl_type is PLType.INVALID:
            if not np.all(result.certificate @ self.a_matrix >= 0):
                return False

            if not np.all(result.certificate @ self.b_vec < 0):
                return False

            return True

        elif result.pl_type is PLType.LIMITED:
            if not eq(self.compute(result.opt_x), result.certificate @ self.b_vec):
                return False

            if not np.all(result.certificate @ self.a_matrix - self.c_vec >= 0):
                return False

            return True

        raise InvalidPLType

    def __float_verify__ (self, result: Result) -> bool:
        if result.pl_type is PLType.ILIMITED:
            if not np.all(np.round(self.a_matrix @ result.certificate, 5) <= 0):
                raise WrongAnswer

            if not np.all(np.round(self.c_vec @ result.certificate, 5) > 0):
                raise WrongAnswer

            return True

        elif result.pl_type is PLType.INVALID:
            if not np.all(np.round(result.certificate @ self.a_matrix, 7) >= 0):
                raise WrongAnswer

            if not np.all(np.round(result.certificate @ self.b_vec, 7) < 0):
                raise WrongAnswer

            return True

        elif result.pl_type is PLType.LIMITED:
            if not eq(self.compute(result.opt_x), result.certificate @ self.b_vec):
                raise WrongAnswer

            if not np.all(np.round(result.certificate @ self.a_matrix - self.c_vec, 5) >= 0):
                raise WrongAnswer

            return True

        raise InvalidPLType

    def verify (self, result: Result) -> bool:
        return self.__fraction_verify__(result) if self.fraction else self.__float_verify__(result)

    def solve (self) -> Rational:
        fpi = FPI(self)
        result = fpi.solve()

        if result.pl_type is PLType.LIMITED:
            result.opt_x = result.opt_x[ : self.num_vars ]

        if result.pl_type is not PLType.ILIMITED:
            result.certificate = result.certificate[ : self.num_res ]

            for row in range(len(self.b_vec)):
                if lt(self.b_vec[row], 0):
                    result.certificate[row] *= -1
        else:

            result.certificate = result.certificate[ : self.num_vars ]
            result.opt_x = result.opt_x[ : self.num_vars ]

        self.verify(result)
        return result


    def compute (self, x: np.ndarray) -> Rational:
        return self.c_vec @ padronize(x, self.fraction)

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
        self.debug_mode = pl.debug_mode
        self.problem_class = ProblemClass.MAX
        self.original_class = pl.problem_class
        self.restrictions_type = np.full(pl.restrictions_type.shape, RestrictionType.EQ)

        self.num_res = pl.num_res
        aux_vars = self.__aux_vars__(pl)
        self.num_vars = pl.num_vars + self.num_aux

        self.a_matrix = np.empty((self.num_res, self.num_vars), dtype=pl.a_matrix.dtype)
        self.a_matrix[ : , : -self.num_aux ] = pl.a_matrix.copy()
        self.a_matrix[ : , -self.num_aux : ] = aux_vars

        self.b_vec = pl.b_vec.copy()
        self.__init_c__(pl)
        self.__init_tableau__()

        self.__padronize__()

    def __padronize__ (self):
        super()
        self.tableau = np.vectorize(padronize)(self.tableau, self.fraction)

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
        self.tableau[ self.c_slice ] = -self.c_vec.copy()
        self.tableau[ self.matrix_slice ] = self.a_matrix.copy()
        self.tableau[ self.b_slice ] = self.b_vec.T.copy()

        for row in range(1, self.num_res + 1):
            if lt(self.tableau[ row, -1 ], 0):
                self.a_matrix[ row - 1] *= -1
                self.tableau[ row, : ] *= -1
                self.b_vec[row - 1] *= -1

        self.tableau[ 0, -1 ] = padronize(self.tableau[ 0, -1 ], self.fraction)

    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars, dtype=pl.c_vec.dtype)
        self.c_vec[ : -self.num_aux ] = (
            pl.c_vec.copy() if self.original_class is ProblemClass.MAX else -pl.c_vec
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

    def __get_basic_row__ (self, column: int, start: int = 0) -> int:
        one = np.where(eq(self.tableau[ start : , column ], 1))[0]
        zeros = np.where(eq(self.tableau[ start : , column ], 0))[0]

        if len(one) == 1 and len(zeros) == self.num_res - start:
            return one[0] + start

        raise NotBasic

    def __form_solution__ (self) -> np.ndarray:
        solution = np.zeros(self.num_vars, self.tableau.dtype)
        for column in range(self.num_res, self.num_res + self.num_vars):
            try:
                row = self.__get_basic_row__(column)
                solution[column - self.num_res] = self.tableau[row, -1]
                self.tableau[row, -1] = 0

            except NotBasic:
                continue

        return np.vectorize(padronize)(solution, self.fraction)

    def __ilimited_certificate__ (self, ili_col: int) -> np.ndarray[Rational]:
        certificate = np.zeros(self.num_vars, dtype=self.tableau.dtype)
        certificate[ili_col - self.num_res] = padronize(1, self.fraction)
        for column in range(self.num_res, self.num_res + self.num_vars):
            try:
                row = self.__get_basic_row__(column, 1)
                certificate[column - self.num_res] = self.tableau[row, ili_col] * -1
                self.tableau[row, ili_col] = 0

            except Exception as exc:
                continue

        return certificate

    def get_row (self, column: int) -> int:
        min_row = 1
        for row_indx in range(2, self.num_res + 1):
            if leq(self.tableau[min_row, column], 0):
                min_row = row_indx

            elif leq(self.tableau[row_indx, column], 0):
                continue

            elif lt(self.__get_t__(row_indx, column), self.__get_t__(min_row, column)):
                min_row = row_indx

        return min_row

    def stagger_column (self, column: int) -> Optional[Result]:
        row_t = self.get_row(column)
        if leq(self.tableau[row_t, column], 0):
            return Result(
                PLType.ILIMITED,
                self.__ilimited_certificate__(column),
                self.__form_solution__()
            )

        self.debug(row_t, column)

        self.tableau[row_t] /= self.tableau[row_t, column]
        for row in range(self.num_res + 1):
            if row == row_t:
                continue

            ratio = self.tableau[row, column] / self.tableau[row_t, column]
            self.tableau[row] -= self.tableau[row_t] * ratio

    def __optmize__ (self) -> Optional[Result]:
        flag = True
        res = None
        while flag:
            flag = False
            for column in range(self.num_res, self.tableau.shape[1] - 1):
                if lt(self.tableau[ 0, column ], 0):
                    flag = True
                    res = self.stagger_column(column)
                    break

            if res is not None:
                return res

    @clean_tableau
    def solve (self) -> Result:
        self.debug()

        aux_tableau, aux_res = AuxPL(self).solve()
        if aux_res.pl_type is PLType.INVALID:
            return aux_res

        self.tableau[self.fpi_slice] = aux_tableau[self.fpi_slice]
        self.tableau[self.b_slice] = aux_tableau[self.b_slice]

        self.debug()

        for column in range(self.num_res, self.tableau.shape[1] - 1):
            try:
                self.debug(self.__get_basic_row__(column, 1), column)
                self.stagger_column(column)

            except NotBasic:
                continue

        res = self.__optmize__()
        if res is not None:
            return res

        self.debug()

        return Result(
            PLType.LIMITED, self.tableau[ self.certificate_slice ], self.__form_solution__()
        )

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
        if self.debug_mode:
            print(self.str_tableau(*args))
            print("-" * 100)

class AuxPL(FPI):
    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars)
        self.c_vec[ -self.num_aux : ] = -1

    def __aux_vars__ (self, pl: PL) -> tuple(np.ndarray[np.ndarray[int]], int):
        self.num_aux = pl.num_res
        return np.eye(self.num_aux)

    @clean_tableau
    def solve (self) -> Result:
        self.debug()
        for aux_idx in range(1, self.num_aux + 1):
            self.tableau[0] -= self.tableau[aux_idx]

        self.__optmize__()

        self.debug()

        if lt(self.tableau[ 0, -1 ], 0):
            return None, Result(PLType.INVALID, self.tableau[ self.certificate_slice ])

        return (
            self.tableau.copy(),
            Result(
                PLType.LIMITED,
                self.tableau[ self.certificate_slice ],
                self.__form_solution__()
            )
        )