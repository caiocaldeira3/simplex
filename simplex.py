from __future__ import annotations
import numpy as np

from util import (
    Rational, ProblemClass, RestrictionType,
    add, sub, div, mult, str_ratio
)

class PL:
    num_vars: int
    num_res: int

    problem_class: ProblemClass

    c_vec: np.ndarray[Rational]
    b_vec: np.ndarray[Rational]
    a_matrix: np.ndarray[np.ndarray[Rational]]

    restrictions_type: np.ndarray[RestrictionType]

    def __init__ (
        self, problem_class: ProblemClass, c_vec: np.ndarray[Rational],
        restrictions: np.ndarray[np.ndarray[Rational]],
        restrictions_type: np.ndarray[RestrictionType]
    ) -> None:
        self.problem_class = ProblemClass(problem_class)
        self.restrictions_type = np.array(
            [ RestrictionType(res_id) for res_id in restrictions_type ]
        )

        self.num_vars = len(c_vec)
        self.num_res = restrictions.shape[0]

        self.a_matrix = restrictions[ : , : -1 ]
        self.b_vec = restrictions[ : , -1 ]
        self.c_vec = c_vec

    def solve (self) -> Rational:
        return FPI(self).solve()

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
        self.problem_class = ProblemClass.MAX
        self.original_class = pl.problem_class
        self.restrictions_type = np.full(pl.restrictions_type.shape, RestrictionType.EQ)

        self.num_res = pl.num_res
        aux_vars = self.__aux_vars__(pl)
        self.num_vars = pl.num_vars + self.num_aux

        self.a_matrix = np.empty((self.num_res, self.num_vars))
        self.a_matrix[ : , : -self.num_aux ] = pl.a_matrix
        self.a_matrix[ : , -self.num_aux : ] = aux_vars

        self.b_vec = pl.b_vec
        self.__init_c__(pl)

        self.__init_tableau__()

    def __init_tableau__ (self) -> None:
        self.tableau = np.zeros((self.num_res + 1, self.num_vars + 1))
        self.tableau[ 0 , : -1 ] = -self.c_vec
        self.tableau[ 1 : , : self.num_vars ] = self.a_matrix
        self.tableau[ 1 : , -1 ] = self.b_vec.T

    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars)
        self.c_vec[ : -self.num_aux] = (
            pl.c_vec if self.original_class is ProblemClass.MAX else -pl.c_vec
        )

    def __aux_vars__ (self, pl: PL) -> np.ndarray[np.ndarray[int]]:
        inequalities = []
        for idx in range(self.num_res):
            if pl.restrictions_type[idx] is not RestrictionType.EQ:
                inequalities.append((idx, len(inequalities), pl.restrictions_type[idx]))

        aux_vars = np.zeros((self.num_res, len(inequalities)))
        self.num_aux = len(inequalities)
        for row, column, res_type in inequalities:
            aux_vars[row, column] = 1 if res_type is RestrictionType.LEQ else -1

        return aux_vars

    def solve (self) -> Rational:
        pass

    def __get_t__ (self, row: int, column: int, flag: bool = False) -> Rational:
        return div(self.tableau[row, -1], self.tableau[row, column], flag)

    def stagger_column (self, column: int) -> None:
        row = min(range(1, self.num_res + 1), key=lambda r: self.__get_t__(r, column))

    def str_simplex (self, row_t: int = -1, column_t: int = -1, flag: bool = False) -> str:
        if row_t != -1 and column_t != -1:
            min_t = self.__get_t__(row_t, column_t, flag)

        elif row_t != -1 or column_t != -1:
            raise Exception

        st = "| " + " ".join([
            f"{str_ratio(self.tableau[ 0, column ])}" + ("*" if column == column_t else "")
            for column in range(self.tableau.shape[1] - 1)
        ]) + f" | {self.tableau[ 0, -1 ]:>+7.3f} |\n"

        return st + "\n".join([
            f"| {' '.join([ f'{res:>+7.3f}' for res in self.tableau[ row, : -1 ] ] )} | " +
            f" {self.tableau[ row, -1 ]:>+7.3f} |" +
            ((f"{min_t:>7.3f}" if not flag else f"   {min_t}") if row_t == row else "")
            for row in range(1, self.num_res + 1)
        ]) + "\n"

class AuxPL(FPI):
    def __init_c__ (self, pl: PL) -> None:
        self.c_vec = np.zeros(self.num_vars)
        self.c_vec[ -self.num_aux : ] = -1

    def __aux_vars__ (self, pl: PL) -> tuple(np.ndarray[np.ndarray[int]], int):
        self.num_aux = pl.num_res
        return np.eye(self.num_aux)

    def solve (self, flag: bool = False) -> None:
        aux_matrix = self.tableau.copy()

        self.debug()
        for aux_idx in range(1, self.num_aux + 1):
            self.tableau[0] -= self.tableau[aux_idx]
            self.debug()

        while True:
            for column in range(self.tableau.shape[1] - 1):
                if self.tableau[ 0, column ] < 0:
                    self.stagger_column(column)
                    break


    def debug (self) -> None:
        print(self.str_simplex())
        print("-" * 100)