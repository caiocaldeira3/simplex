import numpy as np
from util import WrongAnswer

from simplex import PL

if __name__ == "__main__":
    num_res, num_vars = map(int, input().split(" "))

    restrictions = np.empty((num_res, num_vars + 1))
    c_vec =  np.fromiter(map(float, input().split(" ")), dtype=np.float32)

    for row in range(num_res):
        row_str = input().split(" ")
        for column in range(num_vars + 1):
            restrictions[row, column] = float(row_str[column])

    try:
        pl = PL(c_vec, restrictions, fraction=False)
        res = pl.solve()

        res.print(pl)

    except WrongAnswer:
        plf = PL(c_vec, restrictions, fraction=True)
        res = plf.solve()

        res.print(pl)