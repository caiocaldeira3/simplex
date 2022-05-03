import numpy as np

from simplex import PL

if __name__ == "__main__":
    num_res, num_vars = input("Número de váriaveis e número de restrições:\n").split(" ")
    num_vars = int(num_vars)
    num_res = int(num_res)

    c_vec = np.empty(num_vars)
    restrictions = np.empty((num_res, num_vars + 1))

    c_str = input("C_vec\n").split(" ")
    for indx in range(num_vars):
        c_vec[indx] = float(c_str[indx])

    for row in range(num_res):
        row_str = input(f"{row}-th row: ").split(" ")
        for column in range(num_vars + 1):
            restrictions[row, column] = float(row_str[column])

    pl = PL(c_vec, restrictions, fraction=False)
    print(pl.solve())

