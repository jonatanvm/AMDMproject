import numpy as np

def normalize_u(U):
    T = U
    k = U.shape[1]
    print(k)
    for j in range(k):
        row = U[:,j]
        for i in range(U.shape[0]):
            sum_row = np.sum(np.power(row, 2))
            T[i, j] = np.divide(U[i, j], np.sqrt(sum_row))
    return T
