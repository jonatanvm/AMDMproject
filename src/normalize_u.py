import numpy as np

def normalize_u(U):
    T = U
    for j in range(U.shape[1]):
        T[:, j] = U[:, j] / np.sum(U[:, j])
    return T
