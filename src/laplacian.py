import numpy as np


def unnormalized_laplacian(D, A):
    return D - A


def normalized_laplacian(D, A):
    inv_sqrt = np.divide(1, -np.sqrt(D), where=D != 0)
    return np.eye(D.shape[0]) - np.dot(np.dot(inv_sqrt, A), inv_sqrt)
