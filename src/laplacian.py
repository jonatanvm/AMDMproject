import numpy as np


def unnormalized_laplacian(D, A):
    """
    Calculate the unnormalized laplacian.
    :param D: degree matrix.
    :param A: adjacency matrix.
    :return: unnormalized laplacian
    """
    return D - A


def normalized_laplacian(D, A):
    """
    Calculate the normalized laplacian.
    :param D: degree matrix.
    :param A: adjacency matrix.
    :return: normalized laplacian
    """
    inv_sqrt = np.divide(1, np.sqrt(D), where=D != 0)
    return np.eye(D.shape[0]) - np.dot(np.dot(inv_sqrt, A), inv_sqrt)
