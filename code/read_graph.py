import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.cluster import KMeans
from scipy import linalg


def read_graph(file_name):
    with open('../graphs/' + file_name, 'r') as graph:
        _, name, nVertices, nEdges, k = graph.readline().split(" ")
        matrix = np.zeros([int(nVertices), int(nVertices)])  # slower than empty
        degrees = np.zeros(int(nVertices))
        n_lines = 0
        while True:
            line = graph.readline()
            if not line:
                break
            v0, v1 = line.split(" ")
            v0, v1 = int(v0), int(v1)
            degrees[v0] += 1
            degrees[v1] += 1
            matrix[v0][v1] += 1
            matrix[v1][v0] += 1
            n_lines += 2

        assert np.sum(matrix) == n_lines  # Check all lines read
        return matrix, np.diag(degrees), int(k)


def unnormalized_laplacian(D, A):
    return D - A


def normalized_laplacian(D, A):
    inv_sqrt = np.divide(1, np.sqrt(D), where=D != 0)
    return np.eye(D.shape[0]) - np.dot(np.dot(inv_sqrt, A), inv_sqrt)


files = ['ca-AstroPh.txt', 'ca-CondMat.txt', 'ca-GrQc.txt', 'ca-HepPh.txt', 'ca-HepTh.txt']

A, D, k = read_graph(files[4])
print(np.sum(D, axis=0))
# Calculate laplacian matrix
laplacian_matrix = normalized_laplacian(D, A)
print(laplacian_matrix)
D = None  # Free memory
A = None  # Free memory

# Eigen-decomposition of Laplacian matrix
e_values, e_vectors = eig(laplacian_matrix)
laplacian_matrix = None  # Free memory
X = np.real(e_vectors)


# TODO: compute the first k eigenvectors u1, . . . , uk of L
index_k = np.argsort(e_values)[k-1]
partition = [val >= 0 for val in e_vectors[:, index_k]]
