import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.cluster import KMeans
from scipy import linalg


def read_graph(file_name):
    with open('../graphs/' + file_name, 'r') as graph:
        _, name, nVertices, nEdges = graph.readline().split(" ")
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
            n_lines += 1

        assert np.sum(matrix) == n_lines  # Check all lines read
        return matrix, np.diag(degrees)


def unnormalized_laplacian(D, A):
    return D - A


files = ['ca-AstroPh.txt', 'ca-CondMat.txt', 'ca-GrQc.txt', 'ca-HepPh.txt', 'ca-HepTh.txt']

A, D = read_graph(files[3])
# Calculate laplacian matrix
laplacian_matrix = unnormalized_laplacian(D, A)
print(laplacian_matrix)
D = None  # Free memory
A = None  # Free memory

# Eigen-decomposition of Laplacian matrix
e_values, e_vectors = eig(laplacian_matrix)
laplacian_matrix = None  # Free memory
X = np.real(e_vectors)

# TODO: compute the first k eigenvectors u1, . . . , uk of L