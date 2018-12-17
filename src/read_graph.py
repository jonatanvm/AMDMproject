import numpy as np
from scipy.sparse import coo_matrix


def read_graph_sparse(loc, file_name):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix
    # https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs
    with open(loc + file_name, 'r') as graph:
        line = graph.readline().split(" ")
        k = None
        try:
            _, name, nVertices, nEdges = line
        except ValueError:
            _, name, nVertices, nEdges, k = line
        header = ['#', str(name), str(nVertices), str(nEdges), str(k)]
        print(header)
        matrix = coo_matrix((int(nVertices), int(nVertices))).toarray()  # slower than empty
        degrees = coo_matrix((int(nVertices), int(nVertices))).toarray()
        n_lines = 0
        while True:
            line = graph.readline()
            if not line:
                break
            v0, v1 = line.split(" ")
            v0, v1 = int(v0), int(v1)
            degrees[v0] += 1
            degrees[v1] += 1
            matrix[v0][v1] = 1
            matrix[v1][v0] = 1
            n_lines += 1

        # assert np.sum(matrix) == n_lines  # Check all lines read
        return matrix, np.diag(degrees), int(k), header


def read_graph(loc, file_name):
    with open(loc + file_name, 'r') as graph:
        line = graph.readline().split(" ")
        k = None
        try:
            _, name, nVertices, nEdges = line
        except ValueError:
            _, name, nVertices, nEdges, k = line
        header = ['#', str(name), str(nVertices), str(nEdges), str(k)]
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
            matrix[v0][v1] = 1
            matrix[v1][v0] = 1
            n_lines += 2

        # assert np.sum(matrix) == n_lines  # Check all lines read
        return matrix, np.diag(degrees), int(k), header
