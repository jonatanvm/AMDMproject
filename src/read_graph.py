from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix


def read_graph_sparse(file_path, return_D=False):
    """
    Read a graph and form the laplacian in sparse matrix format.
    :param file_path: path of graph.
    :param return_D: boolean indicating whether to return the degree matrix D or not.
    :return: sparse Laplacian matrix
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix
    # https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.eigs.html#scipy.sparse.linalg.eigs
    with open(file_path, 'r') as graph:
        line = graph.readline().split(" ")
        k = None
        try:
            _, name, nVertices, nEdges = line
        except ValueError:
            _, name, nVertices, nEdges, k = line
        header = ['#', str(name), str(nVertices), str(nEdges), str(k)]
        print(header)
        row = np.zeros(int(nEdges) * 2 + int(nVertices))  # slower than empty
        col = np.zeros(int(nEdges) * 2 + int(nVertices))
        data = np.zeros(int(nEdges) * 2 + int(nVertices))
        if return_D:
            row_d = np.zeros(int(nVertices))  # slower than empty
            col_d = np.zeros(int(nVertices))
        degrees = np.zeros(int(nVertices))
        ind = 0
        d = defaultdict(list)
        while True:
            line = graph.readline()
            if not line:
                break
            v0, v1 = line.split(" ")
            v0, v1 = int(v0), int(v1)

            if v0 < v1:
                smaller, bigger = v0, v1
            else:
                smaller, bigger = v1, v0

            values = d.get(smaller)
            if values is None or bigger not in values:
                degrees[smaller] += 1
                degrees[bigger] += 1
            d[smaller].append(bigger)

            row[ind] = v0
            col[ind] = v1
            data[ind] = -1
            ind += 1

            row[ind] = v1
            col[ind] = v0
            data[ind] = -1
            ind += 1

        d = None

        for i in range(int(nVertices)):
            if degrees[i] > 0:
                row[ind + i] = i
                col[ind + i] = i
                data[ind + i] = degrees[i]
                if return_D:
                    row_d[i] = i
                    col_d[i] = i
        L = coo_matrix((data, (row, col)), shape=(int(nVertices), int(nVertices)))
        if return_D:
            D = coo_matrix((degrees, (row_d, col_d)), shape=(int(nVertices), int(nVertices)))
            return L, D, int(k), header
        # assert np.sum(matrix) == n_lines  # Check all lines read
        return L, _, int(k), header


def read_graph(file_name):
    """
    Read graph and return its adjacency matrix and degree matrix as well as the number of clusters k and
    header specified in the graph file.
    :param file_name: path of graph.
    :return: adjacency matrix, degree matrix, number of clusters, file header.
    """
    with open(file_name, 'r') as graph:
        line = graph.readline().split(" ")
        k = None
        try:
            _, name, nVertices, nEdges = line
        except ValueError:
            _, name, nVertices, nEdges, k = line
        header = ['#', str(name), str(nVertices), str(nEdges), str(k[-2])]
        matrix = np.zeros([int(nVertices), int(nVertices)])  # slower than empty
        degrees = np.zeros(int(nVertices))
        d = defaultdict(list)
        n_lines = 0
        while True:
            line = graph.readline()
            if not line:
                break
            v0, v1 = line.split(" ")
            v0, v1 = int(v0), int(v1)

            if v0 < v1:
                smaller, bigger = v0, v1
            else:
                smaller, bigger = v1, v0

            values = d.get(smaller)
            if values is None or bigger not in values:
                degrees[smaller] += 1
                degrees[bigger] += 1
            d[smaller].append(bigger)

            matrix[v0][v1] = 1
            matrix[v1][v0] = 1
            n_lines += 2

        d = None
        # assert np.sum(matrix) == n_lines  # Check all lines read
        return matrix, np.diag(degrees), int(k), header
