import pandas as pd
import numpy as np
from numpy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from scipy import linalg

from time import time
import matplotlib.pyplot as plt


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
        return matrix, np.diag(degrees), int(k)


def unnormalized_laplacian(D, A):
    return D - A


def normalized_laplacian(D, A):
    inv_sqrt = np.divide(1, np.sqrt(D), where=D != 0)
    return np.eye(D.shape[0]) - np.dot(np.dot(inv_sqrt, A), inv_sqrt)


def sparse_spectral_clustering1(loc, graph_src, k_user=None):
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph_sparse(loc, graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print("Header: %s" % " ".join(header))
    print("Finished after %.2f seconds" % (time() - start))

    # Calculate laplacian matrix
    print("Calculating laplacian")
    start = time()
    L = unnormalized_laplacian(D, A)
    # print(L)
    print("Finished after %.2f seconds" % (time() - start))

    D = None  # Free memory
    A = None  # Free memory

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigs(L, k=k, which='SR')
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    # eig_vals = np.real(e_values)
    # inds = np.indices(e_values.shape).flatten()
    # z = list(zip(eig_vals, inds))
    # z.sort(key=lambda x: x[0])
    # v, i = list(zip(*z))
    # U = U[:, i]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    plt.show()
    return kmeans.labels_


def spectral_clustering1(loc, graph_src, k_user=None):
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph(loc, graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print("Header: %s" % " ".join(header))
    print("Finished after %.2f seconds" % (time() - start))

    # Calculate laplacian matrix
    print("Calculating laplacian")
    start = time()
    L = unnormalized_laplacian(D, A)
    # print(L)
    print("Finished after %.2f seconds" % (time() - start))

    D = None  # Free memory
    A = None  # Free memory

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigs(L, k=k, which='SR')
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    # eig_vals = np.real(e_values)
    # inds = np.indices(e_values.shape).flatten()
    # z = list(zip(eig_vals, inds))
    # z.sort(key=lambda x: x[0])
    # v, i = list(zip(*z))
    # U = U[:, i]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    plt.show()
    return kmeans.labels_


def spectral_clustering2(loc, graph_src, k_user=None):
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph(loc, graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print(A)
    print(D)
    print("Header: %s" % " ".join(header))
    print("Finished after %.2f seconds" % (time() - start))

    # Calculate laplacian matrix
    print("Calculating laplacian")
    start = time()
    L = unnormalized_laplacian(D, A)
    print("Finished after %.2f seconds" % (time() - start))

    A = None  # Free memory

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigsh(A=L, k=8)

    D = None  # Free memory
    print(e_vectors.shape)
    print(e_values.shape)
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    fig, ax = plt.subplots()
    for i in range(U.shape[0]):
        ax.scatter([U[i, 1]], [U[i, 2]], label=str(i))
    eig_vals = np.real(e_values)
    inds = np.indices(e_values.shape).flatten()
    z = list(zip(eig_vals, inds))
    z.sort(key=lambda x: x[0])
    v, i = list(zip(*z))
    print(v)
    print(i)
    U = U[:, i]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    print(kmeans.cluster_centers_)
    c = kmeans.cluster_centers_
    ax.scatter(c[:, 0], c[:, 1], label='center')
    plt.legend(loc='best')
    # plt.show()
    return kmeans.labels_


def spectral_clustering3(graph_src, k):
    print("Reading graph")
    start = time()
    A, D = read_graph(graph_src)
    print("Finished after %.2f seconds" % (time() - start))

    # Calculate laplacian matrix
    print("Calculating laplacian")
    start = time()
    laplacian_matrix = normalized_laplacian(D, A)
    print("Finished after %.2f seconds" % (time() - start))

    D = None  # Free memory
    A = None  # Free memory

    # Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigsh(laplacian_matrix, k=k)
    print("Finished after %.2f seconds" % (time() - start))
    laplacian_matrix = None  # Free memory
    X = np.real(e_vectors)


def output(name, values):
    df = pd.DataFrame(values)
    df.to_csv('../output/' + name + '.output', index=True, header=False, sep=" ")


test_files = ['ca-AstroPh.txt', 'ca-CondMat.txt', 'ca-HepPh.txt', 'ca-HepTh.txt']
comp_files = ['Oregon-1.txt', 'roadNet-CA.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt']
ptest_files = ['test1.txt', 'test2.txt']


def run_all(loc, files, out=True):
    for file in files:
        # cluster_labels = spectral_clustering1(loc, file)
        cluster_labels = sparse_spectral_clustering1(loc, file)

        if out:
            output(file.split(".")[0], cluster_labels)


# run_all('../graphs/', test_files)
run_all('../graph_tests/', ptest_files)
# run_all('../graphs_competition/', comp_files)
