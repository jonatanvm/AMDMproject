from time import time

import numpy as np
from scipy.sparse.linalg import eigsh, lobpcg

from kmeans2 import nk_means_pp
from read_graph import read_graph_sparse


def custom_sparse_spectral_clustering1(graph_src, k_user=None, e_mode='eigsh'):
    """
    Run the spectral clustering algorithm 1 using the nk-means++ algorithm.
    :param graph_src: path to graph.
    :param k_user: user specified number of clusters (reads number of clusters from graph as default).
    :param e_mode: type of eigenvalue decomposition algorithm.
    :return: cluster labels and seed which resulted in the clustering
    """
    print("Reading sparse graph: " + graph_src)
    start = time()
    L, _, k, header = read_graph_sparse(graph_src, False)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print("Finished after %.2f seconds" % (time() - start))

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    if e_mode == 'lobpcg':
        # Random estimations of eigenvalues
        X = np.random.rand(L.shape[0], k)
        e_values, e_vectors = lobpcg(A=L, X=X, largest=False)
    elif e_mode == 'eigsh':
        e_values, e_vectors = eigsh(L, k=k, which='SA')
    else:
        raise Exception("Invalid eigen-decomposition algorithm (e_mode), either lobpcg or eigsh.")
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    print("Calculating Kmeans")
    start = time()
    clusters_lables, seed = nk_means_pp(graph_src, U, k)
    print("Finished after %.2f seconds" % (time() - start))
    print(clusters_lables)
    return clusters_lables, seed, header


def custom_sparse_spectral_clustering2(graph_src, k_user=None, e_mode='eigsh'):
    """
    Run the spectral clustering algorithm 2 using the nk-means++ algorithm.
    :param graph_src: path to graph.
    :param k_user: user specified number of clusters (reads number of clusters from graph as default).
    :param e_mode: type of eigenvalue decomposition algorithm.
    :return: cluster labels and seed which resulted in the clustering
    """
    print("Reading sparse graph: " + graph_src)
    start = time()
    L, D, k, header = read_graph_sparse(graph_src, True)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print("Finished after %.2f seconds" % (time() - start))

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()

    if e_mode == 'lobpcg':
        # Random estimations of eigenvalues
        X = np.random.rand(L.shape[0], k)
        e_values, e_vectors = lobpcg(A=L, X=X, B=D, largest=False)
    elif e_mode == 'eigsh':
        e_values, e_vectors = eigsh(L, k=k, M=D, which='SA')
    else:
        raise Exception("Invalid eigen-decomposition algorithm (e_mode), either lobpcg or eigsh.")

    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    D = None  # Free memory
    U = np.real(e_vectors)
    print("Calculating Kmeans")
    start = time()
    clusters_lables, seed = nk_means_pp(graph_src, U, k, n=10)
    print("Finished after %.2f seconds" % (time() - start))
    print(clusters_lables)
    return clusters_lables, seed, header
