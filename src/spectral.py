from time import time

import numpy as np
from scipy.sparse.linalg import eigsh

from kmeans import k_means
from laplacian import unnormalized_laplacian
from read_graph import read_graph

ALGORITHM_1 = 1
ALGORITHM_2 = 2
ALGORITHM_3 = 3
ALGORITHM_4 = 4


def spectral_clustering1(graph_src, k_user=None):
    """
    Run the spectral clustering algorithm 1 using the k-means  algorithm.

    :param graph_src: path to graph.
    :param k_user: user specified number of clusters (reads number of clusters from graph as default).
    :return: cluster labels and seed which resulted in the clustering
    """
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph(graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
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
    e_values, e_vectors = eigsh(L, k=k, which='SA')
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    seed = np.random.randint(1000000)
    centroids, clusters = k_means(U, k, random_seed=seed)
    cluster_sizes = [0] * k
    for i in clusters:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    return clusters, seed, header


def spectral_clustering2(graph_src, k_user=None):
    """
    Run the spectral clustering algorithm 2 using the k-means  algorithm.

    :param graph_src: path to graph.
    :param k_user: user specified number of clusters (reads number of clusters from graph as default).
    :return: cluster labels and seed which resulted in the clustering
    """
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph(graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
    print(D)
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
    e_values, e_vectors = eigsh(A=L, k=k, M=D, which='SA')

    D = None  # Free memory
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    seed = np.random.randint(1000000)
    centroids, clusters = k_means(U, k, random_seed=seed)
    cluster_sizes = [0] * k
    for i in clusters:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    return clusters, seed, header

