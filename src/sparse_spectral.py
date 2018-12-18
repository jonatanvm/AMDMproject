from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from read_graph import read_graph_sparse


def sparse_spectral_clustering1(loc, graph_src, k_user=None):
    print("Reading sparse graph: " + graph_src)
    start = time()
    L, _, k, header = read_graph_sparse(loc, graph_src, False)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)

    print("Header: %s" % " ".join(header))
    print("Finished after %.2f seconds" % (time() - start))

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigsh(L, k=k, which='SA')
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    plt.show()
    return kmeans.labels_


def sparse_spectral_clustering2(loc, graph_src, k_user=None):
    print("Reading sparse graph: " + graph_src)
    start = time()
    L, D, k, header = read_graph_sparse(loc, graph_src, True)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)

    print("Header: %s" % " ".join(header))
    print("Finished after %.2f seconds" % (time() - start))

    # Generalized Eigen-decomposition of Laplacian matrix
    print("Calculating Eigen-decomposition")
    start = time()
    e_values, e_vectors = eigsh(L, k=k, M=D, which='SA')
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    D = None  # Free memory
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
