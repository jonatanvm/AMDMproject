from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from laplacian import unnormalized_laplacian, normalized_laplacian, normalized_laplacian_2
from read_graph import read_graph
from normalize_u import normalize_u

ALGORITHM_1 = 1
ALGORITHM_2 = 2
ALGORITHM_3 = 3
ALGORITHM_4 = 4
ALGORITHM_5 = 5


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
    e_values, e_vectors = eigsh(L, k=k, which='SA')
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
    e_values, e_vectors = eigsh(A=L, k=k, M=D, which='SA')

    D = None  # Free memory
    print("Finished after %.2f seconds" % (time() - start))
    L = None  # Free memory
    U = np.real(e_vectors)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    print(kmeans.cluster_centers_)
    return kmeans.labels_


def spectral_clustering3(loc, graph_src, k_user=None):
    print("Reading graph: " + graph_src)
    start = time()
    A, D, k, header = read_graph(loc, graph_src)
    if k_user or k is None:
        k = k_user
        header[4] = str(k_user)
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
    e_values, e_vectors = eigsh(laplacian_matrix, k=k, which='SA')
    print("Finished after %.2f seconds" % (time() - start))
    laplacian_matrix = None  # Free memory
    U = np.real(e_vectors)
    print("Normalizing U")
    start = time()
    T = normalize_u(U)
    print("Finished after %.2f seconds" % (time() - start))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(T[:, :k])
    cluster_sizes = [0] * k
    for i in kmeans.labels_:
        cluster_sizes[i] += 1
    print("Cluster sizes: %s" % cluster_sizes)
    print(kmeans.cluster_centers_)
    return kmeans.labels_
