from enum import Enum
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from sklearn.cluster import KMeans

from src.laplacian import unnormalized_laplacian, normalized_laplacian
from src.read_graph import read_graph


class ALGORITHM(Enum):
    _1 = 1
    _2 = 2
    _3 = 3


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