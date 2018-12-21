from multiprocessing import Process, Manager
from multiprocessing.pool import Pool
from time import time

import numpy as np

from calculate_comp_value import calculate_objective_function


def dist(x, y, yy):
    """
    Calculate distance in an optimized way.
    :param x: centroids
    :param y: data points
    :param yy: precomputed y-squared
    :return: Distances between data points and clusters
    """
    h = (x * x).sum(axis=1).T
    h = h[:, np.newaxis]
    j = np.dot(x, y.T)
    res = j
    res *= -2
    res += h
    res += yy
    return res


def select_centroids(data, k, random_seed=1, debug=False):
    """
    Initialize centroids using k-means++ algorithm.
    :param data: data
    :param k: amount of clusters
    :param random_seed: random seed
    :param debug: Show running time of algorithm; true or false
    :return: centroids
    """

    if debug:
        print("select_centroids")
        start = time()

    np.random.seed(seed=random_seed)

    # Select first uniformly at random
    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.randint(data.shape[0])]

    # Pre compute y squared
    data2 = (data * data).sum(axis=1).T
    data2 = data2[np.newaxis, :]

    # Select remaining centers
    for i in range(1, k):

        d = np.min(dist(centroids[:i], data, data2), axis=0)

        probs = d / d.sum()
        r = np.random.rand()
        sum = 0
        for j, p in enumerate(probs):
            sum += p
            if r < sum:
                centroids[i] = data[j]
                break
    if debug:
        print("Finished after %.2f seconds" % (time() - start))

    return np.array(centroids)


def assign_points(data, centroids, debug=False):
    """
    Assign clusters to points.
    :param data: data points.
    :param centroids: centroid locations.
    :param debug: Show running time of algorithm; true or false
    :return: cluster assignments.
    """
    if debug:
        print("assign_points")
        start = time()
    clusters = np.array([np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data], dtype=np.int32)
    if debug:
        print("Finished after %.2f seconds" % (time() - start))
    return clusters


def move_centroids(data, old_centroids, new_centroids, clusters, debug=False):
    """
    Calculate new centroid positions.
    :param data: data
    :param old_centroids: old centroids
    :param new_centroids: new centroids
    :param clusters:
    :param debug: Show running time of algorithm; true or false
    :return: new centroids.
    """
    if debug:
        print("move_centroids")
        start = time()

    for c in range(old_centroids.shape[0]):
        new_centroids[c] = np.array([np.mean(data[clusters == c], axis=0)])

    if debug:
        print("Finished after %.2f seconds" % (time() - start))

    return new_centroids


def nk_means_pp(path_to_graph, data, k, n=10, n_jobs=8, num_iters=300, tol=1e-4):
    """
    Run k-means n times in parallel.

    :param file_name: Name of file
    :param path_to_graph: Path to graph
    :param data: eigenvectors of laplacian
    :param k: number of clusters
    :param n: number of times to run kmeans.
    :param n_jobs: Number of parallel processes.
    :param num_iters: max number of iterations.
    :param tol: max error of kmeans
    :return: best clusters labels
    """
    q = Manager().list()
    pool = Pool(n_jobs)
    # Parallelize across n processes for faster multi-threaded computing
    for i in range(n):
        pool.apply_async(random_k_means_pp, (q, i, path_to_graph, data, k, num_iters, tol))

    # Prevent more tasks from being added
    pool.close()

    # Wait for all tasks to finnish
    pool.join()

    # Get best result
    best = min(q, key=lambda t: t[0])
    _, _, seed, best_clusters = best
    print("Best output: " + str(best))
    return best_clusters, seed


def random_k_means_pp(q, i, path_to_graph, data, k, num_iters=300, tol=1e-4):
    """
    Helper method for running k-means in parallel.

    :param q: Priority queue to add results to.
    :param i: Process id.
    :param file_name: Name of file
    :param path_to_graph: Path to graph
    :param data: eigenvectors of laplacian
    :param k: number of clusters
    :param num_iters: max number of iterations.
    :param tol: max error of kmeans
    :return:
    """
    np.random.seed()
    seed = np.random.randint(1000000)
    old_centroids, clusters = k_means_pp(data, k, random_seed=seed, num_iters=num_iters, tol=tol)
    value = calculate_objective_function(path_to_graph, clusters)
    print("Process %s finished with competition value %.2f with seed %s" % (i, value, seed))
    q.append((value, i, seed, clusters))


def k_means_pp(data, k, random_seed=1, num_iters=300, tol=1e-4, debug=False):
    """

    :param data: eigenvectors of laplacian
    :param k: number of clusters
    :param random_seed: Random seed
    :param num_iters: max number of iterations.
    :param tol: max error of k-means
    :param debug: true or false for printing debug texts.
    :return: centroids and clusters
    """
    if debug:
        print("Select centroids")
    old_centroids = select_centroids(data, k, random_seed, debug=debug)
    new_centroids = np.zeros(shape=old_centroids.shape)

    i = 0
    error = np.inf
    while i < num_iters and error > tol:
        clusters = assign_points(data, old_centroids, debug=debug)

        move_centroids(data, old_centroids, new_centroids, clusters, debug=debug)

        error = np.linalg.norm(new_centroids - old_centroids)
        if debug:
            print("Error after %s iterations %s" % (i, error))

        old_centroids = new_centroids.copy()
        i += 1
    if debug:
        print("Converged after %s iterations" % i)
    return old_centroids, clusters
