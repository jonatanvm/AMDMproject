import numpy as np


def select_centroids(data, k, random_seed=1):
    """
    Select inital centroid at random.
    :param data: data.
    :param k: number of clusters.
    :param random_seed: random seed to be able to reproduce results.
    :return: centroids
    """
    np.random.seed(seed=random_seed)

    centroids = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        centroids[:, i] = np.random.uniform(np.min(data[:, i]), np.max(data[:, i]), size=k)

    return centroids


def assign_points(data, centroids):
    """
    Assign points to clusters.
    :param data: data.
    :param centroids: current centroids
    :return: new clusters.
    """
    clusters = np.array([np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data], dtype=np.int32)
    return clusters


def move_centroids(data, old_centroids, clusters):
    """
    Move centroids to cluster means.
    :param data: data
    :param old_centroids: old centroids.
    :param clusters: current centroids
    :return: new centroids as well as boolean indicating whether all clusters have at least one point.
    """
    new_centroids = np.zeros(shape=old_centroids.shape)
    all_assigned = True
    for c in range(old_centroids.shape[0]):
        d = [d for i, d in enumerate(data) if clusters[i] == c]
        if d:
            new_centroids[c] = np.array([np.mean([d for i, d in enumerate(data) if clusters[i] == c], axis=0)])
        else:
            all_assigned = False
            new_centroids[c] = old_centroids[c]

    return new_centroids, all_assigned


def k_means(data, k, random_seed=1, num_iters=10):
    """
    Run k-means for num_iters iterations.
    :param data: data to be clustered.
    :param k: number of clusters.
    :param random_seed: random seed to be able to reproduce results.
    :param num_iters: number of iterations in k-means
    :return:
    """

    centroids = select_centroids(data, k, random_seed)

    ass = True
    i = 0
    while i < num_iters and ass:
        clusters = assign_points(data, centroids)

        centroids, assigned = move_centroids(data, centroids, clusters)
        ass = assigned
        i += 1

    return centroids, clusters
