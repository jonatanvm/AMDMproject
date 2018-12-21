import numpy as np


def select_centroids(data, k, random_seed=1):
    # INPUT: N x d data array, k number of clusters.
    # OUTPUT: k x d array of k randomly assigned mean vectors with d dimensions.

    np.random.seed(seed=random_seed)

    centroids = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        centroids[:, i] = np.random.uniform(np.min(data[:, i]), np.max(data[:, i]), size=k)

    return centroids


def assign_points(data, centroids):
    # INPUT: N x d data array, k x d centroids array.
    # OUTPUT: N x 1 array of cluster assignments in {0,...,k-1}.

    clusters = np.array([np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data], dtype=np.int32)
    return clusters


def move_centroids(data, old_centroids, clusters):
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


def k_means(data, k, random_seed=1, num_iters=10, plot=True):
    # INPUT: N x d data array, k number of clusters, number of iterations, boolean plot.
    # OUTPUT: N x 1 array of cluster assignments.

    centroids = select_centroids(data, k, random_seed)

    ass = True
    i = 0
    while i < num_iters and ass:
        clusters = assign_points(data, centroids)

        centroids, assigned = move_centroids(data, centroids, clusters)
        ass = assigned
        i += 1

    return centroids, clusters

# centroids, clusters = k_means(data, 2, plot=False)
# print("The final cluster mean values are:", centroids)
