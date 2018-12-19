from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np

from calculate_comp_value import calculate_value
from main import output

cmpd = ['orangered', 'dodgerblue', 'springgreen']
cmpcent = ['red', 'darkblue', 'limegreen']


def plotting(data, centroids=None, clusters=None):
    # this function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}
    # Output: a scatter plot of the data in the clusters with cluster means
    plt.figure(figsize=(5.75, 5.25))
    plt.style.use('ggplot')
    plt.title("Data")

    alp = 0.5  # data alpha
    dt_sz = 20  # data point size
    cent_sz = 130  # centroid sz

    if centroids is None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp)
    if centroids is not None and clusters is None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=cent_sz)
    if centroids is not None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:, 0], data[:, 1], s=dt_sz, alpha=alp)

    plt.show()


def dist(x, y, yy):
    h = (x * x).sum(axis=1).T
    h = h[:, np.newaxis]
    j = np.dot(x, y.T)
    res = j
    res *= -2
    res += h
    res += yy
    return res


def select_centroids(data, k, random_seed=1):
    # INPUT: N x d data array, k number of clusters.
    # OUTPUT: k x d array of k randomly assigned mean vectors with d dimensions.

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

    return np.array(centroids)


def assign_points(data, centroids):
    # INPUT: N x d data array, k x d centroids array.
    # OUTPUT: N x 1 array of cluster assignments in {0,...,k-1}.
    # print("assign_points")
    # start = time()
    clusters = np.array([np.argmin(np.linalg.norm(d - centroids, axis=1)) for d in data], dtype=np.int32)
    # print("Finished after %.2f seconds" % (time() - start))
    return clusters


def move_centroids(data, old_centroids, new_centroids, clusters):
    # print("move_centroids")
    # start = time()
    for c in range(old_centroids.shape[0]):
        new_centroids[c] = np.array([np.mean(data[clusters == c], axis=0)])
    # print("Finished after %.2f seconds" % (time() - start))

    return new_centroids


def nk_means_pp(file_name, original, data, k, n=10, num_iters=300, tol=1e-4, plot=False):
    q = PriorityQueue()
    for i in range(n):
        seed = np.random.randint(1000000)
        old_centroids, clusters = k_means_pp(data, k, random_seed=seed, num_iters=num_iters, tol=tol, plot=plot)
        output_name = output('temp/'+file_name + str(seed), clusters)
        value = calculate_value(output_name, original)
        q.put((value, i, seed, clusters))
    best = q.get()
    _, _, _, best_clusters = best
    print("Best output: " + str(best))
    return best_clusters


def k_means_pp(data, k, random_seed=1, num_iters=300, tol=1e-4, plot=False, debug=False):
    # INPUT: N x d data array, k number of clusters, number of iterations, boolean plot.
    # OUTPUT: N x 1 array of cluster assignments.
    if debug:
        print("Select centroids")
    old_centroids = select_centroids(data, k, random_seed)
    new_centroids = np.zeros(shape=old_centroids.shape)

    i = 0
    error = np.inf
    while i < num_iters and error > tol:
        # print("%s" % i)
        clusters = assign_points(data, old_centroids)

        # plotting
        if plot is True and i < 10:
            plotting(data, old_centroids, clusters)

        move_centroids(data, old_centroids, new_centroids, clusters)
        error = np.linalg.norm(new_centroids - old_centroids)
        old_centroids = new_centroids.copy()
        i += 1
    if debug:
        print("Converged after %s iterations" % i)
    plt.show()
    return old_centroids, clusters

# centroids, clusters = k_means(data, 2, plot=False)
# print("The final cluster mean values are:", centroids)
