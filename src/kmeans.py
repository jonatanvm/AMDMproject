import matplotlib.pyplot as plt
import numpy as np

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


def select_centroids(data, k, random_seed=1):
    # INPUT: N x d data array, k number of clusters.
    # OUTPUT: k x d array of k randomly assigned mean vectors with d dimensions.

    # np.random.seed(seed=random_seed)
    print("Select")
    print(data)
    print(data.shape)
    centroids = np.zeros((k, data.shape[1]))
    for i in range(data.shape[1]):
        centroids[:, i] = np.random.uniform(np.min(data[:, i]), np.max(data[:, i]), size=k)
        print(centroids[:, i])
    print(centroids)
    print(centroids.shape)
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

        # plotting
        if plot == True and i < 3:
            plotting(data, centroids, clusters)

        centroids, assigned = move_centroids(data, centroids, clusters)
        ass = assigned
        i += 1

    plt.show()
    return centroids, clusters

# centroids, clusters = k_means(data, 2, plot=False)
# print("The final cluster mean values are:", centroids)
