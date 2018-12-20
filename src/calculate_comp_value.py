def calculate_objective_function(original_file, cluster_labels):
    """
    Calculate goodness of partition, dividing number of edges between two vertices
    which are not in same cluster by number of vertices in smallest cluster.

    :param original_file: path for edges in graph
    :param cluster_labels: list of cluster labels for each vertice
    :return: Calculate goodness of partition
    """
    with open(original_file, 'r') as graph:
        _, _, edges, lines, k = graph.readline().split(" ")

        cluster_sizes = [0] * int(k)
        for i in cluster_labels:
            cluster_sizes[i] += 1
        min_value = min(cluster_sizes)
        edge_are_not_in_same_vertex = 0

        while True:
            line = graph.readline()
            if not line:
                break
            edge1, edge2 = line.split(" ")
            edge1 = int(edge1)
            edge2 = int(edge2)
            vertex1 = cluster_labels[edge1]
            vertex2 = cluster_labels[edge2]
            if vertex1 != vertex2:
                edge_are_not_in_same_vertex += 1

        value = edge_are_not_in_same_vertex / min_value
        return value
