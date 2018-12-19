import numpy as np


def calculate_value(output_name, original_file):
    print(output_name)
    with open(original_file, 'r') as graph:
        _, _, edges, lines, k = graph.readline().split(" ")

    graph.close()

    with open(output_name, 'r') as output_file:
        k = int(k)
        vertices = np.arange(k).reshape((k, 1))
        count = np.zeros(k).reshape((k, 1))
        results = np.concatenate((vertices, count), axis=1)

        edges = int(edges)
        edges2 = np.arange(edges).reshape((edges, 1))
        vertice = np.arange(edges).reshape((edges, 1))
        output = np.concatenate((edges2, vertice), axis=1)

        while True:
            line = output_file.readline()
            if not line:
                break
            edge, vertice = line.split(" ")
            output[int(edge), 1] = vertice
            results[int(vertice), 1] += 1
        results = results.astype(int)
        min_value = results.min(0)[1]

        edgeAreNotInSameVertice = 0

    with open(original_file, 'r') as graph:
        _, _, edges, lines, k = graph.readline().split(" ")

        while True:
            line = graph.readline()
            if not line:
                break
            edge1, edge2 = line.split(" ")
            edge1 = int(edge1)
            edge2 = int(edge2)
            vertice1 = output[edge1, 1]
            vertice2 = output[edge2, 1]
            if (vertice1 != vertice2):
                edgeAreNotInSameVertice += 1

        value = edgeAreNotInSameVertice / min_value
        print("value for competition = " + str(value))
        return value
