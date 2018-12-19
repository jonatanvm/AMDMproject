import sys

import pandas as pd

from sparse_spectral import sparse_spectral_clustering1, sparse_spectral_clustering2, custom_sparse_spectral_clustering2
from spectral import spectral_clustering1, spectral_clustering2, spectral_clustering3, ALGORITHM_1, \
    ALGORITHM_2, ALGORITHM_3, ALGORITHM_4, ALGORITHM_5
from calculate_comp_value import calculate_value
from sparse_spectral import *
from spectral import *


def output(name, values):
    df = pd.DataFrame(values)
    file_name = '../output/' + name + '.output'
    df.to_csv(file_name, index=True, header=False, sep=" ")
    return file_name


def run_all(loc, files, algorithm, out=True):
    for file in files:
        try:
            cluster_labels = None
            if algorithm == ALGORITHM_1:
                cluster_labels = spectral_clustering1(loc, file)
            elif algorithm == ALGORITHM_2:
                cluster_labels = spectral_clustering2(loc, file)
            elif algorithm == ALGORITHM_3:
                cluster_labels = spectral_clustering3(loc, file)
            elif algorithm == ALGORITHM_4:
                cluster_labels = sparse_spectral_clustering1(loc, file)
            elif algorithm == ALGORITHM_5:
                cluster_labels = sparse_spectral_clustering2(loc, file)
            elif algorithm == ALGORITHM_5:
                cluster_labels = custom_sparse_spectral_clustering2(loc, file)
            # cluster_labels = sparse_spectral_clustering1(loc, file)

            if out and cluster_labels.any():
                output_name = output(file.split(".")[0], cluster_labels)
                calculate_value(output_name, loc + file)
            print("Finished with %s \n" % file)
        except MemoryError:
            print("Ran out of memory!")
            pass


test_files = ['ca-HepTh.txt', 'ca-HepPh.txt', 'ca-AstroPh.txt', 'ca-CondMat.txt']
comp_files = ['ca-GrQc.txt', 'Oregon-1.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt', 'roadNet-CA.txt', ]
ptest_files = ['test1.txt', 'test2.txt', 'test3.txt']
ptest_files2 = ['test3.txt']

if __name__ == "__main__":
    if len(sys.argv[1:]) is 2:
        files = str(sys.argv[1])
        algorithm = int(sys.argv[2])
        if files == "comp":
            run_all('../graphs_competition/', comp_files, algorithm)
        elif files == "ptest":
            run_all('../graph_tests/', ptest_files, algorithm)
        elif files == "test":
            run_all('../graphs/', test_files, algorithm)
        else:
            print("Not enough arguments.")
            sys.exit()
    else:
        run_all('../graphs/', test_files, ALGORITHM_3, True)
        # run_all('../graph_tests/', ptest_files2, ALGORITHM_5, False)
        # run_all('../graphs_competition/', comp_files, ALGORITHM_4, True)
