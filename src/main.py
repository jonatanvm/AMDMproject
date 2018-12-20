import sys

import pandas as pd

from calculate_comp_value import calculate_value
from sparse_spectral import *
from spectral import *


def output(name, values):
    df = pd.DataFrame(values)
    file_name = '../output/' + name + '.output'
    df.to_csv(file_name, index=True, header=False, sep=" ")
    return file_name


def run_all(files, algorithm, out=True):
    for file_name in files:
        try:
            cluster_labels = None
            path = '../graphs/' + file_name
            if algorithm == ALGORITHM_1:
                cluster_labels, seed = spectral_clustering1(path)
            elif algorithm == ALGORITHM_2:
                cluster_labels, seed = spectral_clustering2(path)
            elif algorithm == ALGORITHM_3:
                cluster_labels, seed = custom_sparse_spectral_clustering1(path, e_mode='eigsh')
            elif algorithm == ALGORITHM_4:
                cluster_labels, seed = custom_sparse_spectral_clustering2(path, e_mode='lobpcg')

            if out and cluster_labels.any():
                output(file_name.split(".")[0] + "-" + str(seed), cluster_labels)
                value = calculate_value(path, cluster_labels)
                print("Competition value: %s" % value)
            print("Finished with %s \n" % file_name)
        except MemoryError:
            print("Ran out of memory!")
            pass


test_files = ['ca-HepTh.txt', 'ca-HepPh.txt', 'ca-AstroPh.txt', 'ca-CondMat.txt']
comp_files = ['ca-GrQc.txt', 'Oregon-1.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt', 'roadNet-CA.txt', ]
ptest_files = ['test1.txt', 'test2.txt', 'test3.txt']

if __name__ == "__main__":
    if len(sys.argv[1:]) is 2:
        files = str(sys.argv[1])
        algorithm = int(sys.argv[2])
        if files == "comp":
            run_all(comp_files, algorithm)
        elif files == "ptest":
            run_all(ptest_files, algorithm)
        elif files == "test":
            run_all(test_files, algorithm)
        elif files in test_files or files in comp_files or files in ptest_files:
            run_all([files], algorithm)
        else:
            print("Not enough arguments.")
            sys.exit()

    else:
        # run_all(test_files, ALGORITHM_7, True)
        run_all(ptest_files, ALGORITHM_4, True)
        # run_all(comp_files, ALGORITHM_4, True)
