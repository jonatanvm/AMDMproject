import os
import sys

import pandas as pd

from calculate_comp_value import calculate_objective_function
from sparse_spectral import *
from spectral import *

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(MODULE_PATH)


def output(name, values, header):
    """
    Output the cluster labels to a file <name>.output
    :param name: of file
    :param values: cluster labels
    :param header: header for file
    :return: path to file.
    """
    output_path = ROOT_PATH + '/output/' + name + '.output'

    # Add header
    header = ' '.join(header)
    if not header[-1] == '\n':
        header += '\n'

    with open(output_path, 'w') as f:
        f.write(header)
        f.close()

    df = pd.DataFrame(values)
    df.to_csv(output_path, index=True, header=False, sep=" ", mode='a')

    return output_path


def run_all(files, algorithm, e_mode, n, n_jobs, out=True):
    for file_name in files:
        try:
            cluster_labels = None
            path = ROOT_PATH + '/graphs/' + file_name
            if algorithm == ALGORITHM_1:
                cluster_labels, seed, header = spectral_clustering1(path)
            elif algorithm == ALGORITHM_2:
                cluster_labels, seed, header = spectral_clustering2(path)
            elif algorithm == ALGORITHM_3:
                cluster_labels, seed, header = custom_sparse_spectral_clustering1(path, e_mode=e_mode, n=n, n_jobs=n_jobs)
            elif algorithm == ALGORITHM_4:
                cluster_labels, seed, header = custom_sparse_spectral_clustering2(path, e_mode=e_mode, n=n, n_jobs=n_jobs)

            if out and cluster_labels.any():
                output(file_name.split(".")[0] + "-" + str(seed), cluster_labels, header)
                value = calculate_objective_function(path, cluster_labels)
                print("Competition value: %s" % value)
            print("Finished with %s \n" % file_name)
        except MemoryError:
            print("Ran out of memory!")
            pass


test_files = ['ca-HepTh.txt', 'ca-HepPh.txt', 'ca-AstroPh.txt', 'ca-CondMat.txt']
comp_files = ['ca-GrQc.txt', 'Oregon-1.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt', 'roadNet-CA.txt', ]
ptest_files = ['test1.txt', 'test2.txt', 'test3.txt']

if __name__ == "__main__":
    n_args = len(sys.argv[1:])
    if n_args > 2:
        files = str(sys.argv[1])
        algorithm = int(sys.argv[2])

        if n_args is 2:
            e_mode = 'eigsh'
            n = 10
            n_jobs = 4
        if n_args is 5:
            e_mode = str(sys.argv[3])
            n = int(sys.argv[4])
            n_jobs = int(sys.argv[5])
        else:
            print("Invalid parameter line.")
            sys.exit()

        if files == "comp":
            run_all(comp_files, algorithm, e_mode, n, n_jobs)
        elif files == "ptest":
            run_all(ptest_files, algorithm, e_mode, n, n_jobs)
        elif files == "test":
            run_all(test_files, algorithm, e_mode, n, n_jobs)
        elif files in test_files or files in comp_files or files in ptest_files:
            run_all([files], algorithm, e_mode, n, n_jobs)
        else:
            print("Invalid parameter line.")
            sys.exit()

    else:
        # run_all(test_files, ALGORITHM_3, 'eigsh', n=100, n_jobs=10)
        run_all(comp_files, ALGORITHM_4, 'eigsh', n=1, n_jobs=10)
        # run_all(comp_files, ALGORITHM_1, 'eigsh', n=10, n_jobs=10)
