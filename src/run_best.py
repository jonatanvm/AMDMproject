import sys

from main import ROOT_PATH, output
from sparse_spectral import custom_sparse_spectral_clustering2, custom_sparse_spectral_clustering1


def run_best_test():
    cluster_labels, seed, header = custom_sparse_spectral_clustering1(ROOT_PATH + '/graphs/ca-HepTh.txt', n=1, n_jobs=1, seed=380854)
    output('ca-HepTh', cluster_labels, header)
    cluster_labels, seed, header = custom_sparse_spectral_clustering1(ROOT_PATH + '/graphs/ca-HepPh.txt', n=1, n_jobs=1, seed=748819)
    output('ca-HepPh', cluster_labels, header)
    cluster_labels, seed, header = custom_sparse_spectral_clustering1(ROOT_PATH + 'graphs/ca-AstroPh.txt', n=1, n_jobs=1, seed=95504)
    output('ca-AstroPh', cluster_labels, header)
    cluster_labels, seed, header = custom_sparse_spectral_clustering1(ROOT_PATH + 'graphs/ca-CondMat.txt', n=1, n_jobs=1, seed=31882)
    output('ca-CondMat', cluster_labels, header)


def run_best_comp():
    cluster_labels, seed, header = custom_sparse_spectral_clustering2(ROOT_PATH + '/graphs/ca-GrQc.txt', n=1, n_jobs=1, seed=834113)
    output('ca-GrQc', cluster_labels, header)
    cluster_labels, seed, header = custom_sparse_spectral_clustering2(ROOT_PATH + '/graphs/Oregon-1.txt', n=1, n_jobs=1, seed=952417)
    output('Oregon-1', cluster_labels, header)
    cluster_labels, seed, header = custom_sparse_spectral_clustering2(ROOT_PATH + '/graphs/soc-Epinions1.txt', n=1, n_jobs=1, seed=987761)
    output('soc-Epinions1', cluster_labels, header)
    print('\n' + '*' * 30)
    print("Warning graph partitioning for web-NotreDame.txt is slow\nand took about an hour on a 12 threaded Ryzen 2600 processor!")
    print('*' * 30 + '\n')
    cluster_labels, seed, header = custom_sparse_spectral_clustering2(ROOT_PATH + '/graphs/web-NotreDame.txt', n=1, n_jobs=1, seed=361760)
    output('web-NotreDame', cluster_labels, header)
    print('\n' + '*' * 30)
    print("Warning graph partitioning for roadNet-CA.txt is slow\nand took about 15 hours on the force.aalto.fi network!")
    print('*' * 30 + '\n')
    cluster_labels, seed, header = custom_sparse_spectral_clustering2(ROOT_PATH + '/graphs/roadNet-CA.txt', n=1, n_jobs=1, seed=826647)
    output('roadNet-CA', cluster_labels, header)


if __name__ == "__main__":
    n_args = len(sys.argv[1:])
    if n_args == 1:
        data_set = str(sys.argv[1])
        if data_set == 'comp':
            run_best_comp()
        elif data_set == 'test':
            run_best_test()
        else:
            print("Invalid parameter line. First parameter should be 'comp' or 'test'")
            sys.exit()
    else:
        print("Invalid parameter line. Should only include one parameter!")
        sys.exit()
