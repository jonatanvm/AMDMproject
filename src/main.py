import pandas as pd

from src.spectral import spectral_clustering1, ALGORITHM, spectral_clustering2, spectral_clustering3


def output(name, values):
    df = pd.DataFrame(values)
    df.to_csv('../output/' + name + '.output', index=True, header=False, sep=" ")


def run_all(loc, files, algorithm, out=True):
    for file in files:
        cluster_labels = None
        if algorithm is ALGORITHM._1:
            cluster_labels = spectral_clustering1(loc, file)
        elif algorithm is ALGORITHM._2:
            cluster_labels = spectral_clustering2(loc, file)
        elif algorithm is ALGORITHM._3:
            cluster_labels = spectral_clustering3(loc, file)
        # cluster_labels = sparse_spectral_clustering1(loc, file)

        if out and cluster_labels.any():
            output(file.split(".")[0], cluster_labels)


test_files = ['ca-AstroPh.txt', 'ca-CondMat.txt', 'ca-HepPh.txt', 'ca-HepTh.txt']
comp_files = ['Oregon-1.txt', 'roadNet-CA.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt']
ptest_files = ['test1.txt', 'test2.txt', 'test3.txt']

# run_all('../graphs/', test_files)
run_all('../graph_tests/', ptest_files, ALGORITHM._2)
# run_all('../graphs_competition/', comp_files)
