import pandas as pd

from code.spectral import spectral_clustering1


def output(name, values):
    df = pd.DataFrame(values)
    df.to_csv('../output/' + name + '.output', index=True, header=False, sep=" ")


test_files = ['ca-AstroPh.txt', 'ca-CondMat.txt', 'ca-HepPh.txt', 'ca-HepTh.txt']
comp_files = ['Oregon-1.txt', 'roadNet-CA.txt', 'soc-Epinions1.txt', 'web-NotreDame.txt']
ptest_files = ['test1.txt', 'test2.txt', 'test3.txt']


def run_all(loc, files, out=True):
    for file in files:
        cluster_labels = spectral_clustering1(loc, file)
        # cluster_labels = sparse_spectral_clustering1(loc, file)

        if out:
            output(file.split(".")[0], cluster_labels)


# run_all('../graphs/', test_files)
run_all('../graph_tests/', ptest_files)
# run_all('../graphs_competition/', comp_files)
