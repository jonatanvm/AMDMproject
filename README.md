# AMDMproject
Algorithmic methods for data mining course project

# How to use

##### Run competition files

> python main.py comp _algnum_

##### Run test files

> python main.py test _algnum_

##### Run private test files

> python main.py ptest _algnum_

##### Run single file
Graph has to be in the graphs directory

> python main.py ca-AstroPh.output _algnum_

##### Algorithm param _algnum_

Takes values:  
**1** = unnormalized spectral partitioning with k-means  
**2** = normalized spectral partitioning with k-means  
**3** = unnormalized sparse spectral partitioning with custom nk-means++  
**4** = normalized sparse spectral partitioning with custom nk-means++  

Example:
> python main.py ca-AstroPh.output 4

__normalized__ and __unnormalized__ refers to how the laplacian and eigenvalue decomposition is calculated.  

**nk-means++** is a custom k-means++ algorithm which runs k-means n times and returns the clustering with the best objective function.


# Objective function


# Algorithm

## Basic steps in algorithm

* Read graph
* Make adjacency matrix
* From Laplacian of the adjacency matrix
* Cumpute eigen-decomposition of Laplacian
* Apply k-means on the vector representation of the vertices provided by the eigenvectors.


## Best results

* AstroPh: custom sparse algo1 - 48
* CondMat: custom sparse algo1 -154
* HepPh: custom sparse algo1 - 14.25 - 748819
* HepTh: custom sparse algo1 - 5.6 - 380854
* GrQc: custom sparse algo2 - 0.075
* Oregon: custom sparse algo2 - 2.8181818181818183 - 278141
* soc-Epinions: custom sparse algo2 - 1.5714285714285714
* web-NotreDame: custom sparse algo2 - 0.2621951219512195
* roadNet-CA: custom sparse algo2 - 12.3043
