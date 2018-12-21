# AMDMproject
Algorithmic methods for data mining course project.

# Before you run
Install the requirements listed in the requirements.txt file.
If you have pip you can run.

> pip install -r requirements.txt

# How to use

> python main.py \<data_set> \<alg_num> \<dec_alg>  \<n_ki> \<nt>


Where:  
**\<data_set>** = Data set (test, ptest or comp)  
**\<alg_num>** = Algorithm number (1-4)  
**\<dec_alg>** = decomposition algorithm (eigsh or lobpcg)  
**\<n_ki>** = unumber of k-means++ iterations (1 -\>)   
**\<nt>** = number of threads to parallelize the k-means++ iterations (1 -\>)    

##### Algorithm param \<alg_num>

Takes values:  
**1** = unnormalized spectral partitioning with k-means  
**2** = normalized spectral partitioning with k-means  
**3** = unnormalized sparse spectral partitioning with custom nk-means++  
**4** = normalized sparse spectral partitioning with custom nk-means++  

__normalized__ and __unnormalized__ refers to how the laplacian and eigenvalue decomposition is calculated.  

**nk-means++** is a custom k-means++ algorithm which runs k-means n times and returns the clustering with the best objective function.

### Examples
Either 2 or all five parameters have to be specified.  

If only two parameters a specified k-means++ will be run 10 times with 4 threads.

##### Run competition files with algorithm 4

> python main.py comp 4

##### Run test files with algorithm 3

> python main.py test 3

##### Run private test files with algorithm 2

> python main.py ptest 2

##### Run single file with algorithm 3
Graph has to be in the graphs directory

> python main.py ca-AstroPh.output 3

##### Run competition files with algorithm 4 with 10 k-means++ iterations parallelized to 10 threads.

> python main.py comp 4 10 10

#### Reproduce best results
Depending on the parameters and the fact the best graph partition is found semi-randomly, the best results might not be reproduced every time.
So to reproduce the outputs for the competition run the following script.

> python run_best.py \<data_set>

Where:  
**\<data_set>** = Data set (test or comp)  

Example:

> python run_best.py comp

## Best results

* AstroPh: custom sparse algo1 - 40.667  
Seed: 95504
* CondMat: custom sparse algo1 - 124  
Seed: 31882
* HepPh: custom sparse algo1 - 14.25  
Seed: 748819
* HepTh: custom sparse algo1 - 5.6  
Seed: 380854
* GrQc: custom sparse algo2 - 0.075  
Seed: 834113
* Oregon: custom sparse algo2 - 2.242   
Seed: 952417
* soc-Epinions: custom sparse algo2 - 1.571  
Seed: 987761
* web-NotreDame: custom sparse algo2 - 0.262  
Seed: 361760
* roadNet-CA: custom sparse algo2 - 0.379   
Seed: 826647
