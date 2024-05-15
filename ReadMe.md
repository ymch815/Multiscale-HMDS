# Multiscale-H-MDS
An algorithm for performing hyperbolic MDS on large scale datasets

## Dependencies

* cmdstanpy 1.1.0
* numpy 1.23.5
* scikit-learn 1.3.0
* pandas 2.0.3
* scipy 1.11.1


## Quick Start

### Compile .stan files
Stan files should be compiled before performing any embedding. 

To ensure compiling of .stan files, we recommend install only cmdstanpy in a independent environment, and compile all .stan files. This only needs to be done once. After executable files have been compiled, it will be ready to perform optimization and there's no need to re-compile. 

To do so, 
```
conda create -n stan-env -c conda-forge cmdstanpy
conda activate stan-env 
cd MHMDS 
python compile_models.py
```

* Note: if there is anything wrong about the cmdstanpy, you can try to reinstall it:
```
from cmdstanpy import install_cmdstan
install_cmdstan(overwrite = True)
```
### Example use
Perform Multiscale H-MDS on the Toggle Switch dataset onto 3-d hyperbolic space
```
cd MultiscalehMDS_feature
python Multiscale_hmds.py FeatMat_ToggleSwitch.txt cls_20 10 3 1 200 1 1 1 10 0
```

## Usage

### MultiscalehMDS_feature and MultiscalehMDS_distance

We provide options for starting from either **distance matrix** or **feature matrix**. Depending on the matrix you have, you should go to either of the directory before performing the following steps. 

We recommend using the feature matrix, since it is computationally more efficient. 

### Clustering

Generate_cls.py performs clustering on the dataset. 

* In MultiscalehMDS_feature
```
python Generate_cls.py FeatMat_ToggleSwitch.txt 20,7  
```
* In MultiscalehMDS_distance
```
python Generate_cls.py DistMat_ToggleSwitch.txt 0.2,0.6
```

### Embedding
After getting cluster assignment, Multiscale_hmds.py performs embedding. 

* In MultiscalehMDS_feature
```
python Multiscale_hmds.py FeatMat_ToggleSwitch.txt cls_20 10 3 1 200 1 1 1 10 0
```
* In MultiscalehMDS_distance
```
python Multiscale_hmds.py DistMat_ToggleSwitch.txt cls_0.2_0.6 20,10 3 4 40 1 1 1 10 0
```

parameters: `name of dataset file` `name of cluster assignment file` `n neighbors` `n dimension` `min cluster size` `max cluster size` `save matrix or not` `compute metrics or not` `map outlier or not` `n neighbors for outlier` `correction`

### Outputs

The program saved embedding coordinates for both cluster centroids and individual samples, if `save matrix or not == True`. 

It also saved a list of index indicating the order of samples after filtering and re-ordering, and the **embedding coordinates comes in this order**. Note that if `min cluster size > 1` and `map outlier or not == False`, outliers are not embedded and `len(list of index)` will be smaller than the original sample size. 


## Reproduce simulation results and figures
Code, data and embedding results required for all of the figures and supplementary figures are included in this repository. Jupyter notebooks are provided for plotting the figures in `figure` and `WordNet`

Embedding results are provided in MultiscalehMDS_feature/result and MultiscalehMDS_distance/result. To reproduce them, please read the .txt files inside these directories which provided all parameters to reproduce the embeddings. 

