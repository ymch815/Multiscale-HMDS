# Multiscale-H-MDS
An algorithm for performing hyperbolic MDS on large scale datasets.

This file covers set up and quick start guide. See [Tutorial](docs/1-Overview.md) for more detailed guides. 

## Dependencies

* cmdstanpy 1.1.0
* numpy 1.23.5
* scikit-learn 1.3.0
* pandas 2.0.3
* scipy 1.11.1


## Quick Start

### Compile .stan files
Stan files should be compiled before performing any embedding. 

To ensure compiling of .stan files, we recommend install only cmdstanpy in a independent environment, and compile all .stan files. 
**This only needs to be done once.** After executable files have been compiled, it will be ready to perform optimization and there's no need to re-compile. 

**Important check before first compiling**: Please make sure you only have .stan files in the directory, and remove executable files, .hpp files, etc. 
 
To compile, 
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
```bash
cd MultiscalehMDS_feature
python Multiscale_hmds.py --featmat FeatMat_ToggleSwitch.txt --clusters cls_20 \
       --neighbors 10 --dimension 3 --min-cluster-size 1 --max-cluster-size 200 \
       --save-matrix --compute-metrics --map-outlier --outlier-neighbors 10
```

To see all available options:
```bash
python Multiscale_hmds.py -h
```

## Usage

### MultiscalehMDS_feature and MultiscalehMDS_distance

We provide options for starting from either **distance matrix** or **feature matrix**. Depending on the matrix you have, you should go to either of the directory before performing the following steps. 

We recommend using the feature matrix, since it is computationally more efficient. 

### Clustering

Generate_cls.py performs clustering on the dataset. 

* In MultiscalehMDS_feature (K-means clustering)
```bash
python Generate_cls.py --featmat FeatMat_ToggleSwitch.txt --n-clusters 20,7
```

* In MultiscalehMDS_distance (Agglomerative clustering)
```bash
python Generate_cls.py --distmat DistMat_ToggleSwitch.txt --thresholds 0.2,0.6
```

To see all available options:
```bash
python Generate_cls.py -h
```

### Embedding
After getting cluster assignment, Multiscale_hmds.py performs embedding. 

* In MultiscalehMDS_feature
```bash
python Multiscale_hmds.py --featmat FeatMat_ToggleSwitch.txt --clusters cls_20 \
       --neighbors 10 --dimension 3 --min-cluster-size 1 --max-cluster-size 200 \
       --save-matrix --compute-metrics --map-outlier --outlier-neighbors 10
```

* In MultiscalehMDS_distance
```bash
python Multiscale_hmds.py --distmat DistMat_ToggleSwitch.txt --clusters cls_0.2_0.6 \
       --neighbors 20,10 --dimension 3 --min-cluster-size 4 --max-cluster-size 40 \
       --save-matrix --compute-metrics --map-outlier --outlier-neighbors 10
```

#### Parameters:
- `--featmat` / `--distmat`: Name of dataset file (feature or distance matrix)
- `--clusters`: Name of cluster assignment file
- `--neighbors`: Number of neighbors for each layer (comma-separated for multiple layers)
- `--dimension`: Number of embedding dimensions
- `--min-cluster-size`: Minimum cluster size for filtering outliers
- `--max-cluster-size`: Maximum cluster size allowed. Large clusters will be re-divided
- `--save-matrix`: Save embedding coordinate matrices (flag)
- `--compute-metrics`: Compute quality metrics (flag)
- `--map-outlier`: Map outliers to embedding space (flag)
- `--outlier-neighbors`: Number of neighbors for outlier mapping

### Outputs

The program saved embedding coordinates for both cluster centroids and individual samples, if `--save-matrix == True`. 

It also saved a list of index indicating the order of samples after filtering and re-ordering, and the **embedding coordinates comes in this order**. Note that if `--min-cluster-size > 1` and `--map-outlier == False`, outliers are not embedded and `len(list of index)` will be smaller than the original sample size. 


## Reproduce simulation results and figures
Code, data and embedding results required for all of the figures and supplementary figures are included in this repository. Jupyter notebooks are provided for plotting the figures in `figure` and `WordNet`

Embedding results are provided in MultiscalehMDS_feature/result and MultiscalehMDS_distance/result. To reproduce them, please read the .txt files inside these directories which provided all parameters to reproduce the embeddings. 

## Resources
For more detailed tutorial, please refer to the following docs: 
- **Algorithm intuition**: See [1-Overview.md](1-Overview.md) for conceptual background
- **Example usage**: See [2-Example Usage.md](2-Example%20Usage.md) for hands-on tutorials
- **Interactive examples**: Run [example-usage.ipynb](example-usage.ipynb) for working code for 2-Example Usage.md
- **Parameter selection**:Review [3-Parameters](3-Parameters.md) for tuning advice
