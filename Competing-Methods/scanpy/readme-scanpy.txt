Results for competing method (1) diffusion map, (2) UMAP, (3) T-sne, (4) PCA, (5) ForceAtlas2 
were fitted using scanpy (scanpy 1.9.8 ), a single cell analysis tool. 

T-sne and ForceAtlas2 only in 2-dimension, and other methods in either 2- or 3- dimension. 

Scanpy Documentations: 
https://scanpy.readthedocs.io/en/stable/

Example use: 
Run phate on ToggleSwitch dataset in 2-dimensional space, with number of nearest neighbor = 15
python scanpy-methods.py DistMat_ToggleSwitch.txt 15 2

paramater: 
nns = 15 for ToggleSwitch and Olsson, nns = 20 for other datasets.

Note: Large datasets (like C. elegans, even the 10k version) take a large computation resources (both time and memory) to run and compute metrics. 