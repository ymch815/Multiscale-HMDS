Results for competing method Poincare Map was simulated using code from the following repository: 
https://github.com/facebookresearch/PoincareMaps 

Embedding parameters for each dataset: 

python main.py --dset ToggleSwitch  --batchsize -1 --cuda 0 --knn 15 --gamma 2.0 --sigma 1.0 --pca 0  --root root

python main.py --dset Krumsiek11 --batchsize -1 --cuda 0 --knn 30 --gamma 2.0 --sigma 2.0 --pca 0  --root root

python main.py --dset Paul_PCA100 --batchsize -1 --cuda 0 --knn 15 --gamma 2.0 --sigma 1.0 --pca 0 --root root

python main.py --dset Olsson_PCA100   --batchsize -1 --cuda 0 --knn 15 --gamma 2.0 --sigma 1.0 --pca 0 --root HSPC-1

python main.py --dset Celegans_10k_PCA100 --batchsize -1 --cuda 0 --knn 30 --gamma 1.0 --sigma 2.0 --pca 0  --root root

python main.py --dset Celegans_40k_PCA100 --batchsize -1 --cuda 0 --knn 30 --gamma 1.0 --sigma 2.0 --pca 0  --root root
