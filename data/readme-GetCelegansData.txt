If you got a version of the repository without the following files:
/FeatMat_Celegans_10k_PCA100/FeatMat_Celegans_10k_PCA100.txt
/FeatMat_Celegans_40k_PCA100/FeatMat_Celegans_40k_PCA100.txt
/FeatMat_Celegans_PCA100/FeatMat_Celegans_PCA100.txt

This is because the submission has a constraint on file size. Therefore we removed the original feature matrix of C. elegans datasets to reduce size. 

In this situation, we provide codes to obtain the same feature matrix, if the reader want to reproduce the embedding results: 
1. Go to ../preprocessing/preprocessing_data_scanpy.ipynb, find section "C. elegans" and follow the instructions to download the original sequencing dataset from Packer et al 2019. 
2. Run the preprocessing pipeline, which generates the feature matrix for 85333 cells and save it. This gives FeatMat_Celegans_PCA100.txt
3. Load the metadata provided in /data/metadata/Celegans_10k_PCA100_metadata.csv and Celegans_40k_PCA100_metadata.csv and use the index to extract data from the full C. Elegans dataset. 