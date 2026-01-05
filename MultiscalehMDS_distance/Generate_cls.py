'''
 AgglomerativeClustering as clustering method
 feature matrix or distance matrix as input
 

 for generating different cluster labels
 input: feature matrix/distance matrix, 
         resolution of hierarchical clustering in each layer: list[float], 0<x<1
 output: cluster assignment in each layer: list[list[int]]

'''

import numpy as np

import sys, socket, time
import argparse

import random

from sklearn.cluster import AgglomerativeClustering

sys.path.append('../')
import MHMDS.embed_funs as emb


print(socket.gethostname())

random.seed()

def generate_cls(dmat,res):
    dmat_norm = 2.0*dmat/np.max(dmat)
    # check layers 
    cluster_list = []
    for thr in res:
        cluster_list.append(AgglomerativeClustering(n_clusters = None, \
                            metric = 'precomputed',linkage = 'complete', \
                            distance_threshold=thr).fit(dmat_norm).labels_)
    return np.array(cluster_list)

def savecls(cluster_list,datafdname,res):
    filename = 'cls_%s'%('_'.join([str(n) for n in res]))
    np.savetxt(datafdname+filename, cluster_list,fmt='%i')
    

'''
for generating different cluster labels
input: dmat, resolution of clustering[list[float]]
'''


if __name__ == "__main__":
   t = time.time()
   
   parser = argparse.ArgumentParser(
       description='Generate hierarchical cluster assignments using Agglomerative Clustering',
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog='''
Example usage:
  python Generate_cls.py --distmat DistMat_ToggleSwitch.txt --thresholds 0.2,0.6
       '''
   )
   
   parser.add_argument('--distmat', type=str, required=True,
                       help='Distance or feature matrix filename (e.g., DistMat_ToggleSwitch.txt)')
   parser.add_argument('--thresholds', type=str, required=True,
                       help='Distance thresholds for each layer, comma-separated (e.g., 0.2,0.6). Values between 0 and 1')
   
   args = parser.parse_args()
   
   dmatname = str(args.distmat.split('.')[0])
   datafdname = '../data/%s/'%(dmatname)

   if dmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+args.distmat, dtype=float)
        dmat = emb.get_dmat_euc(featmat)
   elif dmatname.split('_')[0].lower() == 'distmat':
        dmat  = np.loadtxt(datafdname+args.distmat, dtype=float)

   res = [float(n) for n in args.thresholds.split(',')]
   
   cluster_list = generate_cls(dmat,res)
   
   savecls(cluster_list,datafdname,res)

   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)



