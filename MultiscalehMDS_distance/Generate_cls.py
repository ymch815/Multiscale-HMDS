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
   t = time.time() # Count the time of running
   
   #first argument is python filename
   print("the name of the program:", sys.argv[0])
   print("----------------------------------------------------\n")

   if len(sys.argv) < (1+2):
      print('  [Error] No input argument provided. Program ended!\n')
      exit()

   #set global parameters and load distance matrix
   
   dmatname = str(sys.argv[1].split('.')[0])
   datafdname = '../data/%s/'%(dmatname)

   if dmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)
        dmat = emb.get_dmat_euc(featmat)
   elif dmatname.split('_')[0].lower() == 'distmat':
        dmat  = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)

   res = [float(n) for n in sys.argv[2].split(',')]
   
   cluster_list = generate_cls(dmat,res)
   
   savecls(cluster_list,datafdname,res)

   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)



