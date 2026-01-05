'''
 k-means as clustering method
 feature matrix as input
 

 for generating different cluster labels
 input: feature matrix, number of clusters in each layer: list[int]
 output: cluster assignment in each layer: list[list[int]]

'''

import numpy as np

import sys, socket, time
import argparse

import random

from sklearn.cluster import KMeans

sys.path.append('../')

print(socket.gethostname())

random.seed()

def generate_cls(featmat,ncls):
    
    cluster_list = []
    
    curr_ = featmat
    
    for n_ in ncls:
        pred = KMeans(n_clusters = n_,n_init=10).fit(curr_)
        curr_ = pred.cluster_centers_
        
        if len(cluster_list) == 0:
            cluster_list.append(pred.labels_)
        else:
            cluster_list.append(pred.labels_[cluster_list[-1]])
        
    return np.array(cluster_list)

def savecls(cluster_list,datafdname,res):
    filename = 'cls_%s'%('_'.join([str(n) for n in res]))
    np.savetxt(datafdname+filename, cluster_list,fmt='%i')
    




if __name__ == "__main__":
   t = time.time()
   
   parser = argparse.ArgumentParser(
       description='Generate hierarchical cluster assignments using K-means on feature matrix',
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog='''
Example usage:
  python Generate_cls.py --featmat FeatMat_ToggleSwitch.txt --n-clusters 20,7
       '''
   )
   
   parser.add_argument('--featmat', type=str, required=True,
                       help='Feature matrix filename (e.g., FeatMat_ToggleSwitch.txt)')
   parser.add_argument('--n-clusters', type=str, required=True,
                       help='Number of clusters for each layer, comma-separated (e.g., 20,7)')
   
   args = parser.parse_args()
   
   featmatname = str(args.featmat.split('.')[0])
   datafdname = '../data/%s/'%(featmatname)

   if featmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+args.featmat, dtype=float)
   else: 
       print('  [Error] Must provide feature matrix. Program ended!\n')
       exit()
   
   ncls = [int(n) for n in args.n_clusters.split(',')]

   cluster_list = generate_cls(featmat,ncls)
   
   savecls(cluster_list,datafdname,ncls)

    
   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)



