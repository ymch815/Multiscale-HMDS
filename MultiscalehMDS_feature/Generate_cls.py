'''
 k-means as clustering method
 feature matrix as input
 

 for generating different cluster labels
 input: feature matrix, number of clusters in each layer: list[int]
 output: cluster assignment in each layer: list[list[int]]

'''

import numpy as np

import sys, socket, time

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
   t = time.time() # Count the time of running
   
   #first argument is python filename
   print("the name of the program:", sys.argv[0])
   print("----------------------------------------------------\n")

   if len(sys.argv) < (1+2):
      print('  [Error] No input argument provided. Program ended!\n')
      exit()

   #set global parameters and load distance matrix
   
   featmatname = str(sys.argv[1].split('.')[0])
   datafdname = '../data/%s/'%(featmatname)

   if featmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)
   else: 
       print('  [Error] Must provide feature matrix. Program ended!\n')
       exit()
   
   '''
   In this case, we need the number of clusters in each layer
   Then we need a function to "group" the clusters into higher levels
   '''
   ncls = [int(n) for n in sys.argv[2].split(',')]

   cluster_list = generate_cls(featmat,ncls)
   
   savecls(cluster_list,datafdname,ncls)

    
   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)



