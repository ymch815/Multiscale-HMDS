import phate
import numpy as np
import pandas as pd
import sys

import os

sys.path.append('../../')
import MHMDS.quality_metrics as qms
import MHMDS.embed_funs as emb

from scipy import stats

import time


# import scanpy.external as sce
print('sucessful imported')

def dimension_reduction(featmat,nd):
     # phate_op = phate.PHATE(knn_dist="euclidean", gamma=0, t=12, decay=15,n_components=nd)
     phate_op = phate.PHATE(n_components=nd)

     data_phate = phate_op.fit_transform(featmat)

     dmat_toemb = emb.get_dmat_euc(featmat)
     mat_emb = emb.get_dmat_euc(data_phate)

     qlocal,qglobal,_ = qms.get_quality_metrics(dmat_toemb,mat_emb,verbose=True)
     N_sample = len(dmat_toemb)
     corr = stats.pearsonr(mat_emb[np.triu_indices(N_sample, k=1)],
                                dmat_toemb[np.triu_indices(N_sample, k=1)]).statistic


     res = pd.DataFrame({'Method':['PHATE'],
                        'Qlocal':[qlocal],'Qglobal':[qglobal],'Corr':[corr]})
     return res,data_phate


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
   datafdname = '../../data/%s/'%(dmatname)
   nd = int(sys.argv[2])

   if nd not in [2,3]:
        print('invalid dimension')
        exit()

   # must be feature matrix

   if dmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)
   elif dmatname.split('_')[0].lower() == 'distmat':
        print('[Error] Not a feature matrix')
        exit()

   res,data_phate = dimension_reduction(featmat,nd)

   if not os.path.exists('./test/'):
    os.makedirs('./test/')
   targetdrname = './test/metrics_%s_nd_%i.csv'%(dmatname,nd)

   res.to_csv(targetdrname)

   np.savetxt('./test/phatemat_%s_nd_%i.txt'%(dmatname,nd),data_phate)
   

   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)