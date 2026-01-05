# env: scanpy-env 
import scanpy as sc
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

def dimension_reduction(featmat,nns,nd,dmatname):
    # featmat: numpy array
    adata = sc.AnnData(featmat)

    dmat_toemb = emb.get_dmat_euc(featmat) # get dmat for metric computation

    sc.pp.neighbors(adata,n_neighbors=nns)

    sc.tl.diffmap(adata,n_comps=nd+1)
    sc.tl.tsne(adata)
    sc.tl.umap(adata,n_components=nd)
    sc.tl.pca(adata, n_comps=nd,svd_solver='auto')

    sc.tl.louvain(adata, resolution=0.9)

    sc.tl.paga(adata)
    sc.tl.draw_graph(adata)

    # compute metrics for each method
    methods = {'DiffMap':'X_diffmap','t-sne':'X_tsne','umap':'X_umap','PCA':'X_pca','ForceAtlas2':'X_draw_graph_fr'}
    Qlocals = []
    Qglobals = []
    corrs = []
    for method in methods:
        name = methods[method]
        if name == 'X_diffmap':
            mat_emb = emb.get_dmat_euc(adata.obsm[name][:,1:1+nd])
            # save the mat
            np.savetxt('./test/%s_%s_nn_%i_nd_%i.txt'%(method,dmatname,nns,nd),adata.obsm[name][:,1:1+nd])
        else:
            mat_emb = emb.get_dmat_euc(adata.obsm[name])
            np.savetxt('./test/%s_%s_nn_%i_nd_%i.txt'%(method,dmatname,nns,nd),adata.obsm[name])
        print(method)
        Qlocal, Qglobal, _  = qms.get_quality_metrics(dmat_toemb,mat_emb,verbose=True)
        Qlocals.append(Qlocal)
        Qglobals.append(Qglobal)
        N_sample = len(dmat_toemb)
        corrs.append(stats.pearsonr(mat_emb[np.triu_indices(N_sample, k=1)],
                                dmat_toemb[np.triu_indices(N_sample, k=1)]).statistic)

    res = pd.DataFrame({'Method':['DiffMap','t-sne','umap','PCA','ForceAtlas2'],
                        'Qlocal':Qlocals,'Qglobal':Qglobals,'Corr':corrs})
    return res



if __name__ == "__main__":
   t = time.time() # Count the time of running
   
   #first argument is python filename
   print("the name of the program:", sys.argv[0])
   print("----------------------------------------------------\n")

   if len(sys.argv) < (1+3):
      print('  [Error] No input argument provided. Program ended!\n')
      exit()

   #set global parameters and load distance matrix
   
   dmatname = str(sys.argv[1].split('.')[0])
   datafdname = '../../data/%s/'%(dmatname)
   nns = int(sys.argv[2])
   nd = int(sys.argv[3])

   if nd not in [2,3]:
        print('invalid dimension')
        exit()

   # must be feature matrix

   if dmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)
   elif dmatname.split('_')[0].lower() == 'distmat':
        print('[Error] Not a feature matrix')
        exit()

   

   if not os.path.exists('./test/'):
    os.makedirs('./test/')
   targetdrname = './test/metrics_%s_nn_%i_nd_%i.csv'%(dmatname,nns,nd)

   res = dimension_reduction(featmat,nns,nd,dmatname)

   res.to_csv(targetdrname)
   

   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)