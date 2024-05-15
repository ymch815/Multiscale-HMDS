#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for large scale embedding method 
"""
import numpy as np
import time
import os


import sys
sys.path.append(os.getcwd())
import MHMDS.embed_funs as emb

import cmdstanpy as stan

from sklearn.manifold import MDS

cpath =  os.path.dirname(__file__)

import warnings
warnings.filterwarnings("ignore")

# define function for averaging embedding mat
'''
For cluster = 0:Ncluster
dmat -> avedmat(dmat, currcluster)
1114 modification: shrink the mat by the order of variance
'''
def avedmat(dmat, labels): # dmat and labels must be arranged by clusters
    labellist = np.unique(labels) # must be arranged in the cluster order in the dmat
    Nlabels = len(labellist)
    
    # compute mean of intra-distance in each group
    mean_interd = []
    labellens = []
    for i in range(Nlabels):
        ind_ = np.where(labels == labellist[i])[0]
        mean_interd.append(np.mean(dmat[np.ix_(ind_,ind_)].flatten()))
        labellens.append(len(ind_))
    labellens = np.array(labellens)
    ords = np.argsort(mean_interd)
    
    shrinked = np.zeros((len(ords),))
    
    # shrink the mat by order of variance
    for idx in ords:
        # current location of the cluster in current dmat = np.arange(len(label)) + displacement
        # displacement += all clusters of smaller index, length if not shrinked, 1 if shrinked
        if idx == 0:
            disp = 0
        else:
            disp = np.dot(1-shrinked[:idx],labellens[:idx]) + np.sum(shrinked[:idx])
        currinds = int(disp) + np.arange(len(np.where(labels == labellist[idx])[0])) # location of the cluster in current dmat
        if len(currinds)>1: # if only one sample: no need to update dmat itself
            dmat = aveone(dmat,currinds) # update dmat
        shrinked[idx] = 1.0
    return dmat
    
def aveone(dmat,currinds):
    Ncluster = len(currinds)
    
    cross = np.sum(np.sum(np.power(dmat[np.ix_(currinds,currinds)],2)))/Ncluster**2/2.0
    
    left = np.delete(np.arange(len(dmat)),currinds)
    dist = np.sum(np.power(dmat[np.ix_(left,currinds)],2),axis=1)/Ncluster
    
    newdist = np.sqrt(dist-cross)
    
    avedmat = np.zeros((len(left)+1,len(left)+1))
    
    m1 = np.ix_(np.arange(currinds[0]),np.arange(currinds[0]))
#     print(m1)
    m2 = np.ix_(np.arange(currinds[-1]+1,len(dmat)),np.arange(currinds[-1]+1,len(dmat)))
#     print(m2,len(m2[0]))
    mcross = np.ix_(np.arange(currinds[0]),np.arange(currinds[-1]+1,len(dmat)))
    
    avedmat[m1] = dmat[m1]
    avedmat[np.ix_(np.arange(currinds[0]+1,len(avedmat)),np.arange(currinds[0]+1,len(avedmat)))] = dmat[m2]
    avedmat[np.ix_(np.arange(currinds[0]),np.arange(currinds[0]+1,len(avedmat)))] = dmat[mcross]
    avedmat[np.ix_(np.arange(currinds[0]+1,len(avedmat)),np.arange(currinds[0]))] = dmat[mcross].T
    
#     print(avedmat,currinds[0],avedmat[currinds[0],np.arange(currinds[0])],newdist[:len(m1)],len(m1))
    avedmat[currinds[0],np.arange(currinds[0])] = newdist[:len(m1[0])]
    avedmat[np.arange(currinds[0]),currinds[0]] = newdist[:len(m1[0])].T
    
    avedmat[currinds[0],np.arange(currinds[0]+1,len(avedmat))] = newdist[len(m1[0]):]
    avedmat[np.arange(currinds[0]+1,len(avedmat)),currinds[0]] = newdist[len(m1[0]):].T
    return avedmat

'''
functions on averaging dmat using feature matrix
'''
def avefeatmat_dmat(featmat, labels): # dmat and labels must be arranged by clusters
    avefeatmat = []
    labellist = []
    for l in labels:
        if l not in labellist:
            labellist.append(l)
    for l in labellist:
        avefeatmat.append(np.mean(featmat[np.where(labels==l)[0]],axis=0))

    return emb.get_dmat_euc(np.array(avefeatmat))

def embed_clusters(dmat,D):
    # prepare model

    ltz_model = stan.CmdStanModel(stan_file=cpath+'/lorentz2.stan')
    data={'N':dmat.shape[0], 'D':D, 'deltaij':dmat}
    model = ltz_model.optimize(data=data, iter=500000, algorithm='LBFGS', 
                                   # tol_rel_grad=1e2, 
                                   show_console=True, refresh=5000)
    
    hyp_emb = {'euc':model.euc, 'sig':model.sig, 'lambda':model.stan_variable('lambda')}
    emb.process_sim(hyp_emb)
    return hyp_emb

'''
Embed each cluster locally: 
directly relax with mds initialization
'''
# embed mds locally
def embedlocal(center, localmat):
    Ndim = center.shape[0]
    mds = MDS(n_components=Ndim, dissimilarity='precomputed')
    locs = mds.fit_transform(localmat)
    mds_coords = emb.expmap(emb.logmap(center)+locs)
    
    return mds_coords

# init_pos should be in euc

def relax(prevfit, dmat_mutual, dmat_local, init_pos):
    # relax with fixed values

    relax_model = stan.CmdStanModel(stan_file=cpath+'/relax.stan')
    
    Nnn = dmat_mutual.shape[1]
    Nnew = dmat_local.shape[0]
    D = prevfit['euc'].shape[1]
    
    init = {'euc_new':init_pos}

    data = {'Ne':Nnn, 'Nn':Nnew, 'D':D, 'lambda':prevfit['lambda'], 
                   'euc_emb':prevfit['euc'], 'deltaij_mutual':dmat_mutual,'deltaij':dmat_local}
    
    model = relax_model.optimize(data=data, iter=500000, algorithm='LBFGS', inits = init,
                               # tol_rel_grad=1e2, 
                               show_console=False, refresh=5000)
    hyp_emb = {'euc':model.euc_new, 'sig':np.ones((len(model.euc_new),)), 'lambda':prevfit['lambda']}
    emb.process_sim(hyp_emb)
    
    return hyp_emb

'''
distances to the center of mass as reference distances
'''

def get_dmat_to_cm(localmat):
    res = []
    N = len(localmat)
    for i in range(len(localmat)):
        inter_term = np.sum(localmat[i]**2)
        ind_ = np.delete(np.arange(N),i)
        intra_term = np.sum(localmat[np.ix_(ind_,ind_)].flatten()**2)/2.0
        res.append(np.sqrt((N-1)/N**2*inter_term-1.0/N**2*intra_term))
    return res
    
def get_nn_dmat(dmat, avemat, labels, i, N, labellist, mode):
    '''
    dmat: original dmat or feature mat
    avemat: averaged dmat
    labels: cluster inds 
    i: ith cluster as the local cluster
    N: number of neighbor to find
    labellist: labels ordered by the order of occurence in avemat
    '''
    NNs = np.argsort(avemat[i])[1:N+1]
    
    curr_inds = np.where(labels == labellist[i])[0]
    
    res = []
    
#     # distance to the self center of mass
    if mode == 'dmat':
        res.append(get_dmat_to_cm(dmat[np.ix_(curr_inds,curr_inds)]))
    elif mode == 'featmat':
        res.append(np.linalg.norm(dmat[curr_inds]-np.mean(dmat[curr_inds],axis=0),axis=1))
    
    for n in NNs:
        nn_inds = np.where(labels == labellist[n])[0]
        inds = np.concatenate([nn_inds,curr_inds])
        if mode == 'dmat':
            res.append(aveone(dmat[np.ix_(inds,inds)],np.arange(len(nn_inds)))[0,1:])
        elif mode == 'featmat':
            res.append(np.linalg.norm(dmat[curr_inds]-np.mean(dmat[nn_inds],axis=0),axis=1))
    
    return np.array(res).T

"""
directly relax with mds initialization
"""
def LargeScaleEmb(clusterlist_f,dmat,labels,cluster_emb,avemat,Nn,correction=0,
                  verbose=False, mode = 'dmat'):
    alllocs = []
    start_time = time.time()
    for i,l in enumerate(clusterlist_f): # order of clusterlist_f = order in cluster_emb and avemat
        ind_ = np.where(labels==l)[0]
        
        if mode == 'dmat':
            localmat = dmat[np.ix_(ind_,ind_)]
        elif mode == 'featmat':
            localmat = emb.get_dmat_euc(dmat[ind_])

        # mds embedding of local mat as initialization
        
        if len(ind_) <= 1:
            print('n=1 dont need embedding')
            alllocs.append(cluster_emb['pcoords'][i].reshape(1,-1))
        else:

            locpoin = embedlocal(cluster_emb['pcoords'][i], localmat)
            euclocs = emb.poin2euc(locpoin)
        
            # mutual dmat considering if it is featmat or dmat
            dmat_mutual = get_nn_dmat(dmat,avemat,labels,i,Nn,clusterlist_f, mode)
        
            # prevfit with only the NNs
       
            nnind = np.argsort(avemat[i])[0:Nn+1]
            prevfit = {'euc':cluster_emb['euc'][nnind],'lambda':cluster_emb['lambda']}
        
            # relax
            relaxed = relax(prevfit, dmat_mutual, localmat, euclocs)
        
            alllocs.append(relaxed['pcoords'])
        if verbose == True:
            print('%i/%i, %i pts'%(i,len(clusterlist_f),len(ind_)))
            print(time.time()-start_time)
    alllocs = np.concatenate(alllocs)
    
    # inter cluster correction
    if correction == 1:
        # Nn_corr = Nn*2
        print('\n\n\n Performing inter-cluster-correction \n\n\n')

        # ordered from clusters with higher variance

        indlist = [np.where(labels==l)[0] for l in clusterlist_f]
        if mode == 'dmat':
            varlist = [np.var(dmat[np.ix_(ind_, ind_)].flatten()) for ind_ in indlist]
        elif mode == 'featmat':
            varlist = [np.var(emb.get_dmat_euc(dmat[ind_]).flatten()) for ind_ in indlist]
        i_ord = np.argsort(varlist)[::-1]

        for i in i_ord:
            l = clusterlist_f[i]
            ind_ = np.where(labels==l)[0]
            Nn_corr = Nn
            if mode == 'dmat':
                localmat = dmat[np.ix_(ind_,ind_)]
            elif mode == 'featmat':
                localmat = emb.get_dmat_euc(dmat[ind_])

            if len(ind_) > 1:
                others = np.delete(np.arange(len(dmat)),ind_)
                if mode == 'dmat':
                    nns = others[np.argsort(np.mean(dmat[np.ix_(ind_,others)],axis=0))[:Nn_corr]]
                    dmat_mutual = dmat[np.ix_(ind_,nns)]

                elif mode == 'featmat':
                    nns = others[np.argsort(np.mean(np.array([np.linalg.norm(dmat[others]-dmat[ii],axis=1) 
                                      for ii in ind_]),axis=0))[:Nn_corr]]
                    dmat_mutual = np.array([np.linalg.norm(dmat[nns]-dmat[ii],axis=1) 
                                            for ii in ind_])

                Nn_cls_corr = Nn//2
                dmat_cls = get_nn_dmat(dmat,avemat,labels,i,Nn_cls_corr,clusterlist_f, mode)
                nnind = np.argsort(avemat[i])[0:Nn_cls_corr+1]
                
                dmat_mutual = np.concatenate([dmat_mutual,dmat_cls],axis=1)

                # prevfit with only the NNs

                prevfit = {'euc':np.concatenate([emb.poin2euc(alllocs[nns]),cluster_emb['euc'][nnind]],axis=0),
                'lambda':cluster_emb['lambda']}
            
                # relax
                relaxed = relax(prevfit, dmat_mutual, localmat, emb.poin2euc(alllocs[ind_]))
                
                # update
                alllocs[ind_] = relaxed['pcoords']
    print('finished. Computing time: %.2f'%(time.time()-start_time))
    return alllocs


'''
embed previously deleted outliers: use transform.stan to map them into existing space
using the distance matrix of their nearest neighboring points
'''
def transform_new_point(euccoords,mutualmat,curvature):
    
    transform_model = stan.CmdStanModel(stan_file=cpath+'/transform.stan')
    
    Nnn = euccoords.shape[0]
    Nnew = 1
    D = euccoords.shape[1]
    
    init = {'euc_new':np.mean(euccoords,axis=0).reshape(1,-1)}

    data = {'Ne':Nnn, 'Nn':Nnew, 'D':D, 'lambda':curvature, 
                   'euc_emb':euccoords, 'sig_e':np.ones((Nnn,)),
                   'deltaij_mutual':mutualmat,'deltaij':[[0]]}
    
    model = transform_model.optimize(data=data, iter=500000, algorithm='LBFGS', inits = init,
                               # tol_rel_grad=1e2, 
                               show_console=False, refresh=5000)
    hyp_emb = {'euc':model.euc_new, 'sig':model.sig_n, 'lambda':curvature}
    emb.process_sim(hyp_emb)
    
    return hyp_emb['pcoords'].reshape(-1)

