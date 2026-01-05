'''
Perform Multiscale HMDS with distance matrix input
'''
import numpy as np

import sys, socket, time
import argparse

import pandas as pd

import random
import os

from sklearn.cluster import AgglomerativeClustering

from scipy import stats

# load from local package
sys.path.append('../')
import MHMDS.embed_funs as emb
import MHMDS.quality_metrics as qms
import MHMDS.large_scale_hmds as lhmds

print(socket.gethostname())

random.seed()

'''
pipeline: 
1. re-order, filter and re-divide the dataset
2. compute averaging clusters
3. embedding by layers (save positions for each layer)
4. compute and save metrics 
step | Qlocal | Qglobal | corr | runningtime

'''

def get_sublabels(labels, sublabels):
    '''
    given labels to the full group, return labels of "clusters" in 
    intermediate layer to the previous layer
    '''
    res = []
    visit = []
    for i, l in enumerate(labels):
        if sublabels[i] not in visit:
            visit.append(sublabels[i])
            res.append(l)
    return np.array(res)

def get_ord_uniq_sublable(labels,sublabels,labelords):
    '''
    given labels and sublabels, return list of unique labels of how the second labels 
    are included in the first labels
    '''
    res = []
    for i,l in enumerate(labelords):
        res.append(np.unique(sublabels[np.where(labels == l)[0]]))
    return np.concatenate(res)

def get_all_ord_uniq_label(layerlist):
    # ordered by the last layer
    ords = np.unique(layerlist[-1])
    for i in range(len(layerlist)-1):
        temp = get_ord_uniq_sublable(layerlist[-i-1],layerlist[-i-2],ords)
        ords = temp
    return ords

def further_split(dmat, startres,maxcls):
    maxsubsize = 1e5
    while startres > 0.05 and maxsubsize > maxcls:
        newlabel_ = AgglomerativeClustering(n_clusters = None, 
                            metric = 'precomputed',linkage = 'complete',
                                     distance_threshold=startres).fit(dmat).labels_
        maxsubsize = max([len(np.where(newlabel_==x)[0]) 
                          for x in np.unique(newlabel_)])
        startres -= 0.01
    return newlabel_

def split_too_large(labels,uniq_labels,lenlist,minres,maxcls,dmat):
    new_labels = labels.copy()
    curr_use = np.max(new_labels)+1 # clusterlist must be continuous interger numbers
    for i,lens in enumerate(lenlist):
        if lens > maxcls:
            print('cluster %i to be further split, with length %i'%(i, lens))
            ind_ = np.where(labels==uniq_labels[i])[0]
            newlabel_ = further_split(dmat[np.ix_(ind_,ind_)],minres,maxcls)
            for j,ls in enumerate(np.unique(newlabel_)):
                if j != 0:
                    new_labels[ind_[np.where(newlabel_== ls)[0]]] = curr_use
                    curr_use += 1
            print('after splitting, becomed %i clusters with max size %i'
                  %(len(np.unique(newlabel_)),
                    max([len(np.where(newlabel_==x)[0]) 
                     for x in np.unique(newlabel_)])))
    new_uniq_label = np.unique(new_labels) 
    lenlist = np.array([len(np.where(new_labels==x)[0]) 
                    for x in new_uniq_label])
    return new_labels, new_uniq_label, lenlist

def keep_outliers(dmat_norm, indlist,outlier_nn):
    outliermats,outlier_nn_inds = [],[]
    outliers = np.delete(np.arange(len(dmat_norm)),indlist)
    
    for outlier_ in outliers:
        temp = np.argsort(dmat_norm[outlier_])[1:]
        count = 0
        keep = []
        mutualmat = []
        for ind_ in temp:
            if ind_ in indlist:
                count += 1
                keep.append(np.where(indlist==ind_)[0][0])
                mutualmat.append(dmat_norm[outlier_][ind_])
            if count == outlier_nn:
                break
        
        outliermats.append(np.array(mutualmat))
        outlier_nn_inds.append(np.array(keep))
    
    selfdmat = dmat_norm[np.ix_(outliers,outliers)]
    mutualdmat = dmat_norm[np.ix_(indlist,outliers)]
    indlist = np.concatenate([indlist,outliers])
    return outliermats,outlier_nn_inds,indlist,mutualdmat, selfdmat

# re-order, filter and re-divide the dmat
def preprocess_dmat(dmat,clusterlist,mincls,maxcls,minres,targetfdname,filename,map_outlier_flag,outlier_nn):
    """
    Preprocess distance matrix by re-dividing large clusters, filtering outliers, and reordering.
    
    Operations:
    - Re-divide clusters exceeding maximum size
    - Filter clusters below minimum size
    - Reorder samples by hierarchical cluster membership
    """
    newdmat = None
    
    dmat_norm = dmat/np.max(dmat)*2.0
    uniq_labels_0 = np.unique(clusterlist[0])
    lenlist = np.array([len(np.where(clusterlist[0]==x)[0]) 
                    for x in uniq_labels_0])
    print('layer 0:maximum cluster size %i, number of clusters %i'
          %(max(lenlist),len(lenlist)))
    for i in range(len(clusterlist)-1):
        subl_ = get_sublabels(clusterlist[i+1],clusterlist[i])
        lenlist_ = np.array([len(np.where(subl_==x)[0]) for x in np.unique(subl_)])
        print('layer %i:maximum cluster size: %i, number of clusters %i'
          %(i+1,max(lenlist_),len(lenlist_)))
    
    # divide clusters in the lowest layer if they are larger than maxcls
    clusterlist[0],uniq_labels_0, lenlist = split_too_large(clusterlist[0],
                                     uniq_labels_0,lenlist,minres,
                                     maxcls,dmat_norm)

    # filter the lowest layer where the number is smaller than mincls
    filt_c = np.where(lenlist>=mincls)[0]
    indlist = np.concatenate([np.where(clusterlist[0]==x)[0] 
                              for x in uniq_labels_0[filt_c]])
    print('original number of clusters/samples %i/%i, \
          filtered to %i/%i by min size %i'
          %(len(lenlist),len(dmat),len(filt_c),len(indlist),mincls,))
    
    # order the samples so that same clusters are together in the same layer
    
    clusterlist = [l[indlist] for l in clusterlist]
    cluster_ord_0 = get_all_ord_uniq_label(clusterlist)
    # re-order 
    reorder_ind = np.concatenate([np.where(clusterlist[0]==x)[0] 
                                      for x in cluster_ord_0])
    indlist = indlist[reorder_ind]
    clusterlist = [l[reorder_ind] for l in clusterlist]
    
    newdmat = dmat_norm[np.ix_(indlist,indlist)]
    newdmat = newdmat/np.max(newdmat)*2.0
    
    # statistics from lowest layer:
    lenlist = np.array([len(np.where(clusterlist[0]==x)[0]) 
                    for x in np.unique(clusterlist[0])])

    maxunitsize = [np.max(lenlist)] # maximum unit size in each intermediate embedding
    # not including initial embedding of points(clusters) in hyperbolic space
    nunit = [len(lenlist)] # number of units in each intermediate embedding
    # note that nunit_step2  = nunit_step1

    for i in range(len(clusterlist)-1):
        subl_ = get_sublabels(clusterlist[i+1],clusterlist[i])
        lenlist_ = np.array([len(np.where(subl_==x)[0]) for x in np.unique(subl_)])
        maxunitsize.append(max(lenlist_))
        nunit.append(len(lenlist_))

    outliermats,outlier_nn_inds,mutualdmat, selfdmat = None, None, None, None
    if map_outlier_flag == 1:
        outliermats,outlier_nn_inds,indlist,mutualdmat, selfdmat= keep_outliers(dmat_norm, indlist,outlier_nn)
    
    if not os.path.exists(targetfdname):
        os.makedirs(targetfdname)

    np.savetxt(targetfdname+filename+'_inds_after_prep.txt',indlist)

    return newdmat,clusterlist,maxunitsize,nunit,outliermats,outlier_nn_inds, mutualdmat, selfdmat

# compute averaging dmats
def averaging_dmats(newdmat,newcluster):
    '''
    Compute averaged distance based on each layer
    '''
    time_ = time.time()
    avedmats = []
    for ls in newcluster:
        avedmats.append(lhmds.avedmat(newdmat,ls))
    
    return avedmats, time.time()-time_

# embedding by different layers
def layered_embedding(newdmat,newcluster,avedmats,Nn,Nd,verbose=False):

    embtimes = []
    embmats = []
    embcoords = []
    # first layer: hyperbolic embedding
    time_ = time.time()
    hyp_emb = lhmds.embed_clusters(avedmats[-1],D=Nd)
    embtimes.append(time.time() - time_)
    embmats.append(hyp_emb['emb_mat'])
    embcoords.append(hyp_emb['pcoords'])
    
    # following layers
    for i in range(1,len(Nn)):
        time_ = time.time()
        alllocs = lhmds.LargeScaleEmb(list(dict.fromkeys(newcluster[-i])),
                        avedmats[-i-1],
                        get_sublabels(newcluster[-i],newcluster[-i-1]),
                        hyp_emb,avedmats[-i],Nn=Nn[-i],verbose=verbose)
        hyp_emb = {'pcoords':alllocs,'euc':emb.poin2euc(alllocs),
                     'lambda':hyp_emb['lambda']}
        embtimes.append(time.time() - time_)
        embmats.append(emb.get_dmat_poin(alllocs)/hyp_emb['lambda'])
        embcoords.append(alllocs)
        
    # last layer
    time_ = time.time()
    alllocs = lhmds.LargeScaleEmb(list(dict.fromkeys(newcluster[0])),
                                  newdmat,newcluster[0],
                                  hyp_emb,avedmats[0],Nn=Nn[0],verbose=verbose)
    embtimes.append(time.time() - time_)
    embmats.append(emb.get_dmat_poin(alllocs)/hyp_emb['lambda'])
    embcoords.append(alllocs)
    
    return embcoords,embmats, hyp_emb['lambda'], embtimes

def map_outlier(embcoord,outliermats,outlier_nn_inds,curvature,embmat):
    time_ = time.time()
    outliercoords = []
    for i,ind in enumerate(outlier_nn_inds):
        mutualmat = outliermats[i]
        pcoords = embcoord[ind]
        euccoords = emb.poin2euc(pcoords)
        map_pcoord = lhmds.transform_new_point(euccoords,mutualmat.reshape(1,-1),curvature)
        outliercoords.append(map_pcoord)
    # compute distance matrix
    outliercoords = np.array(outliercoords)
    mat_self = emb.get_dmat_poin(outliercoords)/curvature
    mat_mutual = emb.get_dmat_poin_mutual(embcoord,outliercoords)/curvature
    fullembmat = np.concatenate([np.concatenate([embmat,mat_mutual],axis=1),
                                 np.concatenate([mat_mutual.T,mat_self],axis=1)],axis=0)
    return np.concatenate([embcoord,outliercoords],axis=0), fullembmat, time.time()-time_

# compute and save metrics
def compute_metric(embmats, avedmats, newdmat, fullembmat, mutualdmat, selfdmat):
    # note: avedmats are arranged from low levels to high levels
    # while embdmats are arranged from high to low
    # for this mat we arrange from high to low
    # fullembmat, newinds: for computing with outlier mapped
    corrs = []
    Qlocals = []
    Qglobals = []

    for i in range(len(avedmats)):
        N_sample = len(avedmats[-i-1])
        corr,Ql,Qg = np.nan,np.nan,np.nan
        if np.sum(pd.isna(avedmats)[-i-1])==0:
            corr = stats.pearsonr(embmats[i][np.triu_indices(N_sample, k=1)],
                       avedmats[-i-1][np.triu_indices(N_sample, k=1)])[0]
            
            Ql, Qg,_ = qms.get_quality_metrics(avedmats[-i-1],
                                           embmats[i],verbose=False)
        corrs.append(corr)
        Qlocals.append(Ql)
        Qglobals.append(Qg)
    N_sample = len(newdmat)
    corrs.append(stats.pearsonr(embmats[-1][np.triu_indices(N_sample, k=1)],
                                newdmat[np.triu_indices(N_sample, k=1)])[0])
    Ql, Qg,_ = qms.get_quality_metrics(newdmat,
                                       embmats[-1],verbose=False)
    Qlocals.append(Ql)
    Qglobals.append(Qg)
    
    if fullembmat is not None:
        fulldmat = np.concatenate([np.concatenate([newdmat,mutualdmat],axis=1),
                    np.concatenate([mutualdmat.T,selfdmat],axis=1)],axis=0)
        N_sample = len(fulldmat)
        corrs.append(stats.pearsonr(fullembmat[np.triu_indices(N_sample, k=1)],
                                    fulldmat[np.triu_indices(N_sample, k=1)])[0])
        Ql, Qg,_ = qms.get_quality_metrics(fulldmat,
                                           fullembmat,verbose=False)
        Qlocals.append(Ql)
        Qglobals.append(Qg)
    else:
        corrs.append(np.nan)
        Qlocals.append(np.nan)
        Qglobals.append(np.nan)
    return Qlocals, Qglobals, corrs

# save results
def savemetrics(Qlocals, Qglobals, corrs, embtimes, avetime, maptime,curvature,
                maxunitsize, nunit,
                targetfdname,clusterlistname):
    maxunitsize = np.array(maxunitsize)
    nunit = np.array(nunit)
    metrics = pd.DataFrame({'Step':np.arange(1,len(corrs)+1),'Qlocal':Qlocals,
                  'Qglobal':Qglobals,'corr':corrs,
                  'runningtime':embtimes+[maptime],'curvature':np.ones_like(corrs)*curvature,
                  'nunit':np.concatenate([[nunit[-1]],nunit[::-1],[np.nan]]),
                  'maxsizeofunit':np.concatenate([[1],maxunitsize[::-1],[np.nan]])})
    metrics = pd.concat([metrics,pd.DataFrame({'Step':'Avedmat',
                            'Qlocal':np.nan,'Qglobal':np.nan,'corr':np.nan,
                            'runningtime':avetime,'curvature':np.nan,
                            'nunit':np.nan,'maxsizeofunit':np.nan},index=[0])],
                        axis=0,ignore_index=True)

    metrics.loc[len(corrs)-1,'Step'] = 'MapOutlier'
    
    if not os.path.exists(targetfdname):
        os.makedirs(targetfdname)
    metrics.to_csv(targetfdname+clusterlistname+'_metrics.csv')
    
    
    
def saveembmats(embmats,fullcoords,targetfdname,clusterlistname):
    if not os.path.exists(targetfdname):
        os.makedirs(targetfdname)
    for i in range(len(embmats)):
        np.savetxt(targetfdname+clusterlistname+'_coords_%i.txt'%(len(embmats[i]))
                   ,embmats[i])
    if fullcoords is not None:
        np.savetxt(targetfdname+clusterlistname+'_coords_%i.txt'%(len(fullcoords))
               ,fullcoords)


########################################################

if __name__ == "__main__":
    t = time.time()
    
    parser = argparse.ArgumentParser(
        description='Perform Multiscale Hyperbolic MDS with distance matrix input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python Multiscale_hmds.py --distmat DistMat_ToggleSwitch.txt --clusters cls_0.2_0.6 \
         --neighbors 20,10 --dimension 3 --min-cluster-size 4 --max-cluster-size 40 \
         --save-matrix --compute-metrics --map-outlier --outlier-neighbors 10
        '''
    )
    
    parser.add_argument('--distmat', type=str, required=True,
                        help='Distance/feature matrix filename (e.g., DistMat_ToggleSwitch.txt)')
    parser.add_argument('--clusters', type=str, required=True,
                        help='Cluster assignment filename (e.g., cls_0.2_0.6)')
    parser.add_argument('--neighbors', type=str, required=True,
                        help='Number of neighbors for each layer, comma-separated (e.g., 20,10)')
    parser.add_argument('--dimension', type=int, required=True,
                        help='Number of embedding dimensions (e.g., 2 or 3)')
    parser.add_argument('--min-cluster-size', type=int, required=True,
                        help='Minimum cluster size for filtering outliers')
    parser.add_argument('--max-cluster-size', type=int, required=True,
                        help='Maximum cluster size before re-dividing')
    parser.add_argument('--save-matrix', action='store_true',
                        help='Save embedding coordinate matrices')
    parser.add_argument('--compute-metrics', action='store_true',
                        help='Compute quality metrics (may be slow for large datasets)')
    parser.add_argument('--map-outlier', action='store_true',
                        help='Map outliers to embedding space')
    parser.add_argument('--outlier-neighbors', type=int, required=True,
                        help='Number of neighbors for outlier mapping')
    
    args = parser.parse_args()
    
    dmatname = str(args.distmat.split('.')[0])
    clusterlistname = str(args.clusters)
    datafdname = '../data/%s/'%(dmatname)
    targetfdname = './test/%s/'%(dmatname)
    
    if dmatname.split('_')[0].lower() == 'featmat':
         featmat = np.loadtxt(datafdname+args.distmat, dtype=float)
         dmat = emb.get_dmat_euc(featmat)
    elif dmatname.split('_')[0].lower() == 'distmat':
         dmat  = np.loadtxt(datafdname+args.distmat, dtype=float)
    
    clusterlist  = np.loadtxt(datafdname+args.clusters, dtype=int).reshape(-1,len(dmat))
    Nn = [int(n) for n in args.neighbors.split(',')]
    Nd = args.dimension
    mincls = args.min_cluster_size
    maxcls = args.max_cluster_size
    savematflag = 1 if args.save_matrix else 0
    
    '''
    if compute metrics or not 
    this need to compute full distance matrix for both original feat mat 
    and embeded feat mat, therefore would take long when the dataset 
    is large (>10,000 samples)
    '''
    comp_metr_flag = 1 if args.compute_metrics else 0
    if len(dmat)>5000 and comp_metr_flag == 1:
        print("Warning: Computing metrics might take long for large datasets with n = %i"%len(dmat))
    
    
    '''
    map outlier or not
    '''
    map_outlier_flag = 1 if args.map_outlier else 0
    outlier_nn = args.outlier_neighbors

    if mincls == 1:
        map_outlier_flag = 0
    
    
    minres = float(args.clusters.split('_')[1])
    
    filename = clusterlistname+'_Nn_'+args.neighbors+'_Nd_'+str(args.dimension)+'_min_'+str(args.min_cluster_size)+'_max_'+str(args.max_cluster_size)
    
    # re-order, filter and re-divide the clusters, normalize the dmat after those steps
    '''
    if we wanna map outliers: we need to keep a list of outlier distances to its neighbors
    the index of neighbors in the new distance matrix
    the saved index list should be concat(newdmat + outlierlist)
    '''
    
    newdmat,newcluster,maxunitsize,nunit,outliermats,outlier_nn_inds,mutualdmat, selfdmat = preprocess_dmat(dmat,clusterlist,mincls,maxcls,minres,targetfdname,filename,map_outlier_flag,outlier_nn)

    # compute averaging dmats
    avedmats, avetime = averaging_dmats(newdmat,newcluster)

    # # embedding by different layers
    embcoords, embmats, curvature, embtimes = layered_embedding(newdmat,
                newcluster,avedmats,Nn,Nd,verbose=False)
    
    fullcoords, fullembmat, maptime = None, None, 0.0
    # map outlier or not
    if map_outlier_flag == 1:
        fullcoords, fullembmat, maptime = map_outlier(embcoords[-1],outliermats,outlier_nn_inds,curvature,embmats[-1])
    if savematflag == 1:
        saveembmats(embcoords,fullcoords,targetfdname,filename)

    # compute and save metrics
    Qlocals = np.ones((len(clusterlist)+2,))*np.nan
    Qglobals = np.ones((len(clusterlist)+2,))*np.nan
    corrs = np.ones((len(clusterlist)+2,))*np.nan
    '''
    if we wanna map outliers:
        we need to also compute Q's based on mapped outliers
    '''
    if comp_metr_flag == 1:
         Qlocals, Qglobals, corrs = compute_metric(embmats, avedmats, newdmat, fullembmat,mutualdmat, selfdmat)
    # save results
    '''
    if outliers mapped:
     when saving metrics: one more line about mapping outliers
     when saving mats: one more mat with outliers
    '''
    savemetrics(Qlocals, Qglobals, corrs, embtimes, avetime,maptime,curvature, 
                maxunitsize,nunit,
                targetfdname,filename)
    


    elapsed = time.time() - t
    print('\nRun time: %f\n' % elapsed)



