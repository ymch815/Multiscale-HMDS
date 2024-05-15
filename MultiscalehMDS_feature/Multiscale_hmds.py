'''
Perform Multiscale HMDS with feature matrix input
'''
import numpy as np

import sys, socket, time

import pandas as pd


import random
import os

from sklearn.cluster import KMeans

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

def further_split(featmat,maxcls):
    maxsubsize = 1e5
    startncls = len(featmat)//maxcls
    while startncls < 100 and maxsubsize > maxcls:
        newlabel_ = KMeans(n_clusters = startncls).fit(featmat).labels_
        maxsubsize = max([len(np.where(newlabel_==x)[0]) 
                          for x in np.unique(newlabel_)])
        startncls += 5
    return newlabel_

def split_too_large(labels,uniq_labels,lenlist,maxcls,featmat):
    new_labels = labels.copy()
    curr_use = np.max(new_labels)+1 # clusterlist must be continuous interger numbers
    for i,lens in enumerate(lenlist):
        if lens > maxcls:
            print('cluster %i to be further split, with length %i'%(i, lens))
            ind_ = np.where(labels==uniq_labels[i])[0]
            newlabel_ = further_split(featmat[ind_],maxcls)
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

def keep_outliers(featmat, indlist,outlier_nn):
    # for the feature matrix: we keep the outlier feature matrix and 
    # a list of nn's with index in the new featmat
    outlierfeatmat,outlier_nn_inds = [],[]
    # a list of outlier index
    outliers = np.delete(np.arange(len(featmat)),indlist)
    # for each outlier, find its nn, which must be in the indlist
    # keep a list of nn index in the NEW dmat
    for outlier_ in outliers: # index of outliers
        # compute distance to all selected points
        dists = np.linalg.norm(featmat[indlist]-featmat[outlier_],axis=1)
        outlier_nn_inds.append(np.array(np.argsort(dists)[:outlier_nn]))
    
    #
    outlierfeatmat = featmat[outliers]

    indlist = np.concatenate([indlist,outliers])
    return outlierfeatmat,outlier_nn_inds,indlist

# re-order, filter and re-divide the featmat
def preprocess_featmat(featmat,clusterlist,mincls,maxcls,targetfdname,filename,map_outlier_flag,outlier_nn):
    '''
    given dmat and a list of clusters:
        re-divide if too large cluster, 
        filter
        e-order based on layers
    '''
    newfeatmat = None
    
    # print statistics of each layer
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
                                     uniq_labels_0,lenlist,
                                     maxcls,featmat)

    # filter the lowest layer where the number is smaller than mincls
    filt_c = np.where(lenlist>=mincls)[0]
    indlist = np.concatenate([np.where(clusterlist[0]==x)[0] 
                              for x in uniq_labels_0[filt_c]])
    print('original number of clusters/samples %i/%i, \
          filtered to %i/%i by min size %i'
          %(len(lenlist),len(featmat),len(filt_c),len(indlist),mincls,))
    
    # order the samples so that same clusters are together in the same layer
    
    clusterlist = [l[indlist] for l in clusterlist]
    cluster_ord_0 = get_all_ord_uniq_label(clusterlist)
    # re-order 
    reorder_ind = np.concatenate([np.where(clusterlist[0]==x)[0] 
                                      for x in cluster_ord_0])
    indlist = indlist[reorder_ind]
    clusterlist = [l[reorder_ind] for l in clusterlist]
    
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
    newfeatmat = featmat[indlist]
    
    # if we want to keep outliers:
    # for feature matrix ver: only need to keep outlier featmat 
    # and corresponding nn index
    outlierfeatmat,outlier_nn_inds= None, None
    if map_outlier_flag == 1:
        # keep a list of outlier to its nns
        outlierfeatmat,outlier_nn_inds,indlist= keep_outliers(featmat, indlist,outlier_nn)
    
    
    if not os.path.exists(targetfdname):
        os.makedirs(targetfdname)

    np.savetxt(targetfdname+filename+'_inds_after_prep.txt',indlist)

    return newfeatmat,clusterlist, maxunitsize, nunit,outlierfeatmat,outlier_nn_inds

# compute averaging dmats
def averaging_dmats(newfeatmat,newcluster):
    '''
    Compute averaged distance based on each layer
    '''
    time_ = time.time()
    avedmats = [lhmds.avefeatmat_dmat(newfeatmat, newcluster[0])]
    
    norfac = np.max(avedmats[0])/2.0
    
    avedmats[0] = avedmats[0]/norfac
    
    if len(newcluster)>1:
        for ls in newcluster[1:]:
            avedmats.append(lhmds.avefeatmat_dmat(newfeatmat, ls)/norfac)
    norfeatmat = newfeatmat/norfac

    return avedmats, time.time()-time_,norfeatmat,norfac

# embedding by different layers
def layered_embedding(norfeatmat,newcluster,avedmats,Nn,Nd,correction=0,verbose=False,metricflag=0):

    embtimes = []
    embmats = []
    embcoords = []

    # first layer: hyperbolic embedding
    time_ = time.time()
    hyp_emb = lhmds.embed_clusters(avedmats[-1],D=Nd)
    embtimes.append(time.time() - time_)
    if metricflag == 1:
        embmats.append(hyp_emb['emb_mat'])
    embcoords.append(hyp_emb['pcoords'])
    
    # following layers
    for i in range(1,len(Nn)):
        time_ = time.time()
        
        alllocs = lhmds.LargeScaleEmb(list(dict.fromkeys(newcluster[-i])),
                        avedmats[-i-1],
                        get_sublabels(newcluster[-i],newcluster[-i-1]),
                        hyp_emb,avedmats[-i],Nn=Nn[-i],correction=correction,verbose=verbose,mode='dmat')
        hyp_emb = {'pcoords':alllocs,'euc':emb.poin2euc(alllocs),
                     'lambda':hyp_emb['lambda']}
        embtimes.append(time.time() - time_)
        if metricflag == 1:
            embmats.append(emb.get_dmat_poin(alllocs)/hyp_emb['lambda'])
        embcoords.append(alllocs)
        
    # last layer
    time_ = time.time()
    alllocs = lhmds.LargeScaleEmb(list(dict.fromkeys(newcluster[0])),
                                  norfeatmat,newcluster[0],
                                  hyp_emb,avedmats[0],Nn=Nn[0],correction=correction,verbose=verbose,mode='featmat')
    embtimes.append(time.time() - time_)
    if metricflag == 1:
        embmats.append(emb.get_dmat_poin(alllocs)/hyp_emb['lambda'])
    embcoords.append(alllocs)
    
    return embcoords,embmats, hyp_emb['lambda'], embtimes

def map_outlier(embcoord,norfeatmat,outlierfeatnor,outlier_nn_inds,curvature,embmat,metricflag=0):
    time_ = time.time()
    outliercoords = []
    if len(outlier_nn_inds) == 0:
        return None, None, 0.0
    for i,ind in enumerate(outlier_nn_inds):
        mutualmat = np.linalg.norm(norfeatmat[ind]-outlierfeatnor[i],axis=1)
        pcoords = embcoord[ind]
        euccoords = emb.poin2euc(pcoords)
        map_pcoord = lhmds.transform_new_point(euccoords,mutualmat.reshape(1,-1),curvature)
        outliercoords.append(map_pcoord)
    # compute distance matrix
    outliercoords = np.array(outliercoords)
    mat_self = emb.get_dmat_poin(outliercoords)/curvature
    mat_mutual = emb.get_dmat_poin_mutual(embcoord,outliercoords)/curvature

    fullembmat = None
    if metricflag == 1:
        fullembmat = np.concatenate([np.concatenate([embmat[-1],mat_mutual],axis=1),
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
        corrs.append(stats.pearsonr(embmats[i][np.triu_indices(N_sample, k=1)],
                       avedmats[-i-1][np.triu_indices(N_sample, k=1)])[0])
        Ql, Qg,_ = qms.get_quality_metrics(avedmats[-i-1],
                                           embmats[i],verbose=False)
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
    return metrics
    
    
    
def saveembmats(embcoords,fullcoords,targetfdname,clusterlistname):
    if not os.path.exists(targetfdname):
        os.makedirs(targetfdname)
    for i in range(len(embcoords)):
        np.savetxt(targetfdname+clusterlistname+'_coords_%i.txt'%(len(embcoords[i]))
                   ,embcoords[i])
    if fullcoords is not None:
        np.savetxt(targetfdname+clusterlistname+'_coords_%i.txt'%(len(fullcoords))
                   ,fullcoords)


def get_euc_mutual(coords1, coords2):
    return np.array([np.linalg.norm(coords1-c_,axis=1) for c_ in coords2]).T
########################################################



if __name__ == "__main__":
   t = time.time() # Count the time of running
   
   #first argument is python filename
   print("the name of the program:", sys.argv[0])
   print("----------------------------------------------------\n")

   if len(sys.argv) < (1+8):
      print('  [Error] No input argument provided. Program ended!\n')
      exit()

   #set global parameters and load distance matrix
   
   featmatname = str(sys.argv[1].split('.')[0])
   clusterlistname = str(sys.argv[2])
   datafdname = '../data/%s/'%(featmatname)
   targetfdname = './test/%s/'%(featmatname)

   if featmatname.split('_')[0].lower() == 'featmat':
        featmat = np.loadtxt(datafdname+str(sys.argv[1]),dtype = float)
   else: 
       print('  [Error] Must provide feature matrix. Program ended!\n')
       exit()
   clusterlist  = np.loadtxt(datafdname+str(sys.argv[2]),dtype = int).reshape(-1,len(featmat))
   Nn = [int(n) for n in str(sys.argv[3]).split(',')] # number of neighbors
   Nd = int(sys.argv[4]) # number of dimensions
   mincls = int(sys.argv[5]) # minimum cluster size for filtering 
   maxcls = int(sys.argv[6]) # maximum cluster size for re-dividing
   savematflag = int(sys.argv[7]) # if savine embed mat or not
   
   
   '''
   if compute metrics or not 
   this need to compute full distance matrix for both original feat mat 
   and embeded feat mat, therefore would take long when the dataset 
   is large (>10,000 samples)
   '''
   comp_metr_flag = int(sys.argv[8]) 
   
   if len(featmat)>5000 and comp_metr_flag == 1:
       print("Warning: Computing metrics might take long for large datasets with n = %i"%len(featmat))
   
   '''
   map outlier or not
   '''
   map_outlier_flag = int(sys.argv[9])
   outlier_nn = int(sys.argv[10])

   if mincls == 1:
        map_outlier_flag = 0
   
   
   # additional argument on inter-layer correction, default 0
   correction = int(sys.argv[11])
   
   
   filename = clusterlistname+'_Nn_'+str(sys.argv[3])+'_Nd_'+sys.argv[4]+'_min_'+sys.argv[5]+'_max_'+sys.argv[6]+'_corr_'+sys.argv[11]
   
   # re-order, filter and re-divide the clusters
   newfeatmat,newcluster,maxunitsize, nunit,outlierfeatmat,outlier_nn_inds = preprocess_featmat(
       featmat,clusterlist,mincls,maxcls,targetfdname,filename,map_outlier_flag,outlier_nn)

    # normalize the featmat after those steps and get normalized avemats and normalized featmat
   avedmats, avetime, norfeatmat, norfac = averaging_dmats(newfeatmat,newcluster)


    # embedding by different layers
   embcoords,embmats, curvature, embtimes = layered_embedding(norfeatmat,
                newcluster,avedmats,Nn,Nd,correction=correction,verbose=False,metricflag=comp_metr_flag)
   
   fullcoords,fullembmat,maptime = None, None, 0.0
   # # map outlier or not
   if map_outlier_flag == 1:
       fullcoords, fullembmat,maptime = map_outlier(embcoords[-1],norfeatmat,
        outlierfeatmat/norfac,outlier_nn_inds,curvature,embmats,metricflag=comp_metr_flag)

   
   # # compute and save metrics
   Qlocals = np.ones((len(clusterlist)+2,))*np.nan
   Qglobals = np.ones((len(clusterlist)+2,))*np.nan
   corrs = np.ones((len(clusterlist)+2,))*np.nan
   if comp_metr_flag == 1:
        newdmat = emb.get_dmat_euc(norfeatmat)
        mutualdmat,selfdmat = None, None 
        if fullembmat is not None:
            mutualdmat = get_euc_mutual(norfeatmat,outlierfeatmat/norfac)
            selfdmat = emb.get_dmat_euc(outlierfeatmat/norfac)
        Qlocals, Qglobals, corrs = compute_metric(embmats, avedmats, newdmat, 
                                                  fullembmat, mutualdmat, selfdmat)
   
    # save results
   metrics = savemetrics(Qlocals, Qglobals, corrs, embtimes, avetime,maptime,curvature, 
                maxunitsize, nunit, 
                targetfdname,filename)

   if savematflag == 1:
        saveembmats(embcoords,fullcoords,targetfdname,filename)

   elapsed = time.time() - t
   print('\nRun time: %f\n' % elapsed)



