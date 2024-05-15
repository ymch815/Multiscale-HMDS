#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# functions for embedding and visulization

import numpy as np
import cmdstanpy as stan

import os
cpath =  os.path.dirname(__file__)

def logmap(vecarr):
    vec_r =  np.sqrt(np.sum(np.square(vecarr.T),axis=0))
    return (vecarr.T*(np.arctanh(vec_r)/vec_r)).T
def expmap(vecarr):
    vec_r =  np.sqrt(np.sum(np.square(vecarr.T),axis=0))
    return (vecarr.T*(np.tanh(vec_r)/vec_r)).T 

# return euc distance matrix between high-dimension data
def get_dmat_euc(coords):
    N = coords.shape[0]
    dists = np.zeros((N,N))
    for i in np.arange(N):
         # if i%100 == 0: print(i)
         dists[i] = np.linalg.norm(coords - coords[i],axis=1)
    return dists

# return distance matrix between data in poincare coordinates

def get_dmat_poin(coords):
    N = coords.shape[0]
    dists = np.zeros((N,N))
    norms = np.linalg.norm(coords,axis=1)**2
    for i in np.arange(N):
         # if i%100 == 0: print(i)
         diff = np.linalg.norm(coords - coords[i],axis=1)**2
         dists[i] = np.arccosh(2.0*(diff/(1.0-norms[i])/(1-norms))+1.0)
    return dists

def get_dmat_poin_mutual(coords1,coords2):
    N1 = coords1.shape[0]
    N2 = coords2.shape[0]
    dists = np.zeros((N1,N2))
    norms1 = np.linalg.norm(coords1,axis=1)**2
    norms2 = np.linalg.norm(coords2,axis=1)**2
    for i in np.arange(N1):
         # if i%100 == 0: print(i)
         diff = np.linalg.norm(coords2 - coords1[i],axis=1)**2
         dists[i] = np.arccosh(2.0*(diff/(1.0-norms1[i])/(1-norms2))+1.0)
    return dists

def d_lor(t1, t2, E1, E2):
    temp = t1*t2 - np.dot(E1, E2)
    if temp > 1:
        return np.arccosh(t1*t2 - np.dot(E1, E2))
    else: return 0

#returns embedding distance matrix from optimization fit
def get_embed_dmat(fit):
    N = fit['euc'].shape[0]
    fit_ts = np.sqrt(1.0 + np.sum(np.square(fit['euc']), axis=1))

    fit_mat = np.zeros((N, N))

    for i in np.arange(N):
        for j in np.arange(i+1,N):
            fit_mat[i][j] = d_lor(fit_ts[i], fit_ts[j], fit['euc'][i], fit['euc'][j])
            fit_mat[j][i] = fit_mat[i][j]
            
    return fit_mat

#return poincare coordinates
def get_poin(fit):
    ts = np.sqrt(1.0 + np.sum(np.square(fit['euc']), axis=1))
    return (fit['euc'].T / (ts + 1)).T
def process_sim(fit):
    # fit['emb_mat'] = get_embed_dmat(fit)/6.6
    fit['emb_mat'] = get_embed_dmat(fit)/fit['lambda']
    fit['pcoords'] = get_poin(fit)
    fit['radii'] = 2.0*np.arctanh(np.sqrt(np.sum(np.square(fit['pcoords']), axis=1)))
  
def poin2euc(pcoords):
	norm2 = np.sum(np.square(pcoords),axis=1)
	return (2.0*pcoords.T/(1.0-norm2)).T

# Code for recentering
# recenter

#translation of x so origin is translated to v
#thus -v is translated to origin, so put in -v if you want v to be the new origin
def trans_poin(v, x):
    dp = v.dot(x)
    v2 = v.dot(v)
    x2 = x.dot(x)
    
    return ((1.0 + 2.0*dp + x2)*v + (1.0 - v2)*x) / (1.0 + 2.0*dp + x2*v2)

#given center of mass fit of new center, return poincare coords of fit points translated so new center is at origin
def re_center(fit, CM_fit):
    p_coords = fit['pcoords']
    CM_poin = CM_fit['CM']/(1.0 + CM_fit['CM_t'])
    
    return np.asarray([trans_poin(-CM_poin, pt) for pt in p_coords])

# convert poincare coordinates to native coordinate
def to_native(coords):
    rs = np.linalg.norm(coords,axis=1)
    native_rs = 2.0*np.arctanh(rs)
    coords_new = (coords.T*native_rs).T
    return coords_new

# perform recentering given poincare coordinates
def perform_recenter(pcoords,center_inds):
    CM_m = stan.CmdStanModel(stan_file=cpath+'/CM2.stan')
    eucs = poin2euc(pcoords)
    
    
    # convert pcoords to euc for recentering purpose
    ex_data = {'N':center_inds.shape[0], 'D':pcoords.shape[1], 
               'coords':eucs[center_inds]}
    
    
    cm_fit = CM_m.optimize(data=ex_data)
    cm_fit = {'CM':cm_fit.CM, 'CM_t':cm_fit.CM_t}
    hyp_emb = {'euc':eucs,'pcoords':pcoords}
    data_recenter = re_center(hyp_emb,cm_fit)
    return data_recenter