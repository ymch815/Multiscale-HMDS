#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for computing mettics 

quality score code from the following paper: Klimovskaia et al.
    https://github.com/facebookresearch/PoincareMaps 
"""


import numpy as np
import pandas as pd


def get_ranking(distance_matrix):
    """
    Get ranking from distance matrix: from Supplementary eq. (2)-(3) 
    in Klimovskaia et al.
    """
    # According to this definition, reflexive ranks are set
    # to zero and non-reflexive ranks belong to {1,.., N − 1}.
    n = len(distance_matrix)
    Rank = np.zeros([n, n])
    for i in range(n):
        idx = np.array(list(range(n)))        
        sidx = np.argsort(distance_matrix[i, :])
        Rank[i, idx[sidx][1:]] = idx[1:]

    return Rank


def get_coRanking(Rank_high, Rank_low):
    """
    Computes co-ranking matrix Q from Supplementary eq. (4) in Klimovskaia et al.
    """
    N = len(Rank_high)
    coRank = np.zeros([N-1, N-1])

    for i in range(N):
        for j in range(N):
            k = int(Rank_high[i, j])
            l = int(Rank_low[i, j])
            if (k > 0) and (l > 0):
                coRank[k-1][l-1] += 1
    
    return coRank


def get_score(Rank_high, Rank_low, fname=None):     
    """
    Computes Qnx scores from Supplementary eq. (5) in Klimovskaia et al.
    """
    coRank = get_coRanking(Rank_high, Rank_low)
    N = len(coRank)+1

    df_score = pd.DataFrame(columns=['Qnx', 'Bnx'])
    Qnx = 0
    Bnx = 0
    for K in range(1, N):
        Qnx += sum(coRank[:K, K-1]) + sum(coRank[K-1, :K]) - coRank[K-1, K-1]
        Bnx += sum(coRank[:K, K-1]) - sum(coRank[K-1, :K])
        df_score.loc[len(df_score)] = [Qnx /(K*N), Bnx/(K*N)]

    if not (fname is None):
        df_score.to_csv(fname, sep = ',', index=False)
    
    return df_score


def get_scalars(Qnx):
    """
    Computes scalar scores from Supplementary eq. (6)-(8) in Klimovskaia et al.
    """
    N = len(Qnx) # total length of Qnx is smaller than number of samples
    K_max = 0
    val_max = Qnx[0] - 1/N
    for k in range(1, N):
        if val_max < (Qnx[k] - (k+1)/N):
            val_max = Qnx[k] - (k+1)/N
            K_max = k

    Qlocal = np.mean(Qnx[:K_max+1])
    Qglobal = np.mean(Qnx[K_max:])

    return Qlocal, Qglobal, K_max


def get_quality_metrics(    
    D_high,D_low,
    fname=None,
    k_neighbours=20,
    verbose=False):
    """
    distance : str (default: 'euclidean')
        Distance metric to compute distanced between points in low dimendional 
        space. Possible parameters: 'euclidean' or 'poincare'.
    setting: str (default: 'manifold')
        Setting to compute distances in the high dimensional space: 'global'
        distances or distances on the 'manifold' using a k=20 KNN graph.
    fname: str, optional (default: None)
        Name of the file where to save all the information about the metrics.
    verbose: bool (default: False)
        A flag if to print the results of the computations.    
    k_neighbours: int (default: 20)
        k-nearest neighbours for setting
    Returns
    -------
    Qlocal: float
        Quality criteria for local qualities of the embedding.
        Range from 0 (bad) to 1 (good).
    Qglobal: float
        Quality criteria for global qualities of the embedding.
        Range from 0 (bad) to 1 (good).
    Kmax: int
        Kmax defines the split of the QNX curv.
    """

    Rank_high = get_ranking(D_high)

    Rank_low = get_ranking(D_low)
    df_score = get_score(Rank_high, Rank_low, fname=fname)

    Qlocal, Qglobal, Kmax = get_scalars(df_score['Qnx'].values)
    if verbose:
        print(f"Qlocal = {Qlocal:.2f}, Qglobal = {Qglobal:.2f}, Kmax = {Kmax}")

    return Qlocal, Qglobal, Kmax
