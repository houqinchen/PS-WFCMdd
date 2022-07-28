

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:13:12 2022

@author: hou
"""

import numpy as np
import warnings
from matrixprofile import algorithms

def unique_rows(a, return_index=False, return_inverse=False, return_counts=False):
    try:
        dummy, uniqi, inv_uniqi, counts = np.unique(a.view(a.dtype.descr * a.shape[1]), return_index = True, return_inverse = True, return_counts = True)
        out = [a[uniqi,:]]
        if return_index:
            out.append(uniqi)
        if return_inverse:
            out.append(inv_uniqi)
        if return_counts:
            out.append(counts)
    except ValueError:
        s = set()
        for i in range(a.shape[0]):
            s.add(tuple(a[i,:].tolist()))
        out = [np.array([row for row in s])]
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)
    
def precomputeWeightedDmat(dmat, weights, squared=False):
    #计算 FCMdd 的加权距离矩阵。
    if squared:
        dmat = dmat**2
    if weights is None:
        return dmat
    else:
        assert weights.shape[0] == dmat.shape[0]
        return dmat * weights
    
def computeMembership(dmat, medoids, method='FCM', param=2):
    c = len(medoids)
    r = dmat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method in ['FCM', 2, '2']:
            assert param >= 1
            tmp = (1 / r)**(1 / (param - 1))
        elif method in [3, '3']:
            assert param > 0
            tmp = np.exp(-param * r)
        elif method in [4, '4']:
            assert param.shape == (c,)
            tmp = 1/(1 + r/param[:, None])
        elif method in [5, '5']:
            assert param.shape == (c,)
            tmp = np.exp(-r/param[:, None])

        membership = tmp / tmp.sum(axis=1, keepdims=True)
    for medi, med in enumerate(medoids):
        membership[med,:] = 0.
        membership[med, medi] = 1.
    return membership
    
def fuzzycmedoids(X, c, membership_method=('FCM', 2), weights=None, n_passes=1, max_iter=1000, init_inds=None, potential_medoid_inds=None,It=None):
    N= X.shape[0]

    if init_inds is None:
        init_inds = np.arange(N)

    if not potential_medoid_inds is None:
        init_inds = np.array([i for i in init_inds if i in potential_medoid_inds], dtype=int)
    else:
        potential_medoid_inds = np.arange(N)

    if len(init_inds) == 0:
        print('No possible init_inds provided.')
        return

    allMedoids = np.zeros((n_passes, c))
    bestInertia = None

    for passi in range(n_passes):
        #选择 c 个随机中心点
        currMedoids = np.random.permutation(init_inds)[:c]
        dists=np.zeros((N,c),dtype=complex)
        for i in range(N):
             X1=X[i]
             for j in range(c):
                X2=X[currMedoids[j]]
                distances=algorithms.mass2(X1,X2)
                dists[i:,j]=distances
        Wd = dists
        newMedoids = np.zeros(c, dtype=int)
        newmnInd = np.zeros(c, dtype=int)
        for i in range(max_iter):
            #计算隶属度
            membership = computeMembership(Wd, currMedoids, method=membership_method[0], param=membership_method[1])
            #为每个集群选择新的中心点，最小化模糊目标函数
            weights=(membership+Wd).sum(axis=1, keepdims=True)
            weights=weights**(-1)

            wdmat = precomputeWeightedDmat(Wd, weights)
            totInertia = 0
            for j in range(c):
            #在每个集群内，通过最小化差异（由集群的隶属度加权）来找到新的聚类中心
                inertiaMat = np.tile(membership[:, j][:, None].T, (c, 1)).T * wdmat[potential_medoid_inds,:]
                inertiaVec = inertiaMat.sum(axis=1)
                mnInd = np.argmin(inertiaVec)
                newmnInd[j] = potential_medoid_inds[mnInd]
                newMedoids[j] = potential_medoid_inds[mnInd]
               # 将这个新中心点的Inertia添加到运行总数中
                totInertia += inertiaVec[mnInd]
            dic={}.fromkeys(newmnInd)
            if i>=It:
               if len(dic)==len(newmnInd):
                  break
            elif (newMedoids == currMedoids).all():
             #模型终止条件
                allMedoids[passi,:] = sorted(currMedoids)
                break
            currMedoids = newMedoids.copy()
            
        if bestInertia is None or totInertia < bestInertia:
            bestInertia = totInertia
            bestMedoids = currMedoids.copy()
            bestMembership = membership.copy()
            bestNIter = i + 1
    
    nfound = unique_rows(allMedoids).shape[0]

    return bestMedoids, bestMembership, bestNIter, nfound