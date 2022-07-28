# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:22:16 2022

@author: hou
"""
from FCTMdd import fuzzycmedoids
import scipy.interpolate as interpolate
import tslearn.metrics as metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import trange
import warnings
warnings.filterwarnings("ignore")

def calc_correlation(actual, predic):
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    return numerator / denominator
def getW():
    RmseSum=list()
    SRmsemean=list()
    for i in range(8):
        for j in range(n):
            yT=dataset.iloc[:,j+1]
            yB = yT[0:N:i+2]
            Xtime = X_time[0:N:i+2]
            t_T, c_T, k_T = interpolate.splrep(Xtime, yB,s=0,k=3)
            xmin, xmax = Xtime.min(), Xtime.max()
            xx = np.linspace(xmin, xmax, N)
            spline_y = interpolate.BSpline(t_T, c_T, k_T, extrapolate=False)
            y_true = yT.tolist()
            y_pred = spline_y(xx).tolist()
            #mse_A = mean_squared_error(y_true, y_pred)
            Persen=calc_correlation(y_true, y_pred)
            #rmse_A = np.sqrt(mse_A)
        RmseSum.append(Persen)
        Rmsemean=np.mean(RmseSum)
    SRmsemean.append(Rmsemean) 
    W=SRmsemean.index(min(SRmsemean))
    w=W+2
    return w

def getClusters(membership_mat):
    cluster_labels=list()
    for i in range(n-1):
        max_value,idx=max((val,idx) for (idx,val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx+1)
    return cluster_labels

n_passes=20
max_iter=100
membership_method=('FCM', 2)
w=1

datasets = [("data/BeetleFly.csv",2)]
print("--------------------------------------------")
from sklearn.metrics.cluster import adjusted_rand_score,fowlkes_mallows_score
from metrics import cal_clustering_metric,purity_score
res={"Dataset":[],"ARI":[],"FMI":[],"NMI":[]}
kSARI,kSFMI,kSNMI=[],[],[]
for i in trange(len(datasets)):
    data=datasets[i]
    dataset=pd.read_csv(data[0],header=None)
    res["Dataset"].append(data[0][5:-4])

    label=dataset.iloc[-1:]
    label=np.array(label)
    label=label[:,1:].reshape(-1)

    N = dataset.shape[0]
    n= dataset.shape[1]
    X_time=dataset.iloc[0:N-1,0]
    
    dataset=dataset.iloc[0:N-1,1:]
    c=data[1]
    Xtime = X_time[0:N:w]
    
    Bmat=list()
    for i in range(n-1):
        yT=dataset.iloc[:,i]
        yB = yT[0:N:w]
        t_T, c_T, k_T = interpolate.splrep(Xtime, yB,s=0,k=3)
        Bmat.append(c_T)
    X=np.array(Bmat)

    bestMedoids, bestMembership, bestNIter, nfound = fuzzycmedoids(X, c=c, max_iter=max_iter, n_passes=n_passes, membership_method=membership_method,It=1)
    cluster_labels=getClusters(bestMembership)
    

    init_inds = np.arange(n-1)
    bestMedoids = np.random.permutation(init_inds)[:c]
    bestMedoids=list(bestMedoids)
    cluster_New=list()
    for i in range(c):
        cluster_d= dataset.iloc[:, bestMedoids[i]]
        cluster_New.append(cluster_d) 
    cluster_New=np.array(cluster_New)

    plt.figure(figsize=(12,6),dpi=300)
    plt.plot(dataset,color='#BBFFBB',linewidth = '2',label='Original sample')
    plt.plot(cluster_New.T,linewidth = '2.3',label='Cluster Center')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Timestamp',fontsize=20)
    plt.title(str(data[0][5:-4]),fontsize=35)
    plt.savefig('tu/'+str(data[0][5:-4])+'.png')
    
    ARI=adjusted_rand_score(label,cluster_labels)
    res["ARI"].append(ARI)
    
    FMI=fowlkes_mallows_score(label,cluster_labels,sparse=False)
    res["FMI"].append(FMI)
    
    _,NMI=cal_clustering_metric(label,cluster_labels)
    res["NMI"].append(NMI)
    

    kSARI.append(ARI)
    kSFMI.append(FMI)
    kSNMI.append(NMI)

    
df=pd.DataFrame(res)
print(df)
