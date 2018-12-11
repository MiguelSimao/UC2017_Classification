#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:44:16 2018

@author: simao
"""

import h5py
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

dir_dataset_sg = '/media/linuxdata/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG25_dataset.h5'


#%% LOAD DATA

# Open H5 files to read
f = h5py.File(dir_dataset_sg,'r')

# Load static gesture data set
X = f['Predictors']
T = f['Target']
U = f['User']

X = np.array(X).transpose()
T = np.array(T).transpose()[:,0]
U = np.array(U).transpose()[:,0]


#X = X[:500]
#T = T[:500]
#U = U[:500]

# Limit number of users (subset)
#usubset = np.unique(U)
usubset = [1]
uind = np.isin(U,usubset)
X, T, U = X[uind], T[uind], U[uind]

# Limit classes (subset)
tsubset = np.unique(T)
tind = np.isin(T,tsubset)
X, T, U = X[tind], T[tind], U[tind]


#%%  Apply TSNE (t-distributed Stochastic Neighbor Embedding)

# Standardize predictors
Xp = scale(X)
pca = PCA(n_components=2).fit(Xp)
Xp = pca.transform(Xp)

Xemb = TSNE(n_components=2, early_exaggeration=16).fit_transform(Xp)




#%% PLOT DATA INTERUSER BY CLASS
marker_list = ('o','x','+','^','*','1','2','3','4')

f,axarr = plt.subplots(5,5, sharex=True, sharey=True, figsize=(10,10),dpi=450)
f.subplots_adjust(hspace=0.15,wspace=0)
for i in np.unique(T):
    indarr = np.unravel_index(i-1,(5,5))
    t = T==i
    
    # Scatter 1: not class
    axarr[indarr].scatter(Xemb[:,0],Xemb[:,1],
         s=6,
         facecolor='gray',
         alpha=0.5,
         linewidth=0,
         marker='.',
         label='All classes')
    axarr[indarr].set_title('Class %i' % i)

    for indmarker,j in enumerate(np.unique(U)):
        ji = np.logical_and(t,U==j)
        
        # Scatter 2: class
        axarr[indarr].scatter(Xemb[ji,0], Xemb[ji,1],
             s=8,
             edgecolor='none',
             marker=marker_list[indmarker],
             #facecolors='r',
             linewidth=0.5,
             label=('User %i' % j))

axarr[indarr].legend(loc=2,bbox_to_anchor=(1.05,1.0))

f.savefig('tsne_interuser2.png',)
plt.close()
print('Done!')

#%% PLOT DATA 
f = plt.figure(figsize=(5,5),dpi=300)

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


for ind,ti in enumerate(np.unique(T)):
    indarr = np.unravel_index(ind,(len(marker_list),len(color_list)))
    ji = np.logical_and(T==ti,U==1)
    
    plt.scatter(Xemb[ji,0],Xemb[ji,1],
                s=12,
                linewidth=1.0,
                edgecolors='none',
                facecolor=color_list[indarr[1]],
                marker=marker_list[indarr[0]],
                label=('G%i' % ti))
plt.legend(loc=1, fontsize = 'xx-small', bbox_to_anchor=(1.14,0.9))

f.savefig('tsne_allclasses.png')
plt.close()