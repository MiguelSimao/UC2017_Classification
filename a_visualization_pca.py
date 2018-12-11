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



#%% LOAD DATA

dir_dataset_sg = '/home/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG24_dataset.h5'

# Open H5 files to read
f = h5py.File(dir_dataset_sg,'r')

# Load static gesture data set
X = f['Predictors']
T = f['Target']
U = f['User']

X = np.array(X).transpose()
T = np.array(T).transpose()[:,0]
U = np.array(U).transpose()[:,0]

# Limit number of users
usubset = np.unique(U)
uind = np.isin(U,usubset)
X, T, U = X[uind], T[uind], U[uind]

# Limit classes:
tsubset = np.unique(T)
tind = np.isin(T,tsubset)
X, T, U = X[tind], T[tind], U[tind]

# Standardize predictors
Xp = scale(X)
pca = PCA(n_components=2).fit(Xp)
Xp = pca.transform(Xp)

# Classes in dataset:
class_list = np.unique(T)

# Default configurations
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0),fancybox=False)
plt.rc('lines', markeredgewidth=0.5,linewidth=0.5)



#%% PLOT DATA INTERUSER BY CLASS
marker_list = ('x','+','^','*','1','2','3','4')

f,axarr = plt.subplots(6,4, sharex=True, sharey=True, figsize=(5,7),dpi=300)
f.subplots_adjust(hspace=0.0,wspace=0)

for i in np.unique(T):
    indarr = np.unravel_index(i-1,(6,4))
    t = T==i
    
    # Scatter 1: not class
    axarr[indarr].scatter(Xp[:,0],Xp[:,1],
         s=6,
         facecolor='gray',
         alpha=0.5,
         linewidth=0,
         marker='.',
         label='All',
         zorder=0)
    # Title
    axarr[indarr].text(5.5,4.3,'SG%i' % i, ha='right', fontsize=8)
    axarr[indarr].set_rasterization_zorder(1)
    
    for indmarker,j in enumerate(np.unique(U)):
        ji = np.logical_and(t,U==j)
        
        # Scatter 2: class
        axarr[indarr].scatter(Xp[ji,0], Xp[ji,1],
             s=8,
             edgecolor='none',
             marker=marker_list[indmarker],
             linewidth=0.5,
             label=('U%i' % j))
        
        
lgd = axarr[indarr].legend(loc=2,
                    markerscale=2,
                    borderpad=0.5,
                    handletextpad=0.1,
                    bbox_to_anchor=(1.05,4.0))
ty = f.text(0.03, 0.5, 'PC2', va='center', rotation='vertical')
tx = f.text(0.5, 0.08, 'PC1', ha='center')
f.savefig('pca_interuser.pdf',bbox_extra_artists=(lgd,ty,tx,), bbox_inches='tight')
#plt.close()
print('Done!')

#%% PLOT DATA INTERUSER BY CLASS [reduced]
marker_list = ('x','+','^','*','1','2','3','4')

f,axarr = plt.subplots(3,3, sharex=True, sharey=True, figsize=(3.5,3.5),dpi=300)
f.subplots_adjust(hspace=0.0,wspace=0)

for i in np.unique(T)[:9]:
    indarr = np.unravel_index(i-1,(3,3))
    t = T==i
    
    # Scatter 1: not class
    axarr[indarr].scatter(Xp[:,0],Xp[:,1],
         s=6,
         facecolor='gray',
         alpha=0.5,
         linewidth=0,
         marker='.',
         label='All',
         zorder=0)
    # Title
    axarr[indarr].text(5.5,4.3,'SG%i' % i, ha='right', fontsize=8)
    axarr[indarr].set_rasterization_zorder(1)
    
    for indmarker,j in enumerate(np.unique(U)):
        ji = np.logical_and(t,U==j)
        
        # Scatter 2: class
        axarr[indarr].scatter(Xp[ji,0], Xp[ji,1],
             s=8,
             edgecolor='none',
             marker=marker_list[indmarker],
             linewidth=0.5,
             label=('U%i' % j))
        
        
#lgd = axarr[indarr].legend(loc=2,
#                    markerscale=2,
#                    borderpad=0.5,
#                    handletextpad=0.1,
#                    bbox_to_anchor=(1.05,4.0))
ty = f.text(0.01, 0.5, 'PC2', va='center', rotation='vertical')
tx = f.text(0.5, 0.02, 'PC1', ha='center')
f.savefig('pca_interuser9.pdf',bbox_extra_artists=(ty,tx,), bbox_inches='tight')
#plt.close()
print('Done!')

#%% PLOT DATA 

f = plt.figure(figsize=(5,5),dpi=300)

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


for ind,ti in enumerate(np.unique(T)):
    indarr = np.unravel_index(ind,(len(marker_list),len(color_list)))
    ji = np.logical_and(T==ti,U==1)
    
    plt.scatter(Xp[ji,0],Xp[ji,1],
                s=12,
                linewidth=1.0,
                edgecolors='none',
                facecolor=color_list[indarr[1]],
                marker=marker_list[indarr[0]],
                label=('G%i' % ti))
plt.legend(loc=1, fontsize = 'xx-small', bbox_to_anchor=(1.14,0.9))

#f.savefig('pca_allclasses_miguel5.png')
#plt.close()

#%% PLOT OLD VS NEW SAMPLES
#
#f,axarr = plt.subplots(5,5, sharex=True, sharey=True, figsize=(10,10),dpi=450)
#f.subplots_adjust(hspace=0.0,wspace=0)
#
#
#xind = np.asarray(range(X.shape[0]))
#
#
#for i in np.unique(T):
#    indarr = np.unravel_index(i-1,(5,5))
#    
#    ti = np.isin(T,[i])
#    ji = np.logical_and(ti,xind<500)
#    
#    axarr[indarr].scatter(Xp[ji,0],Xp[ji,1],
#                            s=12,
#                            linewidth=1.0,
#                            edgecolors='none',
#                            facecolor=color_list[0],
#                            marker=marker_list[0],
#                            label=('Old'))
#    
#    ji = np.logical_and(ti,np.logical_and(xind>=500,xind<625))
#    
#    axarr[indarr].scatter(Xp[ji,0],Xp[ji,1],
#                            s=12,
#                            linewidth=1.0,
#                            edgecolors='none',
#                            facecolor=color_list[2],
#                            marker=marker_list[0],
#                            label=('New' % i))
#    axarr[indarr].set_title('G%i' % i)
#    
#axarr[indarr].legend(loc=0)
#f.suptitle('Samples for user 1: Old vs New')
#f.text(0.5, 0.04, 'PCA 1', ha='center')
#f.text(0.04, 0.5, 'PCA 2', va='center', rotation='vertical')
##plt.tight_layout()
#f.savefig('pca_miguel_old_vs_new.png',)
#plt.close()