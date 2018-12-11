#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT TO VISUALIZE DG FEATURES


@date: 2018.04.10
@author: Miguel Simão (miguel.simao@uc.pt)
"""

from sys import stdout
import h5py
import numpy as np
from transforms3d import euler
from matplotlib import pyplot as plt

from pyquaternion import Quaternion
from quaternionfunctions import quaternion_to_euler_angle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.manifold import TSNE, SpectralEmbedding
from scipy import interpolate



#dir_dataset_sg = '/media/linuxdata/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG24_dataset.h5'
dir_dataset_dg = '/home/simao/Drive/PhD/0-Datasets/UC2017/DG10/DG10_dataset.h5'

# Open H5 files to read
f = h5py.File(dir_dataset_dg,'r')

# Load static gesture data set
X = f['Predictors']
T = f['Target']
U = f['User']

X = np.asarray(X).transpose([2,0,1])
T = np.asarray(T).transpose()[:,0]
U = np.asarray(U).transpose()[:,0]

# Dataset statistics
for u in np.unique(U):
    print('User %i: %i samples out of total %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))

#%% PREPROCESSING (SET INDEPENDENT)


mask_all = np.all(X==0,axis=2)

print('Preprocessing data : ')

# For EULER dataset only:
X[:,:,3:6] = X[:,:,3:6] / 180 * np.pi


print('Preprocessing data : ')
# Angle lifiting function
def angle_lift(y, n):
    for i in range(1, n):
        if np.isclose(y[i] - y[i-1], 2*np.pi, atol=0.1):
            # low to high
            y[i] -= 2*np.pi
        elif np.isclose(y[i] - y[i-1], -2*np.pi, atol=0.1):
            y[i] += 2*np.pi


# for each sample
for i,sample in enumerate(X):
    
    ### LIFT EULER ANGLES
    n = np.argwhere(np.all(sample==0,axis=1))
    n = n[0] if len(n)>0 else sample.shape[0]
        
    angle_lift(sample[:,3], int(n))
    angle_lift(sample[:,4], int(n))
    angle_lift(sample[:,5], int(n))
    
    ###############################################################
    ### Establish the base coordinate frame for the gesture sample
    
    # We can either use the first of the last frame of the gesture as the
    # transformation basis. For the following two lines, comment out the one
    # you don't want.
    zeroind = 0 #FIRST
#    zeroind = np.append(np.argwhere(np.all(sample==0,axis=1)), sample.shape[0])[0] - 1 #LAST
    
    f0 = sample[zeroind] # frame zero

    ### YAW CORRECTION ONLY (ORIGINAL SOLUTION)
    
    # Create basis homogeneous transformation matrix
    r0p = np.matrix(euler.euler2mat(f0[3],0,0,'rzyx')) ** -1
    p0 = np.matrix(f0[:3].reshape((-1,1)))
    
    # for each gesture frame
    for j,frame in enumerate(sample):
        if np.all(frame==0) : break
        p1 = np.matrix(frame[:3].reshape((-1,1)))
        p2 = r0p * (p1 - p0)
        frame[:3] = np.squeeze(p2)
        frame[3] = frame[3] - f0[3]


    stdout.write('\r% 5.1f%%' % ((i+1)/(X.shape[0])*100))
stdout.write('\n')
    

#%% SET SPLITTTING
    

ind_all = np.arange(X.shape[0])

ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=0.3,
                                       random_state=42)
ind_val, ind_test = train_test_split(ind_test,
                                     shuffle=True,
                                     stratify=T[ind_test],
                                     test_size=0.5,
                                     random_state=41)
## Set user 8 aside
ind_train_u8 = ind_train[U[ind_train]==8] # indexes of samples of U8
ind_train = ind_train[U[ind_train]!=8] # remove U8 samples
# number of samples to replace in each set
n_train = ind_train_u8.shape[0]
n_val   = n_train // 2
n_test  = n_train - n_val

ind_val = np.concatenate((ind_val, ind_train_u8[:n_val])) # append u8 samples
ind_test = np.concatenate((ind_test, ind_train_u8[-n_test:]))
ind_train = np.concatenate((ind_train, ind_val[:n_val], ind_test[:n_test]))
ind_val = ind_val[n_val:] # remove first n_val samples
ind_test = ind_test[n_test:] # remove first n_test samples

X_train = X[ind_train,:]
X_val   = X[ind_val,:]
X_test  = X[ind_test,:]
t_train = T[ind_train]
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train]
u_val   = U[ind_val]
u_test  = U[ind_test]


#%% FEATURE EXTRACTION



# Predictor scaling:
tmpX = X_train.reshape((-1,X_train.shape[2]))
tmpX = tmpX[~np.all(tmpX==0,1),] # DO NOT USE PADDING TO STANDARDIZE
scaler = preprocessing.StandardScaler().fit(tmpX)

def standardize(X,scaler):
    old_shape = X.shape
    X = X.reshape((-1,X.shape[2]))
    padIdx = np.all(X==0,axis=1)
    X = scaler.transform(X)
    X[padIdx] = 0
    X = X.reshape(old_shape)
    return X

X_train = standardize(X_train,scaler)
X_val = standardize(X_val,scaler)
X_test = standardize(X_test,scaler)

# One hot-encoding

def onehotencoding(t,x_shape,maxclasses):
    T = np.zeros((x_shape[0],x_shape[1],maxclasses))
    T[np.arange(len(t)),:,t-1] = 1
    return T

#t_train = onehotencoding(t_train,X_train.shape,10)
#t_val = onehotencoding(t_val,X_val.shape,10)
#t_test = onehotencoding(t_test,X_test.shape,10)


# All datasets condensed:
Xp = np.concatenate((X_train,X_val,X_test),axis=0)
Tp = np.concatenate((t_train,t_val,t_test),axis=0)
Up = np.concatenate((u_train,u_val,u_test),axis=0)
#Xp = X_train
#Tp = t_train
#Up = u_train 

#%% Feature set 1 (interpolated)



target_len = 40

Xf2 = np.zeros((Xp.shape[0],target_len*Xp.shape[2]))
for i,x in enumerate(Xp):
    # Remove padding (zero-rows)
    x = x[~np.all(x==0,axis=1)]
    # Time parametrization:
    s = np.arange(x.shape[0])/(x.shape[0]-1)
    sp = np.arange(target_len)/(target_len-1)
    # Resampling (cubic-interp 1d)
    x = interpolate.interp1d(s,x,kind='cubic',axis=0)(sp)
    x = x.reshape((1,-1))
    Xf2[i] = x


#%% FEATURE SET 2 (PCA)

Xf1 = np.zeros((Xp.shape[0],Xp.shape[2]))
for i,x in enumerate(Xp):
    # Remove padding (zero-rows)
    x = x[~np.all(x==0,axis=1)]
    pca = decomposition.PCA().fit(x)
    Xf1[i] = pca.components_[0]

#%% PLOTTING

# Default configurations
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0), fancybox=False)
plt.rc('lines', markeredgewidth=0.5, linewidth=0.5)

# Scale features:
Xf1 = Xf2
#Xf1 = preprocessing.StandardScaler().fit_transform(Xf2)
#Xf1 = scaler.transform(Xf1)

Xpca = decomposition.PCA(n_components=2).fit(Xf1).transform(Xf1)
#Xpca = TSNE(n_components=2, perplexity=15, early_exaggeration=6,verbose=1).fit_transform(Xf1)

# Plots
marker_list = ('o','x','+','^','*','1','2','3','4')
cmap = plt.get_cmap('tab10',10)

#fig = plt.figure()
fig = plt.figure(figsize=(4,4),dpi=300)
ax = fig.add_subplot(111)
#fig, ax = plt.subplots(dpi=300)

# FOR USER
for i,user in enumerate(np.unique(Up)):
    uind = Up == user
    # FOR CLASS
    for j,tclass in enumerate(np.unique(Tp)): # 
        cind = Tp == tclass
        ind = np.logical_and(cind,uind)
        ax.scatter(Xpca[ind,0],Xpca[ind,1],
                   s=9,
                   edgecolor='none',
                   c=cmap.colors[j],
                   marker=marker_list[i],
                   linewidth=0.5,
                   label=('DG%i' % tclass))

my_lgd = ax.legend(('DG1','DG2','DG3','DG4','DG5','DG6','DG7','DG8','DG9','DG10'),
           loc='center left',bbox_to_anchor=(1.02,.5),
           borderaxespad=0, frameon=True, fontsize=8,handletextpad=0, fancybox=False)
#my_suptitle = plt.suptitle('Train set, Features: PV, Transformation: Frame Last')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.xlabel('TSNE1')
#plt.ylabel('TSNE2')

fig.savefig('pca_all_dg.pdf',bbox_extra_artists=(my_lgd,), bbox_inches='tight')
#plt.close()