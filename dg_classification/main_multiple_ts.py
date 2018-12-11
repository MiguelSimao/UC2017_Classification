# -*- coding: utf-8 -*-
"""

SCRIPT TO TEST DG CLASSIFICATION


@date: 2018.04.10
@author: Miguel Simão (miguel.simao@uc.pt)
"""


from time import time
from sys import stdout
import h5py
import numpy as np
#from numpy.linalg import inv
from matplotlib import pyplot as plt

import math
from transforms3d import euler

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import decomposition

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis


# ENSURE REPRODUCIBILITY ######################################################
import os
import random
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(123)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###############################################################################



#%%

#dir_dataset_sg = '/media/linuxdata/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG24_dataset.h5'
dir_dataset_dg = '/home/simao/Drive/PhD/0-Datasets/UC2017/DG10/DG10_dataset_euler.h5'

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

def Tinv(T):
    r = T[:3,:3]
    rinv = r.transpose()
    p = T[:3,3]
    T2 = np.eye(4)
    T2[:3,:3] = rinv
    T2[:3,3] = np.matmul(rinv,-p)
    return T2

# Backup data:
Xb = X

print('Preprocessing data : ')

# For EULER dataset only:
X[:,:,3:6] = X[:,:,3:6] / 180 * math.pi


# for each sample
for i,sample in enumerate(X):
    
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
    t0 = np.eye(4)
    t0[:3,:3] = euler.euler2mat(f0[3],0,0,'rzyx')
    t0[:3,3] = f0[:3]
    
    # for each gesture frame
    for j,frame in enumerate(sample):
        if np.all(frame==0) : break
        t1 = np.eye(4)
        t1[:3,:3] = euler.euler2mat(frame[3],frame[4],frame[5],axes='rzyx')
        t1[:3,-1] = frame[:3]
        t2 = np.dot(Tinv(t0),t1)
        framep = frame
        framep[:3] = t2[:3,-1]
        framep[3:6] = euler.mat2euler(t2[:3,:3],axes='rzyx')
        X[i,j] = framep

    ### QUATERNION IMPLEMENTATION
#    q0 = f0[3:7] # quaternion components
    
#    # Create basis transformation matrix
#    t0 = np.eye(4)
#    t0[:3,:3] = quaternions.quat2mat(q0)
#    t0[:3,-1] = f0[:3]
#    r0 = quaternions.quat2mat(q0)    
    
#    # for each gesture frame
#    for j,frame in enumerate(sample):
#        if np.all(frame==0) : break
#        t1 = np.eye(4)
#        t1[:3,:3] = quaternions.quat2mat(frame[3:7])
#        t1[:3,-1] = frame[:3]
#        t2 = np.matmul(Tinv(t1), t0)
#        framep = frame
#        framep[:3] = t2[:3,-1]
#        framep[3:7] = quaternions.mat2quat(t2[:3,:3])
#        X[i,j] = framep
    
    stdout.write('\r% 5.1f%%' % ((i+1)/(X.shape[0])*100))
stdout.write('\n')
    

#%% SET SPLITTTING
    
ind_all = np.asarray(range(X.shape[0]))

# Data splitting 1 : all -> train and rest
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(ind_all,T[ind_all])
ind_train,ind_test = next(sss)

# Data splitting 2 : test -> validation and test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(ind_test,T[ind_test])
ind_val, ind_test = next(sss)

X_train = X[ind_train,:,:]
X_val   = X[ind_val,:,:]
X_test  = X[ind_test,:,:]


#%% FEATURE EXTRACTION

# Variable scaling:
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
X_val   = standardize(X_val,scaler)
X_test  = standardize(X_test,scaler)

# One hot-encoding
def tsonehotencoding(t,x,maxclasses):
    T = np.zeros((x.shape[0],x.shape[1],maxclasses))
    
    # function to turn sample indexes 
    for i,sample in enumerate(x):
        T[i,:,t[i]-1] = 1
        
    return T

#T_train = utils.to_categorical(T[ind_train]-1,10)
#T_val   = utils.to_categorical(T[ind_val]-1,10)
#T_test  = utils.to_categorical(T[ind_test]-1,10)


#%% EXTRACT FEATURES

# FEATURE SET 3 (PCA TS)
def tspvfeatures(X):
    Xf = np.zeros(X.shape)
    # For each sample
    for i,sample in enumerate(X):
        # Find first index of padding:
        padind = np.append(np.argwhere(np.all(sample==0,axis=1)), sample.shape[0])[0] 
        # For each timestep
        for j in range(padind):
            pca = decomposition.PCA().fit(sample[:j+1])
            Xf[i][j] = pca.components_[0]
    return Xf


#F_train = interpolatedfeatures(X_train,20)
#F_val = interpolatedfeatures(X_val,20)
#F_test = interpolatedfeatures(X_test,20)

F_train = tspvfeatures(X_train)
F_val   = tspvfeatures(X_val)
F_test  = tspvfeatures(X_test)
T_train = tsonehotencoding(T[ind_train], X_train, 10)
T_val   = tsonehotencoding(T[ind_val], X_val, 10)
T_test  = tsonehotencoding(T[ind_test], X_test, 10)

def tsunroll(X,T):
    x = []
    t = []
    sz = np.empty((0,1),int)
    for i,sample in enumerate(X):
        ind_include = ~np.all(sample==0,axis=1)
        sample = sample[ind_include]
        x.append(sample)
        t.append(T[i][ind_include])
        sz = np.append(sz,np.argwhere(~ind_include)[0]-1)
    return np.concatenate(x), np.concatenate(t), sz

def tsunroll_testsubset(X,T):
    #function to unroll the test set. we are only interested at a limited
    #subset of timesteps (25, 50, 75 and 100% of gesture completion)
    x = []
    t = []
    sz = np.empty((0,1),int)
    # For each sample
    for i,sample in enumerate(X):
        ind_include = ~np.all(sample==0,axis=1)
        sample = sample[ind_include]
        ind_subset = (sample.shape[0] - 1) * np.asarray([0.25,0.5,0.75,1.0])
        ind_subset = np.ceil(ind_subset).astype(np.int)

        x.append(sample[ind_subset])
        t.append(T[i][ind_subset])
        sz = np.append(sz,np.argwhere(~ind_include)[0]-1)
    return np.concatenate(x), np.concatenate(t), sz


F_train,T_train,sz_train = tsunroll(F_train,T_train)
F_val,T_val,sz_val = tsunroll(F_val,T_val)
F_test,T_test,sz_test = tsunroll_testsubset(F_test,T_test)

# Feature scaling
scaler_features = preprocessing.StandardScaler().fit(F_train)
F_train = scaler_features.transform(F_train)
F_val = scaler_features.transform(F_val)
F_test = scaler_features.transform(F_test)

T_train = np.argmax(T_train,axis=1)
T_val = np.argmax(T_val,axis=1)
T_test = np.argmax(T_test,axis=1)

#%% DEFINE MODELS

names = [
        "Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "QDA",
        "LDA"
        ]

classifiers = [
        KNeighborsClassifier(5, algorithm='auto'),
        DecisionTreeClassifier(max_depth=30),
        RandomForestClassifier(max_depth=30, n_estimators=10, max_features=1),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
        ]

scores = []

for name, clf in zip(names, classifiers):
    print(' ::: %s :::' % (name))
    time_start = time()
    clf.fit(F_train, T_train)
    time_elapsed = time() - time_start
    print('Training time: %.1f s' % time_elapsed)
    
    time_start = time()
    train_score = (
            clf.score(F_train[0:-1:4],T_train[0:-1:4]) * 100,
            clf.score(F_train[1:-1:4],T_train[1:-1:4]) * 100,
            clf.score(F_train[2:-1:4],T_train[2:-1:4]) * 100,
            clf.score(F_train[3:-1:4],T_train[3:-1:4]) * 100)
    val_score = (   clf.score(F_val[0:-1:4],T_val[0:-1:4]) * 100,
                    clf.score(F_val[1:-1:4],T_val[1:-1:4]) * 100,
                    clf.score(F_val[2:-1:4],T_val[2:-1:4]) * 100,
                    clf.score(F_val[3:-1:4],T_val[3:-1:4]) * 100)
    test_score = (  clf.score(F_test[0:-1:4],T_test[0:-1:4]) * 100,
                    clf.score(F_test[1:-1:4],T_test[1:-1:4]) * 100,
                    clf.score(F_test[2:-1:4],T_test[2:-1:4]) * 100,
                    clf.score(F_test[3:-1:4],T_test[3:-1:4]) * 100)
    time_elapsed = time() - time_start
    scores.append((train_score,val_score,test_score))
    print('Testing time: %.1f s' % time_elapsed)
    
    print('Accuracies:')
    print('Train: %.1f | %.1f | %.1f | %.1f' % train_score)
    print('  Val: %.1f | %.1f | %.1f | %.1f' % val_score)
    print(' Test: %.1f | %.1f | %.1f | %.1f' % test_score)
