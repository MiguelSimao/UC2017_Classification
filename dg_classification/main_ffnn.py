# -*- coding: utf-8 -*-
"""

SCRIPT TO TEST DG CLASSIFICATION


@date: 2018.04.10
@author: Miguel Simão (miguel.simao@uc.pt)
"""


from sys import stdout
from time import time
import h5py
import numpy as np
from matplotlib import pyplot as plt

import math
from transforms3d import euler

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn import preprocessing
from sklearn import decomposition
#from sklearn.metrics import confusion_matrix

from scipy import interpolate


# ENSURE REPRODUCIBILITY ######################################################
import os
import random
import tensorflow as tf
from keras import backend as K

def reset_random():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    random.seed(12345)
    
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(123)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
reset_random()
###############################################################################


from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import utils, regularizers

#%% EXPERIMENT PARAMETERS

#param_feat = 'CI'
param_feat = 'PV'


#%% LOAD DATA

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
Xb = X.copy()

print('Preprocessing data : ')

# For EULER dataset only:
X[:,:,3:6] = X[:,:,3:6] / 180 * math.pi

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
    
    f0 = sample[zeroind].copy() # frame zero


    ### YAW CORRECTION ONLY (ORIGINAL SOLUTION)
    
    # Create basis homogeneous transformation matrix
#    t0 = np.eye(4)
#    t0[:3,:3] = euler.euler2mat(f0[3],0,0,'rzyx')
#    t0[:3,3] = f0[:3]
    r0p = np.matrix(euler.euler2mat(f0[3],0,0,'rzyx')) ** -1
    p0 = np.matrix(f0[:3].reshape((-1,1)))
    
    # for each gesture frame
    for j,frame in enumerate(sample):
        if np.all(frame==0) : break
#        t1 = np.eye(4)
#        t1[:3,:3] = euler.euler2mat(frame[3],frame[4],frame[5],axes='rzyx')
#        t1[:3,-1] = frame[:3]
#        t2 = np.dot(Tinv(t0),t1)
#        framep = frame
#        framep[:3] = t2[:3,-1]
#        framep[3:6] = euler.mat2euler(t2[:3,:3],axes='rzyx')
#        X[i,j] = framep
        p1 = np.matrix(frame[:3].reshape((-1,1)))
        p2 = r0p * (p1 - p0)
        frame[:3] = np.squeeze(p2)
        frame[3] = frame[3] - f0[3]

    stdout.write('\r% 5.1f%%' % ((i+1)/(X.shape[0])*100))
stdout.write('\n')
    

#%% SET SPLITTTING
    
#ind_all = np.asarray(range(X.shape[0]))
#
## Data splitting 1 : all -> train and rest
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(ind_all,T)
#ind_train,ind_test = next(sss)
#
## Data splitting 2 : test -> validation and test
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(ind_test,T[ind_test])
#ind_val, ind_test = next(sss)
#
#X_train = X[ind_train,:,:]
#X_val   = X[ind_val,:,:]
#X_test  = X[ind_test,:,:]

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

T_train = utils.to_categorical(T[ind_train]-1,10)
T_val   = utils.to_categorical(T[ind_val]-1,10)
T_test  = utils.to_categorical(T[ind_test]-1,10)


#%% EXTRACT FEATURES

# FEATURE SET 1 (interpolated)
def interpolatedfeatures(X,target_len):
    
    Xf = np.zeros((X.shape[0],target_len*X.shape[2]))
    for i,x in enumerate(X):
        # Remove padding (zero-rows)
        x = x[~np.all(x==0,axis=1)]
        # Time parametrization:
        s = np.arange(x.shape[0])/(x.shape[0]-1)
        sp = np.arange(target_len)/(target_len-1)
        # Resampling (cubic-interp 1d)
        x = interpolate.interp1d(s,x,kind='cubic',axis=0)(sp)
        x = x.reshape((1,-1))
        Xf[i] = x
        
    return Xf

# FEATURE SET 2 (PCA)
def princvecfeatures(X):
    Xf = np.zeros((X.shape[0],X.shape[2]))
    for i,x in enumerate(X):
        # Remove padding (zero-rows)
        x = x[~np.all(x==0,axis=1)]
        pca = decomposition.PCA().fit(x)
        Xf[i] = pca.components_[0]
    return Xf

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


if param_feat == 'CI':
    F_train = interpolatedfeatures(X_train,20)
    F_val = interpolatedfeatures(X_val,20)
    F_test = interpolatedfeatures(X_test,20)
elif param_feat == 'PV':
    F_train = princvecfeatures(X_train)
    F_val = princvecfeatures(X_val)
    F_test = princvecfeatures(X_test)
else:
    NotImplementedError

#F_train = tspvfeatures(X_train)
#F_val   = tspvfeatures(X_val)
#F_test  = tspvfeatures(X_test)
#T_train = tsonehotencoding(T[ind_train], X_train, 10)
#T_val   = tsonehotencoding(T[ind_val], X_val, 10)
#T_test  = tsonehotencoding(T[ind_test], X_test, 10)

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


#F_train,T_train,sz_train = tsunroll(F_train,T_train)
#F_val,T_val,sz_val = tsunroll(F_val,T_val)
#F_test,T_test,sz_test = tsunroll_testsubset(F_test,T_test)

# Feature scaling
scaler_features = preprocessing.StandardScaler().fit(F_train)
F_train = scaler_features.transform(F_train)
F_val = scaler_features.transform(F_val)
F_test = scaler_features.transform(F_test)


#%% DEFINE NETWORK

# RESET RANDOM GENERATION
reset_random()

# CI-FULL
if param_feat == 'CI':
    inputs = Input(shape=(F_train.shape[1],))
    x = Dense(100, activation='relu',
              kernel_regularizer=regularizers.l2())(inputs)
    x = GaussianNoise(.2)(x)
    x = Dense(200, activation='relu',
              kernel_regularizer=regularizers.l2())(x)
    x = BatchNormalization()(x)
elif param_feat == 'PV':
    inputs = Input(shape=(F_train.shape[1],))
    x = Dense(300, activation='relu',
              kernel_regularizer=regularizers.l2())(inputs)
    x = Dense(300, activation='relu',
              kernel_regularizer=regularizers.l2())(x)
    x = BatchNormalization()(x)
else:
    NotImplementedError

outputs = Dense(T_train.shape[1], activation='softmax')(x)

net = Model(inputs, outputs, name='DG_FFNN')

net.compile(optimizer=SGD(0.01, decay=1e-6),
            loss='categorical_crossentropy',
            metrics=['acc'])

time_start = time()
history = net.fit(x=F_train,y=T_train,
                  validation_data=(F_val,T_val),
                  batch_size=128,
                  epochs=1000,
                  callbacks=[EarlyStopping('val_loss',patience=12)],
                  verbose=1)
time_elapsed = time() - time_start
print('Training time: %.1f s' % time_elapsed)

time_start = time()
acc_train = net.evaluate(F_train,T_train)[1]*100
acc_val = net.evaluate(F_val,T_val)[1]*100
acc_test = net.evaluate(F_test[u_test!=8],T_test[u_test!=8])[1]*100
acc_test_8 = net.evaluate(F_test[u_test==8],T_test[u_test==8])[1] * 100

time_elapsed = time() - time_start
print('Testing time: %.1f s' % time_elapsed)


print('Accuracies:')
print('Train: %.1f' % (acc_train))
print('  Val: %.1f' % (acc_val))
print(' Test: %.1f' % (acc_test))
print('Test8: %.1f' % (acc_test_8))