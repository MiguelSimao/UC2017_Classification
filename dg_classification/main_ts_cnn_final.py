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

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn import preprocessing
from sklearn import decomposition

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
from keras.layers import Input, Dense, GaussianNoise, Masking, Dropout, LSTM, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D, Flatten, BatchNormalization
import keras.optimizers
from keras.callbacks import EarlyStopping
from keras import utils


#%%

#dir_dataset_sg = '/media/linuxdata/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG24_dataset.h5'
dir_dataset_dg = '/home/simao/Drive/PhD/0-Datasets/UC2017/DG10/DG10_dataset_euler.h5'
#dir_dataset_dg = '../Datasets/UC2017/DG10/DG10_dataset_euler.h5'

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
    
ind_all = np.arange(X.shape[0])

# Data splitting 1 : all -> train and rest
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(ind_all,T)
#ind_train,ind_test = next(sss)
ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=0.3,
                                       random_state=42)

# Data splitting 2 : test -> validation and test
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(ind_test,T[ind_test])
#ind_val, ind_test = next(sss)
ind_val, ind_test = train_test_split(ind_test,
                                     shuffle=True,
                                     stratify=T[ind_test],
                                     test_size=0.5,
                                       random_state=41)

X_train = X[ind_train,:,:]
X_val   = X[ind_val,:,:]
X_test  = X[ind_test,:,:]

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

X_train = X[ind_train,:,:]
X_val   = X[ind_val,:,:]
X_test  = X[ind_test,:,:]
t_train = T[ind_train]
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train]
u_val = U[ind_val]
u_test = U[ind_test]


# Split statistics : TRAIN
for u in np.unique(u_train):
    print('[TRAIN] U%i: %i/%i (%.1f%%)' % (u, sum(u_train==u), len(u_train),
                                           sum(u_train==u)/len(u_train)*100))
# Split statistics : VAL
for u in np.unique(u_val):
    print('  [VAL] U%i: %i/%i (%.1f%%)' % (u, sum(u_val==u), len(u_val),
                                           sum(u_val==u)/len(u_val)*100))
# Split statistics : TEST
for u in np.unique(u_test):
    print(' [TEST] U%i: %i/%i (%.1f%%)' % (u, sum(u_test==u), len(u_test),
                                           sum(u_test==u)/len(u_test)*100))


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

F_train = X_train
F_val   = X_val
F_test  = X_test
T_train = tsonehotencoding(t_train, X_train, 10)
T_val   = tsonehotencoding(t_val, X_val, 10)
T_test  = tsonehotencoding(t_test, X_test, 10)

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
tmpX = F_train.reshape((-1,F_train.shape[2]))
tmpX = tmpX[~np.all(tmpX==0,1),] # DO NOT USE PADDING TO STANDARDIZE
scaler_features = preprocessing.StandardScaler().fit(tmpX)

F_train = standardize(F_train,scaler_features)
F_val = standardize(F_val,scaler_features)
F_test = standardize(F_test,scaler_features)

#T_train = np.argmax(T_train,axis=1)
#T_val = np.argmax(T_val,axis=1)
#T_test = np.argmax(T_test,axis=1)

#%% PADDING
# Pad both training and validation set. Testing set not needed

batch_size = 64
if (F_train.shape[0] % batch_size) != 0 :
    n = batch_size - F_train.shape[0] % batch_size
    F_train = np.concatenate((F_train,np.zeros((n,F_train.shape[1],F_train.shape[2]))))
    T_train = np.concatenate((T_train,np.zeros((n,T_train.shape[1],T_train.shape[2]))))
if (F_val.shape[0] % batch_size) != 0 :
    n = batch_size - F_val.shape[0] % batch_size
    F_val = np.concatenate((F_val,np.zeros((n,F_val.shape[1],F_val.shape[2]))))
    tmp = np.zeros((n,T_val.shape[1],T_val.shape[2]))
    tmp[:,:,0] = 1
    T_val = np.concatenate((T_val,tmp))
    
mask_train = ~np.all(F_train==0,axis=2)
mask_val = ~np.all(F_val==0,axis=2)
mask_test = ~np.all(F_test==0,axis=2)

#%% Mask weighting

mask_train = mask_train.astype('float')
mask_val = mask_val.astype('float')
mask_test = mask_test.astype('float')

def mask_increasing_weight(mask):
    out_mask = mask
    
    for i,sample in enumerate(mask):
        if len(np.argwhere(sample==0)) > 0:
            len_sample = np.argwhere(sample==0)[0] - 1
        else:
            len_sample = sample.shape[0]    
        len_half_sample = np.ceil(len_sample/2)
        s = np.arange(len_half_sample) / (len_half_sample - 1)
        s = s * .9 + .1
        out_mask[i,range(s.shape[0])] = s
    return out_mask

mask_train = mask_increasing_weight(mask_train)
mask_val = mask_increasing_weight(mask_val)
mask_test = mask_increasing_weight(mask_test)

#%% DEFINE NETWORK

reset_random()

def create_net_cnn(batch_size):
    inputs = Input(batch_shape=(batch_size,F_train.shape[1],F_train.shape[2]))
    x = Dense(512,activation='tanh')(inputs)
    x = Conv1D(100, 5, padding='same', activation='relu')(x)
    x = GaussianNoise(.2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(100, 5, padding='same', activation='relu')(x)
    x = GaussianNoise(.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(T_train.shape[2], activation='softmax')(x)

    net = Model(inputs, outputs)
    return net

net = create_net_cnn(batch_size=batch_size)
net_predict = create_net_cnn(batch_size=1)

opt = keras.optimizers.SGD(lr = .001, momentum = 0.9)

net.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'],
            sample_weight_mode='temporal')


# Show network summary:
net.summary()

##%% TRAIN
# Train and time:
timestart = time()
history = net.fit(x=F_train,y=T_train,
                  sample_weight=mask_train,
                  validation_data=(F_val,T_val,mask_val),
                  epochs=1000,
                  batch_size=batch_size,
                  callbacks=[EarlyStopping('val_loss',patience=12)],
                  verbose=1)

print('Training time: %.1f seconds.' % (time() - timestart))

# Copy weights to final model
net_predict.set_weights(net.get_weights())
net_predict.compile(loss='categorical_crossentropy',optimizer='adam')

timestart = time()
y_train = net_predict.predict(F_train,batch_size=1)
y_val = net_predict.predict(F_val,batch_size=1)
y_test = net_predict.predict(F_test,batch_size=1)

def ts_evaluate(y,t,mask):
    # takes as input the model prediction y, the target classt and the padding mask
    # y,t are 3D arrays with shape (sample,timesteps,features)
    # mask is (sample,timesteps)
    
    yind = np.argmax(y,axis=2)
    yind = yind.astype('float')
    yind[~mask] = np.nan
    
    tind = np.argmax(t,axis=2)
    tind = tind.astype('float')
    tind[~mask] = np.nan
    
    s = np.sum(mask,axis=1) - 1 # last frame of gesture
    ind = np.dot(s[:,np.newaxis],np.array([[0.25,0.5,0.75,1.0],]))
    ind = np.ceil(ind).astype('int')
    
    truf = (yind == tind)
    truf[~mask] = np.nan
    
    tp = np.array([ truf[np.arange(truf.shape[0]),ind[:,0]],
                    truf[np.arange(truf.shape[0]),ind[:,1]],
                    truf[np.arange(truf.shape[0]),ind[:,2]],
                    truf[np.arange(truf.shape[0]),ind[:,3]] ]).transpose()
    tp = tp[mask[:,0]]
    acc = np.sum(tp,axis=0) / tp.shape[0]
    return acc

acc_train = ts_evaluate(y_train,T_train,mask_train > 0) * 100
acc_val = ts_evaluate(y_val,T_val,mask_val > 0) * 100
acc_test = ts_evaluate(y_test[u_test!=8],T_test[u_test!=8],mask_test[u_test!=8] > 0) * 100
acc_test8 = ts_evaluate(y_test[u_test==8],T_test[u_test==8],mask_test[u_test==8] > 0) * 100

#acc_train = []
#acc_train.append(net.evaluate(F_train[0:-1:4],T_train[0:-1:4])[1] * 100)
#acc_train.append(net.evaluate(F_train[1:-1:4],T_train[1:-1:4])[1] * 100)
#acc_train.append(net.evaluate(F_train[2:-1:4],T_train[2:-1:4])[1] * 100)
#acc_train.append(net.evaluate(F_train[3:-1:4],T_train[3:-1:4])[1] * 100)
#acc_val = []
#acc_val.append(net.evaluate(F_val[0:-1:4],T_val[0:-1:4])[1] * 100)
#acc_val.append(net.evaluate(F_val[1:-1:4],T_val[1:-1:4])[1] * 100)
#acc_val.append(net.evaluate(F_val[2:-1:4],T_val[2:-1:4])[1] * 100)
#acc_val.append(net.evaluate(F_val[3:-1:4],T_val[3:-1:4])[1] * 100)
#acc_test = []
#acc_test.append(net.evaluate(F_test[0:-1:4],T_test[0:-1:4])[1] * 100)
#acc_test.append(net.evaluate(F_test[1:-1:4],T_test[1:-1:4])[1] * 100)
#acc_test.append(net.evaluate(F_test[2:-1:4],T_test[2:-1:4])[1] * 100)
#acc_test.append(net.evaluate(F_test[3:-1:4],T_test[3:-1:4])[1] * 100)
print('Testing time: %.1f seconds.' % (time() - timestart))

print('TRAIN: %.1f, %.1f, %.1f, %.1f' % (acc_train[0],acc_train[1],acc_train[2],acc_train[3]))
print('  VAL: %.1f, %.1f, %.1f, %.1f' % (acc_val[0],acc_val[1],acc_val[2],acc_val[3]))
print('TEST: %.1f, %.1f, %.1f, %.1f' % (acc_test[0],acc_test[1],acc_test[2],acc_test[3]))
print('TEST8: %.1f, %.1f, %.1f, %.1f' % (acc_test8[0],acc_test8[1],acc_test8[2],acc_test8[3]))

