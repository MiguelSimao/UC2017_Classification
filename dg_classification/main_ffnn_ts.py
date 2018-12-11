# -*- coding: utf-8 -*-
"""

SCRIPT TO TEST DG CLASSIFICATION


@date: 2018.04.10
@author: Miguel Simão (miguel.simao@uc.pt)
"""

# IMPORTS

from time import time
from sys import stdout
import h5py
import numpy as np
#from matplotlib import pyplot as plt

import math
from transforms3d import euler

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition
#from sklearn.metrics import confusion_matrix


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
from keras import utils, regularizers
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

#%% LOAD SOURCE DATA

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
    
#ind_all = np.asarray(range(X.shape[0]))
#
## Data splitting 1 : all -> train and rest
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(ind_all,T)
#ind_train,ind_test = next(sss)
#
## Data splitting 2 : test -> validation and test
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(ind_test,T[ind_test])
#i1, i2 = next(sss)
#ind_val = ind_test[i1]
#ind_test = ind_test[i2]
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


#%% PRESCALING FEATURE EXTRACTION

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

F_train = tspvfeatures(X_train)
F_val   = tspvfeatures(X_val)
F_test  = tspvfeatures(X_test)

# Encode targets
T_train = tsonehotencoding(T[ind_train], X_train, 10)
T_val   = tsonehotencoding(T[ind_val], X_val, 10)
T_test  = tsonehotencoding(T[ind_test], X_test, 10)

def tsunroll(X,T):
    x = []
    t = []
    for i,sample in enumerate(X):
        ind_include = ~np.all(sample==0,axis=1)
        sample = sample[ind_include]
        x.append(sample)
        t.append(T[i][ind_include])
    return np.concatenate(x), np.concatenate(t)

def tsunroll_testsubset(X,T):
    #function to unroll the test set. we are only interested at a limited
    #subset of timesteps (25, 50, 75 and 100% of gesture completion)
    x = []
    t = []
    # For each sample
    for i,sample in enumerate(X):
        ind_include = ~np.all(sample==0,axis=1)
        sample = sample[ind_include]
        ind_subset = (sample.shape[0] - 1) * np.asarray([0.25,0.5,0.75,1.0])
        ind_subset = np.ceil(ind_subset).astype(np.int)

        x.append(sample[ind_subset])
        t.append(T[i][ind_subset])
    return np.concatenate(x), np.concatenate(t)

# Feature scaling
tmpX = F_train.reshape((-1,F_train.shape[2]))
tmpX = tmpX[~np.all(tmpX==0,1),] # DO NOT USE PADDING TO STANDARDIZE
scaler_features = preprocessing.StandardScaler().fit(tmpX)
F_train = standardize(F_train, scaler_features)
F_val = standardize(F_val, scaler_features)
F_test = standardize(F_test, scaler_features)

### Change data structure for training and testing ###
# Testing subset
FT_train,TT_train = tsunroll_testsubset(F_train,T_train)
FT_val,TT_val = tsunroll_testsubset(F_val,T_val)
FT_test,TT_test = tsunroll_testsubset(F_test[u_test!=8],T_test[u_test!=8])
FT_test8,TT_test8 = tsunroll_testsubset(F_test[u_test==8],T_test[u_test==8])

# Training
F_train,T_train = tsunroll(F_train,T_train)
F_val,T_val = tsunroll(F_val,T_val)
F_test,T_test = tsunroll(F_test,T_test)
###############################################################################


#%% DEFINE NETWORK

# CONTROL RANDOM GENERATION
reset_random()
new_seed = int(time())

# 
inputs = Input(shape=(F_train.shape[1],))
x = Dense(512, activation='tanh')(inputs)
x = GaussianNoise(0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='tanh')(x)
x = GaussianNoise(0.05)(x)
x = BatchNormalization()(x)
outputs = Dense(T_train.shape[1], activation='softmax')(x)


net = Model(inputs, outputs, name='DG_FFNN')

#opt = SGD(lr=0.01, decay=1e-7)
opt = Adam(0.0001, decay=1e-6)

net.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'])

# Show network summary:
net.summary()

# Train and time:
timestart = time()
history = net.fit(x=F_train,y=T_train,
                  validation_data=(F_val,T_val),
                  batch_size=256,
                  epochs=1000,
                  callbacks=[EarlyStopping('val_loss',patience=8)],
                  verbose=1)
print('Training time: %.1f seconds.' % (time() - timestart))

# Evaluate and score
timestart = time()
acc_train = []
acc_train.append(net.evaluate(FT_train[0::4],TT_train[0::4])[1] * 100)
acc_train.append(net.evaluate(FT_train[1::4],TT_train[1::4])[1] * 100)
acc_train.append(net.evaluate(FT_train[2::4],TT_train[2::4])[1] * 100)
acc_train.append(net.evaluate(FT_train[3::4],TT_train[3::4])[1] * 100)
acc_val = []
acc_val.append(net.evaluate(FT_val[0::4],TT_val[0::4])[1] * 100)
acc_val.append(net.evaluate(FT_val[1::4],TT_val[1::4])[1] * 100)
acc_val.append(net.evaluate(FT_val[2::4],TT_val[2::4])[1] * 100)
acc_val.append(net.evaluate(FT_val[3::4],TT_val[3::4])[1] * 100)
acc_test = []
acc_test.append(net.evaluate(FT_test[0::4],TT_test[0::4])[1] * 100)
acc_test.append(net.evaluate(FT_test[1::4],TT_test[1::4])[1] * 100)
acc_test.append(net.evaluate(FT_test[2::4],TT_test[2::4])[1] * 100)
acc_test.append(net.evaluate(FT_test[3::4],TT_test[3::4])[1] * 100)
acc_test8 = []
acc_test8.append(net.evaluate(FT_test8[0::4],TT_test8[0::4])[1] * 100)
acc_test8.append(net.evaluate(FT_test8[1::4],TT_test8[1::4])[1] * 100)
acc_test8.append(net.evaluate(FT_test8[2::4],TT_test8[2::4])[1] * 100)
acc_test8.append(net.evaluate(FT_test8[3::4],TT_test8[3::4])[1] * 100)

print('Testing time: %.1f seconds.' % (time() - timestart))

print('TRAIN: %.1f | %.1f | %.1f | %.1f' % (acc_train[0],acc_train[1],acc_train[2],acc_train[3]))
print('  VAL: %.1f | %.1f | %.1f | %.1f' % (acc_val[0],acc_val[1],acc_val[2],acc_val[3]))
print(' TEST: %.1f | %.1f | %.1f | %.1f' % (acc_test[0],acc_test[1],acc_test[2],acc_test[3]))
print('TEST8: %.1f | %.1f | %.1f | %.1f' % (acc_test8[0],acc_test8[1],acc_test8[2],acc_test8[3]))
print('Seed: %i' % new_seed)


#%% PLOT F_TEST


from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.rc('legend', edgecolor=(0,0,0),fancybox=False)
plt.rc('lines', markeredgewidth=0.5,linewidth=0.5)


x = decomposition.PCA(n_components=2).fit(FT_train).transform(FT_train)
#x = TSNE(n_components=2, early_exaggeration=8,init='pca').fit_transform(F_test)

t = np.argmax(TT_train,axis=1)
k = ['$J_{0.25}$', '$J_{0.50}$', '$J_{0.75}$', '$J_{1.00}$']

cmap = plt.get_cmap('tab10',10)

f = plt.figure(figsize=(7.12,1.8))

for i in range(4):
    if i==0:
        ax = plt.subplot(1,4,i+1)
        plt.ylabel('PC2')
        plt.xlabel('PC1')
    else:
        ax = plt.subplot(1,4,i+1,sharex=ax,sharey=ax)
        ax.set_yticklabels=[]
    
    xi = x[i::4]
    ti = t[i::4]
    for j in range(10):
        xij = xi[ti==j]
        plt.scatter(xij[:,0],xij[:,1],s=12,c=cmap.colors[j],edgecolors='k',linewidths=0.5,alpha=0.8)
    
    plt.text(0.98,0.98,k[i], horizontalalignment='right',verticalalignment='top', transform=ax.transAxes)
    
lg_labels = np.unique(ti+1).tolist()
lg = ax.legend(lg_labels,markerfirst=True,ncol=11,handletextpad=0.2,columnspacing=0.3,borderpad=.1,
               bbox_to_anchor=(0.7,-0.1),loc='upper right')
lg.get_frame().set_linewidth(0.5)

f.tight_layout(pad=0.0)
f.savefig('test.pdf',bbox_inches='tight',pad_inches=0.0,dpi=300)
f.savefig('test.png',bbox_inches='tight',pad_inches=0.0,dpi=300)
