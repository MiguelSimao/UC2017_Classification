
from sys import stdout
from time import time
import h5py
import numpy as np

import math
from scipy import interpolate
from transforms3d import euler

from sklearn.model_selection import train_test_split
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
from keras.layers import Input, Dense, GaussianNoise, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import utils, regularizers

#%% EXPERIMENT SETTINGS:

#param_feat = 'CI'
param_feat = 'PV'


#%% DATA LOADING

dir_dataset_dg = '../dataset/DG10_dataset_euler.h5'

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

print('Preprocessing data : ')

# Degrees to rads:
X[:,:,3:6] = X[:,:,3:6] / 180 * math.pi

# Angle lifting function (make continuous)
def angle_lift(y, n):
    for i in range(1, n):
        if np.isclose(y[i] - y[i-1], 2*np.pi, atol=0.1):
            # low to high
            y[i] -= 2*np.pi
        elif np.isclose(y[i] - y[i-1], -2*np.pi, atol=0.1):
            y[i] += 2*np.pi

# for each sample in dataset
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


    ### YAW CORRECTION
    r0p = np.matrix(euler.euler2mat(f0[3],0,0,'rzyx')) ** -1
    p0 = np.matrix(f0[:3].reshape((-1,1)))
    
    # for each gesture frame
    for j,frame in enumerate(sample):
        if np.all(frame==0) : break # stop at zero padding start
        p1 = np.matrix(frame[:3].reshape((-1,1)))
        p2 = r0p * (p1 - p0)
        frame[:3] = np.squeeze(p2)
        frame[3] = frame[3] - f0[3]

    # Print progress
    stdout.write('\r% 5.1f%%' % ((i+1)/(X.shape[0])*100))
stdout.write('\n')


#%% SET SPLITTTING
    
# Source indexes:
ind_all = np.arange(X.shape[0])

# All -> 0.7 Train + 0.3 (Val + Test)
ind_train, ind_test = train_test_split(ind_all,
                                       shuffle=True,
                                       stratify=T[ind_all],
                                       test_size=0.3,
                                       random_state=42)
# (Val + Test) -> 0.15 Val + 0.15 Test
ind_val, ind_test = train_test_split(ind_test,
                                     shuffle=True,
                                     stratify=T[ind_test],
                                     test_size=0.5,
                                     random_state=41)

## Set user 8 aside ###########################################################
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
###############################################################################

X_train = X[ind_train,:] # predictors (raw)
X_val   = X[ind_val,:]
X_test  = X[ind_test,:]
t_train = T[ind_train] # prediction targets
t_val   = T[ind_val]
t_test  = T[ind_test]
u_train = U[ind_train] # user numbers
u_val   = U[ind_val]
u_test  = U[ind_test]


#%% FEATURE EXTRACTION

# Variable scaling (get parameters on training data):
tmpX = X_train.reshape((-1,X_train.shape[2]))
tmpX = tmpX[~np.all(tmpX==0,1),] # DO NOT USE PADDING TO STANDARDIZE
scaler = preprocessing.StandardScaler().fit(tmpX)

# Scaling function:
def standardize(X,scaler):
    old_shape = X.shape
    X = X.reshape((-1,X.shape[2]))
    padIdx = np.all(X==0,axis=1)
    X = scaler.transform(X)
    X[padIdx] = 0
    X = X.reshape(old_shape)
    return X

# Apply scaling:
X_train = standardize(X_train,scaler)
X_val   = standardize(X_val,scaler)
X_test  = standardize(X_test,scaler)

# One-hot encoding:
T_train = utils.to_categorical(T[ind_train]-1,10)
T_val   = utils.to_categorical(T[ind_val]-1,10)
T_test  = utils.to_categorical(T[ind_test]-1,10)


#%% EXTRACT FEATURES

# FEATURE SET 1 (interpolated) EXTRACTION FUNCTION:
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

# FEATURE SET 2 (PCA / PV) EXTRACTION FUNCTION:
def princvecfeatures(X):
    Xf = np.zeros((X.shape[0],X.shape[2]))
    for i,x in enumerate(X):
        # Remove padding (zero-rows)
        x = x[~np.all(x==0,axis=1)]
        pca = decomposition.PCA().fit(x)
        Xf[i] = pca.components_[0]
    return Xf

# Select feature set
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

# Feature scaling
scaler_features = preprocessing.StandardScaler().fit(F_train)
F_train = scaler_features.transform(F_train)
F_val = scaler_features.transform(F_val)
F_test = scaler_features.transform(F_test)


#%% DEFINE NETWORK

# RESET RANDOM GENERATION
reset_random()

###### MODEL DEFINITION #######################################################
# As a function of the feature set:
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
###############################################################################

# Model compilation:
net.compile(optimizer=SGD(0.01, decay=1e-6),
            loss='categorical_crossentropy',
            metrics=['acc'])

## FIT CLASSIFIER #############################################################
time_start = time()
history = net.fit(x=F_train,y=T_train,
                  validation_data=(F_val,T_val),
                  batch_size=128,
                  epochs=1000,
                  callbacks=[EarlyStopping('val_loss',patience=12)],
                  verbose=1)
time_elapsed = time() - time_start
print('Training time: %.1f s' % time_elapsed)


## EVALUATE ###################################################################
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