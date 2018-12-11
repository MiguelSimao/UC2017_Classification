#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

SCRIPT TO TEST NOVELTY DETECTION FUNCTIONALITY


@author: simao
"""

import h5py
import numpy as np
import sys
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

np.random.seed(1337)
from keras.models import Model
from keras.layers import Input, Dense, Lambda, GaussianNoise
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import losses
from keras import metrics
from keras import backend as K

dir_dataset_sg = './dataset/SG24_dataset.h5'
dir_dataset_dg = './dataset/DG10_dataset.h5'

# Open H5 files to read
f1 = h5py.File(dir_dataset_sg,'r')
f2 = h5py.File(dir_dataset_dg,'r')


# Load static gesture data set
X = f1['Predictors']
T = f1['Target']
U = f1['User']

X = np.array(X).transpose()
T = np.array(T).transpose()
U = np.array(U).transpose()
T = T[:,0]
U = U[:,0]

# Shuffle dataset
np.random.seed(0)
indShuf = np.random.permutation(X.shape[0])
X = X[indShuf]
T = T[indShuf]
U = U[indShuf]
X[np.isnan(X)] = 0

# Dataset statistics
num_users = np.unique(U).shape[0]

#%% FEATURE EXTRACTION
# Variable selection
def variable_subset(X):
    # X is a numpy array with data Mx29
    output_index_list = []
    output_index_list += range(5,29)
    return X[:,output_index_list]

X = variable_subset(X)


#%% NOVELTY DETECTION SPECIFIC PREPROCESSING
# Change of classes 19+ to outlier (Classes 1-18 are gestures, 19,20,21,22,23,24(,25) are outliers)
# Separate the outliers (unsupervised learning)

inlierInd = np.isin(T,[6])
outlierInd = np.invert(inlierInd)
Xin = X[inlierInd]
Tin = T[inlierInd]
Uin = U[inlierInd]
T[inlierInd] = 1
T[outlierInd] = 2

#%% SET SPLITTTING
# Data splitting : all -> train and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, val_index in sss.split(Xin,Tin):
    X_train, X_val = Xin[train_index], Xin[val_index]
    t_train, t_val = Tin[train_index], Tin[val_index]
    u_train, u_val = Uin[train_index], Uin[val_index]
    

#%% FEATURE EXTRACTION
# Transformations
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


#%% DEFINE NEURAL NETWORK

lr = 0.05
momentum = 0.9
original_dim = X.shape[1]
latent_dim = 2
intermediate_dim = 15
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = K.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

decoder_h = Dense(intermediate_dim, activation='relu')
decode_mean = Dense(original_dim, activation='relu')
h_decoded = decoder_h(z)
x_decoded_mean = decode_mean(h_decoded)

# Instantiate VAE model
vae = Model(x, x_decoded_mean)

encoder = Model(x, z_mean)

#%% TRAIN NETWORK

# Define VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='sgd',loss=None)
vae.summary()

vae.fit(X_train,
        shuffle=True,
        epochs=50,
        validation_data=(X_val,None))
        

def MAE(X,Y):
    # Ensure X and Y are numpy arrays NxD
    # Based on the L2 distance
    # Both have the same dimension
    np.testing.assert_equal(X.shape,Y.shape,err_msg='MAE:Input matrices have different shapes.')
    
    L = np.abs(X - Y)
    L = np.mean(L,axis=1)
    return L

#%% PLOTTING RESULTS
    
Xp = scaler.transform(X)
Y = vae.predict(Xp)
L = MAE(Xp,Y)

x = np.arange(Xp.shape[0])

plt.figure()
# Plot inliers on dataset
plt.scatter(x[inlierInd],L[inlierInd],s=8,c='b',marker='.')
# Plot outliers on dataset
plt.scatter(x[outlierInd],L[outlierInd],s=8,c='r',marker='.')


Y = encoder.predict(Xp)

plt.figure()
plt.scatter(Y[inlierInd,0],Y[inlierInd,1],s=8,c='b',marker='.')
plt.scatter(Y[outlierInd,0],Y[outlierInd,1],s=8,c='r',marker='.')
