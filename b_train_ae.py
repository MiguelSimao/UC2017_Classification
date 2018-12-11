#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

SCRIPT TO TEST NOVELTY DETECTION FUNCTIONALITY


@author: simao
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing

np.random.seed(1337)
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import losses

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
#X = X[np.invert(outlierInd)]
#T = T[np.invert(outlierInd)]
#U = U[np.invert(outlierInd)]


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

# One hot-encoding
#enc = preprocessing.OneHotEncoder(sparse=False).fit(t_train)
#t_train = enc.transform(t_train)
#t_val = enc.transform(t_val)


#%% TRAIN FFNN

lr = 0.05
momentum = 0.9


inputs = Input(shape=(X.shape[1],))

x = GaussianNoise(1.0)(inputs)

x = Dense(25, activation='linear')(x)

encoded = Dense(2, activation='linear')(x)

x = Dense(25, activation='linear')(encoded)

decoded = Dense(X_train.shape[1], activation='linear')(x)

autoencoder = Model(inputs,decoded,name='Autoencoder')
encoder = Model(inputs, encoded)

# Optimizer
sgd = SGD(lr=lr,
          momentum=momentum,
          nesterov=True)



es = EarlyStopping(monitor='val_loss', patience=12)
    
autoencoder.compile(optimizer=sgd,
        loss=losses.mean_absolute_error,
        metrics=['mae'])

autoencoder.fit(x=X_train, y=X_train,
        validation_data=(X_val,X_val),
        epochs=50,
        callbacks=[],
        verbose=1)

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
Y = autoencoder.predict(Xp)
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
