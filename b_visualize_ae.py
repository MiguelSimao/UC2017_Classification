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
from sklearn.preprocessing import StandardScaler


np.random.seed(1337)
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import losses

dir_dataset_sg = '/media/linuxdata/simao/Drive/PhD/0-Datasets/UC2017/SG25/SG25_dataset.h5'


# Open H5 files to read
f = h5py.File(dir_dataset_sg,'r')

# Load static gesture data set
X = f['Predictors']
T = f['Target']
U = f['User']

X = np.array(X).transpose()
T = np.array(T).transpose()[:,0]
U = np.array(U).transpose()[:,0]

# Limit number of users (subset)
#usubset = np.unique(U)
usubset = [1]
uind = np.isin(U,usubset)
X, T, U = X[uind], T[uind], U[uind]

# Limit classes (subset)
#tsubset = np.unique(T)
tsubset = [1,2,3,4,5,6]
tind = np.isin(T,tsubset)
X, T, U = X[tind], T[tind], U[tind]

#%% FEATURE EXTRACTION
# Variable selection
#def variable_subset(X):
#    # X is a numpy array with data Mx29
#    output_index_list = []
#    output_index_list += range(5,29)
#    return X[:,output_index_list]
#
#X = variable_subset(X)


#%% NOVELTY DETECTION SPECIFIC PREPROCESSING
# Change of classes 19+ to outlier (Classes 1-18 are gestures, 19,20,21,22,23,24(,25) are outliers)
# Separate the outliers (unsupervised learning)

#outlierInd = np.isin(T,[4])
outlierInd = np.isin(T,tsubset[0:1])
inlierInd = ~outlierInd
Xin, Tin, Uin = X[inlierInd], T[inlierInd], U[inlierInd]


#%% SET SPLITTTING
# Data splitting : all -> train and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, val_index in sss.split(Xin,Tin):
    X_train, X_val = Xin[train_index], Xin[val_index]
    t_train, t_val = Tin[train_index], Tin[val_index]
    u_train, u_val = Uin[train_index], Uin[val_index]
    

#%% FEATURE EXTRACTION
# Transformations
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# One hot-encoding
#enc = preprocessing.OneHotEncoder(sparse=False).fit(t_train)
#t_train = enc.transform(t_train)
#t_val = enc.transform(t_val)


#%% TRAIN FFNN

lr = 0.05
momentum = 0.9


# Input tensor:
inputs = Input(shape=(X.shape[1],))
# Add Gaussian noise to input data:
x = GaussianNoise(0.0)(inputs)
# Hidden layer 1:
x = Dense(50, activation='linear')(x)
# Hidden layer 2:
encoded = Dense(3, activation='linear')(x)
# Hidden layer 3:
x = Dense(50, activation='linear')(encoded)
# Output layer:
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
        epochs=200,
        callbacks=[es],
        verbose=1)

#%%

def MAE(Xin,Y):
    # Ensure X and Y are numpy arrays NxD
    # Based on the L2 distance
    np.testing.assert_equal(Xin.shape[1],Y.shape[1],err_msg='MAE:Input matrices have different dimensionality (axis 1).')
    
    L = []
    for row in Y:
        R = np.repeat(row[None,:],Xin.shape[0],axis=0)
        L0 = Xin - R
        L0 = np.sum(L0**2,axis=1)
        L0 = np.min(L0)
        L.append( L0 )
        
    return np.asarray(L)

#%% PLOTTING RESULTS
    
Xp = scaler.transform(X)
#Y = autoencoder.predict(Xp)
Y0 = encoder.predict(X_train)
Y = encoder.predict(Xp)
L = MAE(Y0,Y)

x = np.arange(Xp.shape[0])

plt.figure()
# Plot inliers on dataset
plt.scatter(x[inlierInd],L[inlierInd],s=8,c='b',marker='.',label='Inliers')
# Plot outliers on dataset
plt.scatter(x[outlierInd],L[outlierInd],s=8,c='r',marker='.',label='Outliers')
plt.ylabel('MAE')
plt.xlabel('Sample')
plt.legend(loc=0)

#Y = encoder.predict(Xp)
#
#plt.figure()
#plt.scatter(Y[inlierInd,0],Y[inlierInd,1],s=8,c='b',marker='.',label='Inliers')
#plt.scatter(Y[outlierInd,0],Y[outlierInd,1],s=8,c='r',marker='.',label='Outliers')
#plt.ylabel('X2')
#plt.xlabel('X1')
#plt.legend(loc=0)

