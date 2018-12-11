import h5py
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

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

from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


#%% DATA LOADING

dir_dataset_sg = '../dataset/SG24_dataset_euler.h5'

# Open H5 file to read
file = h5py.File(dir_dataset_sg,'r')

# Load static gesture data set
X = file['Predictors']
T = file['Target']
U = file['User']

X = np.array(X).transpose()
T = np.array(T).transpose()
U = np.array(U).transpose()[:,0]

# Dataset statistics
num_users = np.unique(U).shape[0]
for u in np.unique(U):
    print('User %i: %i samples out of total %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))


#%% SET SPLITTING (TRAIN + VALIDATION + TEST)

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
    
# Variable selection
def variable_subset(X):
    # X is a numpy array with data Mx28
    output_index_list = [4]
    output_index_list += range(6,28)
    return X[:,output_index_list]

X_train = variable_subset(X_train)
X_val = variable_subset(X_val)
X_test = variable_subset(X_test)

# Variable scaling (fit on train set)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# One hot-encoding
enc = preprocessing.OneHotEncoder(sparse=False).fit(t_train)
t_train = enc.transform(t_train)
t_val = enc.transform(t_val)
t_test = enc.transform(t_test)


#%% TRAIN NEURAL NETWORK

## MODEL DEFINITION
def create_model(lr=0.01, momentum=0.1, neurons1=30, neurons2=0, l2reg=0.0005, weight_decay=1e-7,noise=.1):
    
    # Fix random generation
    np.random.seed(1337)
    random.seed(12345)
    tf.set_random_seed(123)

    # MODEL DEFINITION
    net = Sequential()
    net.add(Dense(neurons1,
                   input_dim=X_train.shape[1],
                   activation='relu',
                   kernel_regularizer=regularizers.l2(l2reg),
                   ))

    net.add(GaussianNoise(noise))
    if ( neurons2!=0 ):
        net.add(Dense(neurons2,
                      activation='relu',
                      ))

    net.add(Dense(t_train.shape[1],
                   activation='softmax'))

    # OPTIMIZER
    sgd = SGD(lr=lr,
          momentum=momentum,
          decay=weight_decay,
          nesterov=False)
    
    # COMPILE MODEL
    net.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return net

es = EarlyStopping(monitor='val_loss', patience=10)


# Randomized rid search parameters:
param = dict(zip(['l2reg', 'lr', 'momentum', 'neurons1', 'neurons2', 'noise', 'weight_decay'],
                 [0.005, 0.001, 0.4, 200, 200, 0.6, 1e-07]))

# INSTANCIATE MODEL ###########################################################
model = create_model(**param)

# Fit network
time_start = time()
model.fit(X_train,t_train,
          batch_size=32,
          validation_data=(X_val,t_val),
          epochs=2000,
          callbacks=[es],
          verbose=1)

time_elapsed = time() - time_start
print('Training time: %.1f s' % (time_elapsed))

## EVALUATE ###################################################################

time_start = time()
acc_train = model.evaluate(X_train,t_train)[1]*100
acc_val = model.evaluate(X_val,t_val)[1]*100
acc_test = model.evaluate(X_test[u_test!=8],t_test[u_test!=8])[1]*100
acc_test_8 = model.evaluate(X_test[u_test==8],t_test[u_test==8])[1] * 100

time_elapsed = time() - time_start
print('Testing time: %.1f s' % time_elapsed)


print('Accuracies:')
print('Train: %.1f' % (acc_train))
print('  Val: %.1f' % (acc_val))
print(' Test: %.1f' % (acc_test))
print('Test8: %.1f' % (acc_test_8))
