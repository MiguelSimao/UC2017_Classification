import h5py
import numpy as np
import random
import csv
from time import time

from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit, train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import tensorflow as tf



# ENSURE REPRODUCIBILITY ######################################################
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.set_random_seed(123)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
###############################################################################

from keras.models import Sequential
from keras.layers import Dense, GaussianNoise
from keras import regularizers
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


#%% DATA LOADING

dir_dataset_sg = '../dataset/SG24_dataset_euler.h5'
#dir_dataset_sg = './dataset/SG24_dataset_euler.h5'

# Open H5 file to read
file = h5py.File(dir_dataset_sg,'r')


# Load static gesture data set
X = file['Predictors']
T = file['Target']
U = file['User']

X = np.array(X).transpose()
T = np.array(T).transpose()
U = np.array(U).transpose()
U = U[:,0]

# Shuffle dataset
#indShuf = np.random.permutation(X.shape[0])
#X = X[indShuf]
#T = T[indShuf]
#U = U[indShuf]
#X[np.isnan(X)] = 0

# Dataset statistics
num_users = np.unique(U).shape[0]
for u in np.unique(U):
    print('User %i: %i samples out of total %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))


#%% SET SPLITTING (TRAIN + VALIDATION + TEST)
# Data splitting 1 : all -> train and rest
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=123)
#
#for train_index, test_index in sss.split(X,T,groups=U):
#    X_train, X_test = X[train_index], X[test_index]
#    t_train, t_test = T[train_index], T[test_index]
#    u_train, u_test = U[train_index], U[test_index]
#    ind_train, ind_test = train_index, test_index
#    
## Data splitting 2 : test -> validation and test
#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=321)
#
#for val_index, test_index in sss.split(X_test, t_test):
#    X_val, X_test = X_test[val_index], X_test[test_index]
#    t_val, t_test = t_test[val_index], t_test[test_index]
#    u_val, u_test = u_test[val_index], u_test[test_index]
#    ind_val, ind_test = ind_test[val_index], ind_test[val_index]
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
ind_test_8 = np.argwhere(u_test==8)[:,0]


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


def create_model(lr=0.01, momentum=0.1, neurons1=30, neurons2=0, l2reg=0.0005, weight_decay=1e-7,noise=.1):
    
    np.random.seed(1337)
    random.seed(12345)
    tf.set_random_seed(123)

    net = Sequential()
    net.add(Dense(neurons1,
                   input_dim=X_train.shape[1],
                   activation='relu',
                   kernel_regularizer=regularizers.l2(l2reg)
                   ))
    net.add(GaussianNoise(noise))
    if ( neurons2!=0 ):
        net.add(Dense(neurons2,activation='relu'))
    #net.add(Dense(25))
    net.add(Dense(t_train.shape[1],
                   activation='softmax'))


    # Optimizer
    sgd = SGD(lr=lr,
          momentum=momentum,
          decay=weight_decay,
          nesterov=False)
    

    
    net.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return net

es = EarlyStopping(monitor='val_loss', patience=10)


# grid search parameters
lr = [0.001, 0.005, 0.01]
mo = [0.0, 0.1, 0.2, 0.4, 0.6, 0.9]
neurons1 = [20, 25, 40, 50, 100]
neurons2 = [0, 20, 40, 100]
l2reg = [0.0005, 0.001, 0.005]
decay = [1e-8, 1e-7, 1e-6, 1e-5]
noise = [0.1,0.3,0.6,1.0]
param_grid = dict(lr=lr,momentum=mo,neurons1=neurons1,neurons2=neurons2,l2reg=l2reg,weight_decay=decay,noise=noise)

grid = list(ParameterGrid(param_grid))
random.shuffle(grid)

acc_train = []
acc_val = []

for i,param in enumerate(grid):
    print('Iteration %i:' % (i))
    print(param)
    
    acc_current = 0
    for j in range(5):
#        tb = TensorBoard(log_dir='/tmp/tblogs/net_%04i_%i' % (i,j))
        print('Fitting network %i/%i...' % (j+1,5))
        model = create_model(**param)
        model.fit(X_train,t_train,validation_data=(X_val,t_val),epochs=1000,callbacks=[es],verbose=0)
        acc_new = model.evaluate(X_val,t_val,verbose=0)[1]
        if acc_new > acc_current:
            model_new = model
            
    print('Done!\n')
    acc_train.append(model_new.evaluate(X_train,t_train,verbose=0)[1])
    acc_val.append(model_new.evaluate(X_val,t_val,verbose=0)[1])
    print('Training accuracy: %.2f%%' % (acc_train[-1]*100))
    print('Validation accuracy: %.2f%% (Best: %.2f%% (Run %i))\n' % (acc_val[-1]*100,np.max(acc_val)*100,np.argmax(acc_val)))
    model_new.save('./nets/net_%04i.h5' % i)
    
    with open('./traininglog.csv','a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Trial %i' % i])
        csvwriter.writerow([param.keys(),param.values()])
        csvwriter.writerow(['training',acc_train[-1]*100,'validation',acc_val[-1]*100])
    
    
    
    

#grid.fit(X_train,t_train,validation_data=(X_val,t_val),epochs=2000,callbacks=[es,tb],verbose=0)

#net.fit(x=X_train, y=t_train,
#        validation_data=(X_val,t_val),
#        epochs=2000,
#        callbacks=[es,tb],
#        verbose=0)
#
#
#acc_train = net.evaluate(X_train,t_train)[1] * 100
#acc_val = net.evaluate(X_val,t_val)[1] * 100
#acc_test = net.evaluate(X_test,t_test)[1] * 100
#error_ind_val = ind_val[ net.predict_classes(X_val) != np.argmax(t_val,axis=1) ]
#error_ind_val = np.sort(error_ind_val)
#
#print(error_ind_val)
##with open('log.csv',mode='a',newline='\n') as csvfile:
##    csvwriter = csv.writer(csvfile)
##    csvwriter.writerow(error_ind_val.tolist())
#
#print('Training set accuracy: %.2f' % acc_train )
#print('Validation set accuracy: %.2f' % acc_val)
#print('Testing set accuracy: %.2f' % acc_test)

##net.save('sg_classification/sg24_net.h5')
#
##%% PERFORMANCE EVALUATION
#
#
#
#def evaluate_sequential(model,X,T,u):
#    # Calculates the performance of the classification model
#    # Input 1: trained Sequential model
#    # Input 2: test data
#    # Input 3: target data
#    # Input 4: user data
#    
#    # Target index
#    tind = np.argmax(T, axis=1)
#    
#    # Model output:
#    y = model.predict(X)
#    
#    # Most likely classification:
#    yind1 = np.argmax(y, axis=1)
#    yscore1 = y[np.arange(len(y)),yind1]
#    
#    # Second most likely:
#    y[np.arange(len(y)),yind1] = 0
#    yind2 = np.argmax(y, axis=1)
#    yscore2 = y[np.arange(len(y)),yind2]
#    
#    
#    e1 = tind != yind1
##    e2 = tind[e1] != yind2[e1]
#    e2 = np.logical_and((tind != yind1),(tind != yind2))
#    acc1 = (1 - sum(e1)/len(e1) )*100
#    acc2 = (1 - sum(e2)/len(e2) )*100
#
#    # User accuracy
#    ulist = np.unique(u)
#    uacc = []
#    for i in ulist:
#        eu = tind[u==i] != yind1[u==i]
#        uacc.append( (1 - sum(eu)/len(eu))*100 )
#
#    return acc1, acc2, yscore1, yscore2, uacc, e1
#
##%% Training evaluation
#    
#acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_train,t_train,u_train)
#
#tind = np.argmax(t_train,axis=1)
#yind = net.predict_classes(X_train)
#cm = confusion_matrix(tind,yind)
#
#perf_train = {'Acc1' : acc1,
#            'Acc2' : acc2,
#            'score1' : yscore1,
#            'score2' : yscore2,
#            'UserAcc' : uacc,
#            'Errors' : e1,
#            'ConfMat' : cm}
#
#
##%% Validation evaluation
#    
#acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_val,t_val,u_val)
#
#tind = np.argmax(t_val,axis=1)
#yind = net.predict_classes(X_val)
#cm = confusion_matrix(tind,yind)
#
#perf_val = {'Acc1' : acc1,
#            'Acc2' : acc2,
#            'score1' : yscore1,
#            'score2' : yscore2,
#            'UserAcc' : uacc,
#            'Errors' : e1,
#            'ConfMat' : cm}
#
##%% Testing evaluation
#    
#acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_test,t_test,u_test)
#
#tind = np.argmax(t_test,axis=1)
#yind = net.predict_classes(X_test)
#cm = confusion_matrix(tind,yind)
#
#perf_test = {'Acc1' : acc1,
#             'Acc2' : acc2,
#             'score1' : yscore1,
#             'score2' : yscore2,
#             'UserAcc' : uacc,
#             'Errors' : e1,
#             'ConfMat' : cm}


