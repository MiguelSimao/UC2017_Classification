import h5py
import numpy as np
from time import time

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ENSURE REPRODUCIBILITY ######################################################
import os
import random
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(12345)
###############################################################################


#%% DATA LOADING

#dir_dataset_sg = '../dataset/SG24_dataset_euler.h5'
dir_dataset_sg = '../dataset/SG24_dataset_euler.h5'

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

# Dataset statistics
num_users = np.unique(U).shape[0]
for u in np.unique(U):
    print('User %i: %i samples out of %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))


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
T_train = enc.transform(t_train)
T_val = enc.transform(t_val)
T_test = enc.transform(t_test)


#%% DEFINE MODELS

names = [
        "Nearest Neighbors",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
#        "AdaBoost",
        "Gaussian Process",
        "Naive Bayes",
        "QDA"
        ]

classifiers = [
        KNeighborsClassifier(5, p=3, algorithm='auto'),
        SVC(kernel="rbf", C=0.5),
        DecisionTreeClassifier(max_depth=30),
        RandomForestClassifier(40, max_depth=10),
#        AdaBoostClassifier(GaussianNB(), 50),
        GaussianProcessClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
        ]

scores = []

for name, clf in zip(names, classifiers):
    print(' ::: %s :::' % (name))
    time_start = time()
    clf.fit(X_train, t_train.ravel())
    time_elapsed = time() - time_start
    print('Training time: %.1f s' % time_elapsed)
    
    time_start = time()
    train_score = clf.score(X_train,t_train) * 100
    val_score = clf.score(X_val,t_val) * 100
    test_score = clf.score(X_test[u_test!=8],t_test[u_test!=8]) * 100
    test_score_8 = clf.score(X_test[u_test==8],t_test[u_test==8]) * 100
    time_elapsed = time() - time_start
    scores.append((train_score,val_score,test_score))
    print('Testing time: %.1f s' % time_elapsed)
    
    print('Accuracies:')
    print('Train: %.2f' % (train_score))
    print('  Val: %.2f' % (val_score))
    print(' Test: %.2f' % (test_score))
    print('Test8: %.2f' % (test_score_8))
    
    
    

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
#print('Training set accuracy: %.1f' % acc_train )
#print('Validation set accuracy: %.1f' % acc_val)
#print('Testing set accuracy: %.1f' % acc_test)
#
##net.save('sg_classification/sg24_net.h5')
#
#%% PERFORMANCE EVALUATION

"""

def evaluate_sequential(model,X,T,u):
    # Calculates the performance of the classification model
    # Input 1: trained Sequential model
    # Input 2: test data
    # Input 3: target data
    # Input 4: user data
    
    # Target index
    tind = np.argmax(T, axis=1)
    
    # Model output:
    y = model.predict(X)
    
    # Most likely classification:
    yind1 = np.argmax(y, axis=1)
    yscore1 = y[np.arange(len(y)),yind1]
    
    # Second most likely:
    y[np.arange(len(y)),yind1] = 0
    yind2 = np.argmax(y, axis=1)
    yscore2 = y[np.arange(len(y)),yind2]
    
    
    e1 = tind != yind1
#    e2 = tind[e1] != yind2[e1]
    e2 = np.logical_and((tind != yind1),(tind != yind2))
    acc1 = (1 - sum(e1)/len(e1) )*100
    acc2 = (1 - sum(e2)/len(e2) )*100

    # User accuracy
    ulist = np.unique(u)
    uacc = []
    for i in ulist:
        eu = tind[u==i] != yind1[u==i]
        uacc.append( (1 - sum(eu)/len(eu))*100 )

    return acc1, acc2, yscore1, yscore2, uacc, e1

#%% Training evaluation
    
acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_train,t_train,u_train)

tind = np.argmax(t_train,axis=1)
yind = net.predict_classes(X_train)
cm = confusion_matrix(tind,yind)

perf_train = {'Acc1' : acc1,
            'Acc2' : acc2,
            'score1' : yscore1,
            'score2' : yscore2,
            'UserAcc' : uacc,
            'Errors' : e1,
            'ConfMat' : cm}


#%% Validation evaluation
    
acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_val,t_val,u_val)

tind = np.argmax(t_val,axis=1)
yind = net.predict_classes(X_val)
cm = confusion_matrix(tind,yind)

perf_val = {'Acc1' : acc1,
            'Acc2' : acc2,
            'score1' : yscore1,
            'score2' : yscore2,
            'UserAcc' : uacc,
            'Errors' : e1,
            'ConfMat' : cm}

#%% Testing evaluation
    
acc1, acc2, yscore1, yscore2, uacc, e1 = evaluate_sequential(net,X_test,t_test,u_test)

tind = np.argmax(t_test,axis=1)
yind = net.predict_classes(X_test)
cm = confusion_matrix(tind,yind)

perf_test = {'Acc1' : acc1,
             'Acc2' : acc2,
             'score1' : yscore1,
             'score2' : yscore2,
             'UserAcc' : uacc,
             'Errors' : e1,
             'ConfMat' : cm}


"""