import h5py
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ENSURE REPRODUCIBILITY ######################################################
import os
import random
def reset_random():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1337)
    random.seed(12345)
###############################################################################


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
U = np.array(U).transpose()
U = U[:,0]

# Dataset statistics
num_users = np.unique(U).shape[0]
for u in np.unique(U):
    print('User %i: %i samples out of %i (%.1f%%)' % (u, sum(U==u), len(U), sum(U==u)/len(U)*100))


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


for name, clf in zip(names, classifiers):
    
    # RESET RANDOM GENERATION:
    reset_random()
    
    ## FIT CLASSIFIER #########################################################
    print(' ::: %s :::' % (name))
    time_start = time()
    clf.fit(X_train, t_train.ravel())
    time_elapsed = time() - time_start
    print('Training time: %.1f s' % time_elapsed)
    
    ## EVALUATE ###############################################################
    time_start = time()
    train_score = clf.score(X_train,t_train) * 100
    val_score = clf.score(X_val,t_val) * 100
    test_score = clf.score(X_test[u_test!=8],t_test[u_test!=8]) * 100
    test_score_8 = clf.score(X_test[u_test==8],t_test[u_test==8]) * 100
    time_elapsed = time() - time_start
    
    print('Testing time: %.1f s' % time_elapsed)
    
    print('Accuracies:')
    print('Train: %.2f' % (train_score))
    print('  Val: %.2f' % (val_score))
    print(' Test: %.2f' % (test_score))
    print('Test8: %.2f' % (test_score_8))
    
    
