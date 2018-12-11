import numpy as np
import scipy.io as sio

from keras.models import Sequential
from keras.layers import Dense, GaussianNoise

from keras.callbacks import EarlyStopping

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import PredefinedSplit

#RANDOM_SEED = 42

def loadData():
    """  """
    mat = sio.loadmat('featuresSG.mat',squeeze_me=True)
    trainX = mat['trainX']
    trainY = mat['trainT']
    valX = mat['valX']
    valY = mat['valT']
    testX = mat['testX']
    testY = mat['testT']

    # Convert into one-hot vectors
    num_labels = len(np.unique(trainY))
    def to_onehot(X,num_labels):
        X = X - 1
        X = np.eye(num_labels)[X]
        return X

    trainY = to_onehot(trainY,num_labels)
    valY   = to_onehot(  valY,num_labels)
    testY  = to_onehot( testY,num_labels)

    return trainX, valX, testX, trainY, valY, testY


trainX, valX, testX, trainY, valY, testY = loadData()

X = np.concatenate((trainX,valX))
Y = np.concatenate((trainY,valY))
val_fold = - np.ones((X.shape[0],))
val_fold[-valX.shape[0]:] = 0

ps = PredefinedSplit(val_fold)

def createModel(neurons1=20,neurons2=100,neurons3=100,noise_strength=0.5,activ_fcn='tanh',init_mode='uniform'):
    model = Sequential()
    model.add(Dense(neurons1,
                    activation=activ_fcn,
                    kernel_initializer=init_mode,
                    input_shape=(trainX.shape[1],)))
    model.add(GaussianNoise(noise_strength))
    
    if neurons2!=0:
        model.add(Dense(neurons2,activation=activ_fcn,kernel_initializer=init_mode))
    elif neurons3!=0:
        model.add(Dense(neurons3,activation=activ_fcn,kernel_initializer=init_mode))
        

    model.add(Dense(trainY.shape[1],activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model

es = EarlyStopping(monitor='loss',min_delta=0,patience=8,mode='min')

# CREATE KERAS MODEL
model = KerasClassifier(build_fn=createModel,epochs=1000,verbose=0)

# define the grid search parameters
neurons1 = list(range(10,101,10))    
neurons2 = list(range(0,101,25))
neurons3 = list(range(0,101,25))
noise_strength = [0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
activ_fcn = ['elu','tanh','sigmoid','relu'] #['elu','tanh','sigmoid']
init_mode = ['lecun_uniform', 'glorot_uniform', 'he_uniform']


param_grid = dict(#neurons1=neurons1,
                  #neurons2=neurons2,
                  #neurons3=neurons3,
                  noise_strength=noise_strength,
                  init_mode=init_mode,
                  activ_fcn=activ_fcn)

fit_params = {'callbacks': [es]}
grid = GridSearchCV(estimator=model,fit_params=fit_params,param_grid=param_grid,cv=ps,
                    refit=False,verbose=3,n_jobs=1)#
grid_result = grid.fit(X,Y)



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

"""
# Build Keras model
model = Sequential()

model.add(Dense(50,activation='tanh',input_shape=(trainX.shape[1],)))
model.add(GaussianNoise(0.5))
model.add(Dense(50))
model.add(Dense(trainY.shape[1],activation='softmax'))


#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb = TensorBoard(log_dir='/tmp/keras_logs/sg_class1')
es = EarlyStopping(monitor='val_loss',min_delta=0,patience=6,mode='min')
plot_model(model, to_file='model.png')

model.fit(trainX,trainY,
          shuffle=True,
          epochs=1000,
          verbose=1,
          validation_data=(valX,valY),
          callbacks=[es])


# Determine predictions:
trainPred = model.predict(trainX)
valPred = model.predict(valX)
testPred = model.predict(testX)

# Save im MATLAB format
sio.savemat('predictionsSG.mat',{'trainPred':trainPred,
                                 'valPred':valPred,
                                 'testPred':testPred})

print(model.evaluate(trainX,trainY,verbose=0))
print(model.evaluate(valX,valY,verbose=0))
print(model.evaluate(testX,testY,verbose=0))
"""