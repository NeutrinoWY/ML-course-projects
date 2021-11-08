# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l1_l2, l1 ,l2
from keras.wrappers.scikit_learn import KerasClassifier
import string
from keras import optimizers
from keras.callbacks import EarlyStopping

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def string_vectorizer(strng, alphabet=string.ascii_uppercase):
    vector = [[0 if char != letter else 1 for char in alphabet] for letter in strng]
    vector = np.array(vector).flatten()
    #print(vector)
    return vector

def preprocessX(X):
    X_new = list(map(string_vectorizer,  X.tolist()))    # map(fun, iter)
    #print(X_new)
    return np.array(X_new)

def bestThreshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    best_threshold = 0
    best_f_score = 0

    for i, threshold in enumerate(thresholds):
        f_score = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        if f_score > best_f_score:
            best_threshold = threshold
            best_f_score = f_score
    print("best threshold: %.5f, best_f_score: %.5f" % (best_threshold, best_f_score))
    return best_threshold, best_f_score

def trained_model(X,y,hyp,epoch=13,train_epoch=False):
    """
    input: 
    training data X, y; 
    dictionary of hyperparameters, including number of neuros of layers, paramemeters of l1, l2 penalties
    For instance: hyp = {"layers":[500,500,500], 'l1':0, "l2": 0.00001}
    output: 
    ANN model with the hyperparameters trained by training data, best stop epochs
    """

    model = Sequential()
    for i, neurons in enumerate(hyp["layers"]):
        if i == 0:   # input layer
                # 'he_uniform': truncated normal distribution N(0, sqrt(2/n), n is input units in the weight tensor.
            model.add(Dense(neurons, kernel_initializer='he_uniform', input_dim=X.shape[1], kernel_regularizer=l1_l2(l1=hyp["l1"], l2=hyp["l2"]*neurons/1000)))
        else:
            model.add(Dense(neurons,kernel_initializer='he_uniform', kernel_regularizer=l1_l2(l1=hyp["l1"], l2=hyp["l2"]*neurons/1000))) 
        model.add(BatchNormalization()) # normalize data in each batch, accelerate the learning
        model.add(Activation('relu'))   # add activation layer after batch normalization
        model.add(Dropout(rate=0.5))   # drop out some neuros from training, keep the weights of those neuros unchange,avoid overfitting
    model.add(Dense(1, activation='sigmoid'))   # output layer
    #sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # TODO See what loss function is needed....weighting also

    if train_epoch==True:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=13, restore_best_weights=True)
        r = model.fit(X, y, epochs=300, batch_size=2500, validation_split=0.1, callbacks=[es], verbose=1)
        stop_epoch = np.argmin(r.history['val_loss']) + 1
        return model, stop_epoch
    else:
        model.fit(X, y, epochs=epoch, batch_size=2500, verbose=1)
        return model

def generate_hyps():
    """
    generate a list of different combinations of hyperparameters for ANN model
    each combination includes number of neuros of each layer, l1 and l2 parameters
    """
    # in order to short the learning time, 
    # here only generate the best hyperparameters for ANN model according to previous trainings
    hyperparameters = []
    for layers in [[500,500,500]]:
        for l1 in [0]:
            for l2 in [0.00005]:
                hyperparameters.append({ "layers": layers, "l1":l1, "l2": l2})
    return hyperparameters

def Kfold_model_selection(X,y,k=10):
    
    model_hyps = generate_hyps()  # generate a list of hyparameters for ANN model
    print(model_hyps)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    model_counts = len(model_hyps)  # number of models
    
    kf_threshold = np.zeros(shape=(k, model_counts))
    kf_f1 = np.zeros(shape=(k, model_counts))
    kf_epoch = np.zeros(shape=(k, model_counts))
    
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model_threshold = []
        model_f1 = []
        model_epoch = []
        for hyp in model_hyps: #
            #print("model hyperparameter:", hyp)
            model,stop_epoch = trained_model(X_train,y_train, hyp, train_epoch=True) 
        
            y_scores = model.predict(X_test)
            threshold, f_score = bestThreshold(y_test,y_scores)
    
            model_threshold.append(threshold) # best threshold of one model
            model_f1.append(f_score)
            model_epoch.append(stop_epoch)
            
        kf_threshold[index] = model_threshold
        kf_f1[index] = model_f1
        kf_epoch[index] = model_epoch
        
        #print('round of CV:', index)
        #print("best threshold of each model", model_threshold )
        #print("best f1 of each model", model_f1)
        #print("stop epochs of each model", model_epoch)
        #print('\n')
    
    mean_threshold = np.mean(kf_threshold, axis=0)  # column mean, kfold mean threshold for each model
    mean_f1 = np.mean(kf_f1, axis=0)  # kfold f1 mean for each model
    mean_epoch = np.mean(kf_epoch, axis=0)
    
    best_f1_index = np.argmax(mean_f1)
    best_threshold = mean_threshold[best_f1_index]
    best_hyp = model_hyps[best_f1_index]     # get best hyp according to the maximum f1 score of models
    best_epoch = int(mean_epoch[best_f1_index])
    
    #print('kf_thresholds:', kf_threshold)
    #print('kf_f_scores:', kf_f1)
    #print('\n')
    print('best_hyp:%s, best stop epoch for this model: %s'%(best_hyp, best_epoch))
    print("average best f1: %s, average best_threshold:%s"%(max(mean_f1), best_threshold))
    
    return best_hyp, best_threshold, best_epoch

def best_model(X,y):
    
    best_hyp, best_threshold, best_epoch = Kfold_model_selection(X,y)  
    
    bestm = trained_model(X,y,best_hyp,epoch=best_epoch,train_epoch=False)
    
    return bestm, best_threshold

def predictY(X,y,X_pred,thresh=1):
    model, kf_threshold = best_model(X,y)    # this threshold is the average best threshold in 10-fold CV
    y_scores = model.predict(X)
    global_threshold, global_f = bestThreshold(y, y_scores)   # global best threshold
    
    if thresh == 1:
        threshold = kf_threshold
    elif thresh == 0:
        threshold = global_threshold
    else:
        print('thresh is 1 or 0, choose threshold between kf_threshold and global_threshold. ')
    
    y_pred = model.predict(X_pred)
    prediction = (y_pred > threshold).astype(int)
    prediction = list(prediction.flatten())
    print('number of active: ', np.sum(prediction))
    
    df = pd.DataFrame({'Active':prediction})
    df.to_csv("labels.csv",index=False, header=False)

if __name__ == "__main__":
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    
    # feature engineer 
    X = preprocessX(df_train["Sequence"])
    y = df_train["Active"]
    X_pred = preprocessX(df_test["Sequence"])

    
    #predictY(X,y,X_pred,thresh=0)    # global best-threshold
    predictY(X,y,X_pred)   # kfold mean best-threshold

