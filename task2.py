#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Lasso, Ridge, LinearRegression 
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
#from statistics import mean
# from score_submission import get_score, VITALS, TESTS
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# import dataset
train_labels = pd.read_csv('./train_labels.csv')
train_features = pd.read_csv('./train_features.csv')
test_features = pd.read_csv('./test_features.csv')


# In[169]:


print("number of null value:", train_features.isnull().sum())


# In[3]:


# different data preprocessing for classifications and regression
# preprocess the time series
def imputeDF(df,step=12,strategy="mean",good_features = ["RRate","ABPm","ABPd","SpO2","Heartrate","ABPs","Temp","Glucose"]):
    """
    step is the number of time series for each pid 
    for each patient, do the following tasks:
    (1) impute missing value
    (2) combine: add all 12 hours good features as columns to each patient
    (3) add number of missing value for each features as other columns
    """
    rows = int(df.shape[0])   # number of unique pid
    df_impute = pd.DataFrame()
    
    for i in range(0,rows,step):
        df_basic = pd.DataFrame(df.iloc[i:i+step,0:3].mean())
        df_feature = df.iloc[i:i+step,3:]     # the 12 rows for each pid
        missingN = pd.DataFrame(df_feature.isnull().sum())  # number of missing value of each feature`
        
        df_feature_good = df_feature[good_features]
        df_feature_bad = df_feature.drop(columns=good_features)  # feature with a lot NaN
        
        if strategy == "combine":
            df_feature_good = df_feature_good.interpolate(method="linear", limit_direction='both') 
            df_feature_good = np.array(df_feature_good).flatten()
            # linear interpolate missing value of good features
            df_feature_bad = df_feature_bad.mean()
            # replace NaN with mean for bad features
            df_pid = pd.concat((df_basic,pd.DataFrame(df_feature_good),df_feature_bad,
                                missingN),axis=0)  # merge
            
        elif strategy == "mean":
            df_pid = pd.concat((df_basic, pd.DataFrame(df_feature.mean()), missingN), axis=0)
            # repalce NaN with series column mean for all features
            
        df_impute = df_impute.append(df_pid.T)
        
        #if i == 0:
            #print("df_basic",df_basic)
            #print("\n")
            #print("df_feature",df_feature)
            #print("\n")
            #print("missingN",missingN)
            #print("\n")
            #print("df_feature_good", df_feature_good)
            #print("\n")
            #print("df_feature_bad", df_feature_bad)
            #print("\n")
            #print("df_pid", df_pid)
            #print("\n")
            #print("df_impute", df_impute)
            #print("\n")
            
    return df_impute


# In[4]:


# functions to do K-fold classification and regression

def KfoldModel(X, Y, k=3, N=5, method="linearSVC"):
    """
    input
    X, Y: dataset
    k: k fold
    N: number of splited subsets
    output:
    The mean AUC ROCs of all labels in Y
    The mean r2 for regressions
    """
    # define a list to collect the scores of each round CV
    score_list = []
    
    # The number of columns in Y
    m = Y.shape[1]
    
    # define K-fold parameters
    kf = KFold(n_splits=N, shuffle=True, random_state=42)
    
    # split into training and testing datasets
    for index, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # standardize features
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        
        if method == "linearSVC":
            model = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_scaled, Y_train)
            Y_scores = model.decision_function(X_test_scaled)
            scores = np.mean([roc_auc_score(Y_test.iloc[:,i],Y_scores[:,i]) for i in range(m)])
            # for classification the metrics score is auc roc
            score_list.append(scores)
            
        elif method == "ANN":
            model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1500, 500, 100), 
                                  early_stopping=True, max_iter=100, n_iter_no_change=15, batch_size=1000, 
                                  random_state=42, verbose=True).fit(np.array(X_train_scaled), np.array(Y_train))
            Y_scores = model.predict_proba(np.array(X_test_scaled))
            scores = np.mean([roc_auc_score(Y_test.iloc[:,i:i+1],Y_scores[:,i:i+1]) for i in range(m)])
            score_list.append(scores)
            
        elif method == "Lasso":
            model = Lasso(alpha = 0.1, random_state = 42).fit(X_train_scaled, Y_train)
            Y_scores = model.predict(X_test_scaled)
            scores = np.mean([0.5 + 0.5 * np.maximum(0, r2_score(Y_test.iloc[:, i], Y_scores[:, i])) 
                              for i in range(m)])
            # for regression, the metrics is the r2 score    
            score_list.append(scores)
            
    print(score_list)
    print(np.mean(score_list))
    return np.mean(score_list)

def main():
    # In[5]:


    # use pre-defined function to deal with missing value, prepare the input data for model training
    X_clf =  imputeDF(train_features, strategy="mean")
    X_reg = imputeDF(train_features, strategy="combine")


    # In[6]:


    X_reg.head()


    # In[7]:


    # After preprocessing, there are still NaN left, replace them with 0 or global column means

    # fill NaN with 0
    X_clf1 = X_clf.fillna(0)
    X_reg1 = X_reg.fillna(0)  

    # fill NaN with global column means
    #imp = Imputer(missing_values = 'NaN', strategy = 'mean',axis=0).fit(X_reg)
    #X_reg2 = pd.DataFrame(imp.transform(X_reg))
    imp_reg = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)
    imp_reg.fit(X_reg)
    X_reg2 = pd.DataFrame(imp_reg.transform(X_reg))
    X_reg2.columns = X_reg1.columns

    #print(X_clf1.head())
    #print(X_reg1.head())

    # sort according to 'pid' to make X and Y in a corresponding position by 'pid'
    Y = train_labels.sort_values('pid')
    X_clf1 = X_clf1.sort_values('pid')
    X_reg1 = X_reg1.sort_values('pid')
    X_reg2 = X_reg2.sort_values('pid')

    Y_clf = Y.iloc[:, 1:12]
    Y_reg = Y.iloc[:,12:]

    #print(X_clf1.head())
    #print(X_reg1.head())
    print(Y_clf.head())
    #print(Y_reg.head())


    # In[8]:


    # use the prepared training data to feed the models, compare models and get the evaluation scores
    # note: here, X_clf include pid and time
    # KfoldModel(X_clf1, Y_clf, method="ANN")


    # # In[9]:


    # KfoldModel(X_reg1.drop(columns=["pid","Time"]), Y_reg, method="Lasso") # bad performance


    # # In[10]:


    # KfoldModel(X_reg2.drop(columns=['pid','Time']), Y_reg, method="Lasso")  # good performance
    # # better than X_reg1


    # # In[11]:


    # KfoldModel(X_reg2, Y_reg, method="Lasso")  # better performance


    # In[12]:


    # preprocess data for prediction
    X_clf_pred =  imputeDF(test_features, strategy="mean").fillna(0)   
    X_reg_pred = imputeDF(test_features, strategy="combine")


    # In[13]:


    imp_reg = SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)
    imp_reg.fit(X_reg_pred)
    X_reg_pred1 = pd.DataFrame(imp_reg.transform(X_reg_pred))


    # In[18]:



    def predictY(X_train,Y_train, X_pred, method="classification"):
        # standardize data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)
        
        # fit model
        if method == "classification":
            model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1500, 500, 100), 
                                    early_stopping=True, max_iter=50, n_iter_no_change=15, batch_size=1000, 
                                    random_state=42, verbose=True).fit(X_train_scaled, Y_train)
            Y_pred = pd.DataFrame(model.predict_proba(np.array(X_pred_scaled)))
        elif method == "regression":
            model = Lasso(alpha = 0.1).fit(X_train_scaled, Y_train)
            Y_pred = pd.DataFrame(model.predict(X_pred_scaled))
        
        return Y_pred
        
    Y_clf_pred = predictY(X_clf1, Y_clf, X_clf_pred, method="classification")
    Y_reg_pred = predictY(X_reg2, Y_reg, X_reg_pred1, method="regression")
    print(Y_clf_pred.shape)
    print(Y_reg_pred.shape)


    # In[19]:


    # write the predictions into one dataframe
    prediction = pd.DataFrame({'pid':list(X_clf_pred['pid'])})
    prediction = pd.concat((prediction, Y_clf_pred, Y_reg_pred),axis=1)
    prediction['pid'] = prediction['pid'].astype(int)
    prediction.columns = train_labels.columns
    prediction.head()

    return prediction


if __name__ == "__main__":
    df = main()

    # In[20]:

    # output, write into a csv file
    df.to_csv("task2.csv",index=False, header=True)
    df.to_csv('task2.zip', index=False, float_format='%.3f', compression='zip')



