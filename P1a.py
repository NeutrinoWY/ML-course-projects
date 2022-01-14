# import modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import math


# read file
data = pd.read_csv("./train.csv")
# data.head()
col_names = ['y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']


# standardize X
Means = data.mean(axis=0)
STDs = data.std(axis=0)
#print(Means)
#print(STDs)
for i in range(len(col_names)):
    data[col_names[i]] = (data[col_names[i]] - Means[i+1]) / STDs[i+1]
#print(data.head())
X = data.drop(columns=['Id', 'y'])  
y = data['y']
#y.head()


# k-fold cross validation for ridge regularizer
def rmseKfoldRidge(regul, fold=10):
    RMSE_list = []
    for it in regul:
        rmse = []
        ridg = linear_model.Ridge(alpha = it)
        nmse = cross_val_score(ridg, X, y, cv=fold, scoring='neg_mean_squared_error')
        for num in nmse:
            rmse.append(math.sqrt(num*(-1)))
        RMSE_aver = sum(rmse)/fold
        RMSE_list.append(RMSE_aver)
    return RMSE_list

regul = [0.01,0.1,1,10,100]
rmseKfoldRidge(regul)



# write the result of average RMSE of each lamda into a csv file
RMSE_vec = np.array(rmseKfoldRidge(regul)).transpose()
df = pd.DataFrame({'b':RMSE_vec})
df.to_csv("task1a_result3.csv",index=False, header=False)


#  ==> the ridge model with regularizer parameter = 10 has the best performance, with RMSE = 0.5209784851067937
# 
