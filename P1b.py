
# import modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import math


data = pd.read_csv("./train.csv")


# feature transformation, add transformed features
def AddTransformFeature(df, func, index):
    for i in range(1, 6):
        f = "x%s" % i
        t = "x%s" % (index + i)
        df[t] = df[f].apply(func)
 
AddTransformFeature(data, lambda x: x ** 2, 5)
AddTransformFeature(data, math.exp, 10)
AddTransformFeature(data, math.cos, 15)
data['x21'] = 1
data.head()


y = data['y']
# y.head()
X = data.drop(columns=['Id', 'y'])
# X.head()


# K-fold cross validation for linear regression, return RMSE
def rmseLinear(feature, target, fold=5):
    lin = LinearRegression(fit_intercept=False)
    nmse_lin = cross_val_score(lin, feature, target, cv=fold, scoring='neg_mean_squared_error')
    rmse_lin = [math.sqrt(-x) for x in nmse_lin]
    RMSE_lin = sum(rmse_lin) / fold
    print('Linear RMSE: ', RMSE_lin)
    return


# K-fold cross validation for ridge
# print best fit lamda and corresponding RMSE, and return RMSE list for each lamda
def rmseKfoldRidge(feature, target, regul, fold=10):
    RMSE_rid = []
    for it in regul:
        ridg = linear_model.Ridge(alpha = it, fit_intercept=False)
        nmse_rid = cross_val_score(ridg, feature, target, cv=fold, scoring='neg_mean_squared_error')
        rmse_rid = [math.sqrt(-x) for x in nmse_rid]
        RMSE_aver = sum(rmse_rid)/fold
        RMSE_rid.append(RMSE_aver)
    print("best ridge regularizer parameter: ", regul[RMSE_rid.index(min(RMSE_rid))],"\nridge RMSE = %.15f" % min(RMSE_rid))
    return RMSE_rid



# k fold cross validation for lasso
# print best fit lamda and corresponding RMSE, and return RMSE list for each lamda
def rmseKfoldLasso(feature, target, regul, fold=10):
    RMSE_las = []
    for it in regul:
        lass = linear_model.Lasso(alpha = it, fit_intercept=False, max_iter=15000)
        nmse_las = cross_val_score(lass, feature, target, cv=fold, scoring='neg_mean_squared_error')
        rmse_las = [math.sqrt(-x) for x in nmse_las]
        RMSE_aver = sum(rmse_las)/fold
        RMSE_las.append(RMSE_aver)
    print("best lasso regularizer parameter: ", regul[RMSE_las.index(min(RMSE_las))], 
      "\nRMSE lasso = %.15f" % min(RMSE_las))
    return RMSE_las



# check different regularizer parameter to get the best fit model
regul = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,8,10,15,20,25,30,50,100]
np.random.seed(123)
rmseLinear(X, y, fold=10)
rmseKfoldLasso(X, y, regul, fold=10)
rmseKfoldRidge(X, y, regul, fold=10)



# the coefficient of the best fit model
Best_Model = linear_model.Ridge(alpha=25, fit_intercept=False)
reg = Best_Model.fit(X,y)
coefs = reg.coef_
print('Coefficients: ', coefs) 



# write coefficients into a csv file
coefs_arr = np.array(coefs).transpose()
df = pd.DataFrame({'c':coefs_arr})
df.to_csv("task1b_result.csv",index=False, header=False)

