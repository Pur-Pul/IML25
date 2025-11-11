import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from math import sqrt
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm)
from sklearn.model_selection import (cross_validate, KFold, ShuffleSplit)
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.ensemble import RandomForestRegressor as RF, GradientBoostingRegressor as GBR
from sklearn.svm import SVR

def RMSE(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)
    return sqrt(np.mean((testY - model.predict(testX))**2))

trainDF = pd.read_csv("data/train_real.csv")
testDF = pd.read_csv("data/test_real.csv")
trainX = trainDF.drop(columns=['Next_Tmax'])
trainY = trainDF['Next_Tmax']
testX = testDF.drop(columns=['Next_Tmax'])
testY = testDF['Next_Tmax']

dummyModel = Dummy(strategy='mean')
RFModel = RF(random_state=0)
OLSModel = sklearn_sm(sm.OLS)
SVRModel = SVR()
GBRModel = GBR(random_state=0)
cv = KFold(n_splits=10)

print(f'| Regressor\t| Train\t\t\t| Test\t\t\t| CV\t\t\t|')
print('|---------------|-----------------------|-----------------------|-----------------------|')
print(
    '| Dummy\t\t|',
    f'{RMSE(dummyModel, trainX, trainY, trainX, trainY)}\t|',
    f'{RMSE(dummyModel, trainX, trainY, testX, testY)}\t|',
    f'{sqrt(-np.mean(cross_validate(dummyModel, trainX, trainY, cv=cv, scoring='neg_mean_squared_error')['test_score']))}\t|'
)
print(
    '| OLS\t\t|',
    f'{RMSE(OLSModel, trainX, trainY, trainX, trainY)}\t|',
    f'{RMSE(OLSModel, trainX, trainY, testX, testY)}\t|',
    f'{sqrt(-np.mean(cross_validate(OLSModel, trainX, trainY, cv=cv, scoring='neg_mean_squared_error')['test_score']))}\t|'
)
print(
    '| RF\t\t|',
    f'{RMSE(RFModel, trainX, trainY, trainX, trainY)}\t|',
    f'{RMSE(RFModel, trainX, trainY, testX, testY)}\t|',
    f'{sqrt(-np.mean(cross_validate(RFModel, trainX, trainY, cv=cv, scoring='neg_mean_squared_error')['test_score']))}\t|'
)
print(
    '| SVR\t\t|',
    f'{RMSE(SVRModel, trainX, trainY, trainX, trainY)}\t|',
    f'{RMSE(SVRModel, trainX, trainY, testX, testY)}\t|',
    f'{sqrt(-np.mean(cross_validate(SVRModel, trainX, trainY, cv=cv, scoring='neg_mean_squared_error')['test_score']))}\t|'
)
print(
    '| GBR\t\t|',
    f'{RMSE(GBRModel, trainX, trainY, trainX, trainY)}\t|',
    f'{RMSE(GBRModel, trainX, trainY, testX, testY)}\t|',
    f'{sqrt(-np.mean(cross_validate(GBRModel, trainX, trainY, cv=cv, scoring='neg_mean_squared_error')['test_score']))}\t|'
)