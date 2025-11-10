import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from math import sqrt
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm)
from sklearn.model_selection import (cross_validate, KFold, ShuffleSplit)
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.ensemble import RandomForestRegressor as RF

def MSE(model, x, y):
    model.fit(x, y)
    return np.mean((y - model.predict(x))**2)

trainDF = pd.read_csv("data/train_real.csv")
testDF = pd.read_csv("data/test_real.csv")
trainX = trainDF.drop(columns=['Next_Tmax'])
trainY = trainDF['Next_Tmax']
testX = testDF.drop(columns=['Next_Tmax'])
testY = testDF['Next_Tmax']

dummyModel = Dummy(strategy='mean')
RFModel = RF()
OLSmodel = sklearn_sm(sm.OLS)
cv = KFold(n_splits=10, shuffle=True, random_state=0)

print(
    MSE(dummyModel, trainX, trainY),
    MSE(dummyModel, testX, testY),
    np.mean(cross_validate(dummyModel, trainX, trainY, cv=cv)['test_score'])
)
print(
    MSE(OLSmodel, trainX, trainY),
    MSE(OLSmodel, testX, testY),
    np.mean(cross_validate(OLSmodel, trainX, trainY, cv=cv)['test_score'])
)
print(
    MSE(RFModel, trainX, trainY),
    MSE(RFModel, testX, testY),
    np.mean(cross_validate(RFModel, trainX, trainY, cv=cv)['test_score'])
)