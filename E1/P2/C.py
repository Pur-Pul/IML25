import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from ISLP.models import (ModelSpec as MS, poly)

def MSE(model, x, y):
    return ((y - model.predict(x))**2).sum() / len(x)

class dummy():
    def __init__(self, df):
        self.mean = np.mean(trainDF['Next_Tmax'])

    def predict(self, x):
        return self.mean

trainDF = pd.read_csv("data/train_real.csv")

X = trainDF.drop(columns=['Next_Tmax'])
y = trainDF['Next_Tmax']

model = sm.OLS(y, X).fit()
print(MSE(dummy(trainDF), X, y))
print(MSE(model, X, y))