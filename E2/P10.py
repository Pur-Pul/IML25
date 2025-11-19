import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import ceil, sqrt, pi, exp

trainDF = pd.read_csv('data/penguins_train.csv')
testDF = pd.read_csv('data/penguins_test.csv')

#Task a
def getAttributeStats(df, column, y):
    classDF = df[df[column] == y].drop(columns=[column])
    stats = pd.DataFrame({'stat': ['mean', 'std']})
    for attr in classDF.columns.values:
        stats[attr] = [np.mean(classDF[attr]), np.std(classDF[attr])] 
        
    return stats

def laplaceSmooth(df, column, y):
    classDF = df[df[column] == y].drop(columns=[column])
    return (len(classDF) + 1) / (len(df) + 2)

print('Adelie')
print(getAttributeStats(trainDF, 'species', 'Adelie'))

print('prob:', laplaceSmooth(trainDF, 'species', 'Adelie'))

print()
print('not Adelie')
print(getAttributeStats(trainDF, 'species', 'notAdelie'))

print('prob:', laplaceSmooth(trainDF, 'species', 'notAdelie'))
print()

#Task c
def density(x, mean, std):
    return exp(-((x - mean) ** 2) / (2 * std**2)) / (sqrt(2*pi) * std)

def postProb(df, column, y, X):
    numerator = laplaceSmooth(df, column, y)
    for attr in X.columns.values:
        stats = getAttributeStats(df, column, y)
        mean = stats[stats['stat'] == 'mean'][attr].iloc[0]
        std = stats[stats['stat'] == 'std'][attr].iloc[0]
        x = X[attr].iloc[0]

        numerator *= density(x, mean, std)
    
    classes = df[column].unique()
    denominator = 0
    for c in classes:
        product = laplaceSmooth(df, column, c)
        stats = getAttributeStats(df, column, c)
        for attr in X.columns.values:
            mean = stats[stats['stat'] == 'mean'][attr].iloc[0]
            std = stats[stats['stat'] == 'std'][attr].iloc[0]
            x = X[attr].iloc[0]
            product *= density(x, mean, std)
        denominator+=product
    
    return numerator / denominator

X = testDF.drop(columns=['species'])
Y = testDF['species']
preds = []
for i in range(0,len(testDF)):
    prob = postProb(trainDF, 'species', 'Adelie', X.iloc[[i]])

    preds.append('Adelie' if prob >= 0.5 else 'notAdelie')
    if i < 3:
        print(f'p^(y = Adelie | x_{i}) =', prob)
print('Accuracy:', np.mean(testDF['species'].values == np.array(preds)))
