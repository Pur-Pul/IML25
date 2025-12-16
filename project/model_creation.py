import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)

trainDF = pd.read_csv('data/train.csv')
testDF = pd.read_csv('data/test.csv')
class2 = (trainDF['class4'] != 'nonevent').astype(int)
X = trainDF.drop(columns=['date', 'id', 'class4'])
scaledX = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)


def create_model(X, test):
    pcaTrain = PCA().fit(X)
    trainScores = pcaTrain.transform(X)
    trainPCs = pd.DataFrame(trainScores, columns=[f'PC{i}' for i in range(1, trainScores.shape[1] + 1)])

    pcaTest = PCA().fit(test)
    testScores = pcaTest.transform(test)
    testPCs = pd.DataFrame(testScores, columns=[f'PC{i}' for i in range(1, testScores.shape[1] + 1)])

    svc = SVC()
    model = svc.fit(trainPCs[['PC1', 'PC2', 'PC3', 'PC4']], class2)

    print(model.predict(testPCs[['PC1', 'PC2', 'PC3', 'PC4']]))

    #print(f'10-fold cv on {label} with LR (2 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2']], class2, cv=10).mean())
    #print(f'10-fold cv on {label} with LR (4 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class2, cv=10).mean())
    print(f'10-fold cv:', cross_val_score(svc, trainPCs[['PC1', 'PC2', 'PC3', 'PC4']], class2, cv=10).mean())


testX = testDF.filter(regex=r'\.mean$')
scaledTest = pd.DataFrame(scaler.fit_transform(testX), columns = testX.columns)
create_model(scaledX.filter(regex=r'\.mean$'), scaledTest)