import pandas as pd
import numpy as np
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm)
from sklearn.model_selection import (cross_validate, KFold, ShuffleSplit)

def MSE(model, x, y):
    return ((y - model.predict(x))**2).sum() / len(x)

trainDF =   pd.read_csv("data/train_syn.csv")
testDF =    pd.read_csv("data/test_syn.csv")
validDF =   pd.read_csv("data/valid_syn.csv")
trvaDF =    pd.concat([trainDF, validDF], ignore_index=True)

H = np.array(trvaDF['x'])
M = sklearn_sm(sm.OLS)

print('| Degree\t| Train\t\t\t| Validation\t\t| Test\t\t\t| TestTRVA\t\t| CV\t\t\t|')
print('|---------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|')
for p in range(0, 9):
    if p == 0:
        trainPolyDF = pd.DataFrame({ 'const': np.ones(len(trainDF)) })
        testPolyDF = pd.DataFrame({ 'const': np.ones(len(testDF)) })
        validPolyDF = pd.DataFrame({ 'const': np.ones(len(validDF)) })
        trvaPolyDF = pd.DataFrame({ 'const': np.ones(len(trvaDF)) })
    else:
        trainPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(trainDF)
        testPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(testDF)
        validPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(validDF)
        trvaPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(trvaDF)
    
    model = sm.OLS(trainDF['y'], trainPolyDF).fit()
    trvaModel = sm.OLS(trvaDF['y'], trvaPolyDF).fit()
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
   
    X = np.power.outer(H, np.arange((p+1)))
    print(
        f'| {p}\t\t|',
        f'{MSE(model,       trainPolyDF, trainDF['y'])}\t|',
        f'{MSE(model,       validPolyDF, validDF['y'])}\t|',
        f'{MSE(model,       testPolyDF, testDF['y'])}\t|',
        f'{MSE(trvaModel,   testPolyDF, testDF['y'])}\t|',
        f'{np.mean(cross_validate(
            M,
            X,
            trvaDF['y'],
            cv=cv
        )['test_score'])}\t|'
    )
