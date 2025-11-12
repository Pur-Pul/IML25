import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, poly)
import pandas as pd
from math import sqrt

N = 10

def f(x):
    return -2 -x + 0.5*x**2



print(f'| Degree\t| Irreducible\t\t| BiasSq\t\t| Variance\t\t| Total\t\t\t| MSE\t\t\t|')
print('|---------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|')

degreesDF = pd.DataFrame({'degree':[], 'f(0)':[], 'y0':[], 'f^(0)':[] })
for iteration in range(0, 1000):
    x = np.random.uniform(-3, 3, size=N)
    epsilon = np.random.normal(0, 0.4, size=N)
    y = f(x) + epsilon
    D = pd.DataFrame({ 'x': x, 'y': y })
    zeroDF = pd.DataFrame({'x': [0, 0], 'y': [f(0) + np.random.normal(0, 0.4), 0]})
    for p in range(0, 7):
        if p == 0:
            trainPolyDF = pd.DataFrame({ 'const': np.ones(len(D)) })
            zeroPolyDF = pd.DataFrame({ 'const': np.ones(len(zeroDF)) })
        else:
            spec = MS([poly('x', degree=p, raw=True)])
            trainPolyDF = spec.fit_transform(D)
            zeroPolyDF = spec.transform(zeroDF)
            
        model = sm.OLS(D['y'], trainPolyDF).fit()

        degreesDF = pd.concat(
            [
                degreesDF,
                pd.DataFrame([[p, f(0), zeroDF['y'][0], model.predict(zeroPolyDF)[0]]], columns=degreesDF.columns)
            ], 
            ignore_index=True
        )

results = pd.DataFrame({'degree': [], 'irreducible error': [], 'bias term': [], 'variance term': [], 'squared loss': []})
for p in range(0, 7):
    df = degreesDF[degreesDF['degree'] == p]
    irreducible = np.mean((df['y0'] - df['f(0)'])**2)
    biasSq = (np.mean(df['f^(0)']) - f(0))**2
    variance = np.mean((df['f^(0)'] - np.mean(df['f^(0)']))**2)
    mse = np.mean((df['y0'] - df['f^(0)'])**2)
    total = irreducible + biasSq + variance

    print(f'| {p}\t\t|', f'{irreducible}\t|', f'{biasSq}\t|', f'{variance}\t|', f'{total}\t|', f'{mse}\t|')
    results = pd.concat(
        [
            results,
            pd.DataFrame([[p, irreducible, biasSq, variance, total]], columns=results.columns)
        ], 
        ignore_index=True
    )

plt.plot(results['degree'], results['squared loss'], label=f'squared loss')
plt.plot(results['degree'], results['irreducible error'], label=f'irreducible error')
plt.plot(results['degree'], results['bias term'], label=f'bias term')
plt.plot(results['degree'], results['variance term'], label=f'variance term')

plt.xlabel('Degree')

plt.legend()
plt.show()
