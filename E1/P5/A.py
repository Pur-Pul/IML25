import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm, summarize)

dfs = [
    pd.read_csv("data/d1.csv"),
    pd.read_csv("data/d2.csv"),
    pd.read_csv("data/d3.csv"),
    pd.read_csv("data/d4.csv")
]

for i, df in enumerate(dfs):
    x = df['x']
    x = sm.add_constant(x)
    results = sm.OLS(df['y'], x).fit()
    print(summarize(results))
    print(f"Dataset d{i}:")
    print(f'| coeff\t\t| Term estimate\t\t| Standard error\t| P-value\t\t|')
    print(f'|---------------|-----------------------|-----------------------|-----------------------|')
    print(f'| intercept\t| {results.params['const']}\t| {results.bse['const']}\t|{results.pvalues['const']}\t|')
    print(f'| Slope\t\t| {results.params['x']}\t| {results.bse['x']}\t|{results.pvalues['x']}\t|')
    print("R-squared:", results.rsquared)
    print('\n')
    plt.subplot(2,2,i+1)
    plt.scatter(df['x'], df['y'])
    plt.plot(df['x'], results.predict(x), color='red')
    plt.title(f'd{i+1}')
plt.show()