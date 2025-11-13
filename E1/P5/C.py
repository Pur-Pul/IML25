import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm, summarize)

dfs = [
    pd.read_csv("data/d2.csv"),
    pd.read_csv("data/d3.csv"),
    pd.read_csv("data/d4.csv")
]


for i, df in enumerate(dfs):
    
    x = df['x']
    x = sm.add_constant(x)
    results = sm.OLS(df['y'], x).fit()

    plt.subplot(3,1,i+1)
    if i == 0:
        plt.title(f'd2 residual')
        residual = pd.DataFrame({
            'x': df['x'],
            'y': (df['y'] - results.predict(x))
        })
        plt.scatter(residual['x'], residual['y'], color='blue')
    
    elif i == 1:
        plt.title(f'd3 studentized residual')
        studResidual = pd.DataFrame({
            'x': df['x'],
            'y': (df['y'] - results.predict(x))/results.get_prediction(x).summary_frame()['mean_se']
        })
        outliers = studResidual[abs(studResidual['y']) > 3]
        plt.scatter(studResidual['x'], studResidual['y'], color='blue')
        plt.scatter(outliers['x'], outliers['y'], color='red')
    elif i == 2:
        plt.title(f'd4 leverage stats')
        influence = results.get_influence()
        leverage = influence.hat_matrix_diag
        levDF = pd.DataFrame({'x': df['x'], 'y': leverage})
        highLevs = levDF[levDF['y'] > np.mean(leverage)]
        
        plt.scatter(levDF['x'], levDF['y'])
        plt.scatter(highLevs['x'], highLevs['y'], color='red')
    
plt.show()