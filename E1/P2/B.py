import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from ISLP.models import (ModelSpec as MS, poly)

trainDF = pd.read_csv("data/train_syn.csv")

for p in range (0, 9):
    y = trainDF['y']
    new_df = pd.DataFrame({ 'x' : np.linspace(-3, 3, 256) })
    if p == 0 :
        X = sm.add_constant(np.zeros(len(trainDF)))
        new_X = sm.add_constant(np.zeros(len(new_df)))
        
    else :
        X = MS([poly('x', degree=p, raw=True)]).fit_transform(trainDF)
        new_X = MS([poly('x', degree=p, raw=True)]).fit_transform(new_df)

    model = sm.OLS(y, X).fit()
    Y = model.predict(new_X)
    
    plt.subplot(3,3,p+1)
    plt.ylim(trainDF['y'].min() - 1, trainDF['y'].max() + 1)
    plt.scatter(trainDF['x'], trainDF['y'])
    plt.plot(np.linspace(-3, 3, 256), Y, label=f'p={p}', color='red')
    plt.legend()

plt.show()