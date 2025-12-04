import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)

trainDF = pd.read_csv('data/train.csv')
class4 = trainDF['class4']

colors = ['r', 'g', 'b', 'y']
glyphs = ['o', '^', '*', 's']


PCX, PCY = 0, 1
for e, event in enumerate(trainDF['class4'].unique()):
    X = trainDF[trainDF['class4'] == event].filter(regex=r'\.std$')
    scaled_X = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(scaled_X)
    scores = pca.transform(scaled_X)
    plt.scatter(scores[:,PCX], scores[:,PCY], color=colors[e], marker=glyphs[e], label=event)

plt.xlabel('PC%d' % (PCX+1))
plt.ylabel('PC%d' % (PCY+1))
plt.legend()
plt.show()
