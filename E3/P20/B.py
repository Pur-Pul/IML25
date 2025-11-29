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

X = trainDF.filter(regex=r'\.mean$')
scaled_X = scaler.fit_transform(X)

pca = PCA()
pca.fit(scaled_X)
scaled_PVE = pca.explained_variance_ratio_

pca.fit(X)
PVE = pca.explained_variance_ratio_


plt.subplot(1, 2, 1)
#plt.plot(np.linspace(1, len(PVE), len(PVE)), PVE, label='Raw PVE')
plt.plot(np.linspace(1, len(scaled_PVE), len(scaled_PVE)), scaled_PVE, label='Normalized PVE')
plt.xlabel('Principal component')
plt.ylabel('Prop. Variance Explained')
plt.legend()

plt.subplot(1, 2, 2)
#plt.plot(np.linspace(1, len(PVE), len(PVE)), np.cumsum(PVE), label='Raw PVE')
plt.plot(np.linspace(1, len(scaled_PVE), len(scaled_PVE)), np.cumsum(scaled_PVE), label='Normalized PVE')
plt.xlabel('Principal component')
plt.ylabel('Cumulative Prop. Variance Explained')

plt.legend()
plt.show()
