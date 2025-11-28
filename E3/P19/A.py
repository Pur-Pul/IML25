import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

trainDF = pd.read_csv('data/train.csv')

scaler = StandardScaler()
X = scaler.fit_transform(trainDF.filter(regex=r'\.mean$'))

losses = []

for K in range(1, 21):
    kmeans = KMeans(n_clusters=K, random_state=2, n_init=10, init='random').fit(X)
    losses.append(kmeans.inertia_)

plt.plot(np.linspace(1, 20, 20), losses)
plt.show()