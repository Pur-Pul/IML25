import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

trainDF = pd.read_csv('data/train.csv')

scaler = StandardScaler(with_mean=True, with_std=True)

X = scaler.fit_transform(trainDF.filter(regex=r'\.mean$'))

losses = np.array([])

for i in range (0, 1000):
    kmeans = KMeans(n_clusters=4, n_init=1, init='random').fit(X)
    losses = np.append(losses, kmeans.inertia_)
print('random')
print('Max loss', np.max(losses))
print('Min loss', np.min(losses))

plt.hist(losses, 100)
plt.xlabel('loss')
plt.ylabel('count')
plt.title('random')
plt.show()

losses = np.array([])

for i in range (0, 1000):
    kmeansplus = KMeans(n_clusters=4, n_init=1, init='k-means++').fit(X)
    losses = np.append(losses, kmeansplus.inertia_)

print('Max loss', np.max(losses))
print('Min loss', np.min(losses))

plt.hist(losses, 100)
plt.xlabel('loss')
plt.ylabel('count')
plt.title('k-means++')
plt.show()