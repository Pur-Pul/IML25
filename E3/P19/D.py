import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import dendrogram
from ISLP.cluster import compute_linkage
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import confusion_matrix

K = 4

scaler = StandardScaler(with_mean=True, with_std=True)

trainDF = pd.read_csv('data/train.csv')
X = scaler.fit_transform(trainDF.filter(regex=r'\.mean$'))

completeLinkage = compute_linkage(AC(distance_threshold=0, n_clusters=None, linkage='complete').fit(X))
singleLinkage = compute_linkage(AC(distance_threshold=0, n_clusters=None, linkage='single').fit(X))

complete_clusters = cut_tree(completeLinkage, n_clusters=K).flatten()
single_clusters = cut_tree(singleLinkage, n_clusters=K).flatten()

print('Complete cluster labels')
print(complete_clusters)
print()
print('Single cluster labels')
print(single_clusters)



print(confusion_matrix(complete_clusters, single_clusters))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
dendrogram(completeLinkage, ax=ax[0], color_threshold=completeLinkage[-K, 2])
ax[0].set_title("Complete Linkage")

dendrogram(singleLinkage, ax=ax[1], color_threshold=singleLinkage[-K, 2])
ax[1].set_title("Single Linkage")

plt.show()