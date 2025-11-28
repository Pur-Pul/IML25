import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)

trainDF = pd.read_csv('data/train.csv')
class2 = (trainDF['class4'] != 'nonevent').astype(int)
X = trainDF.drop(columns=['date', 'id', 'class4'])
scaled_X = scaler.fit_transform(X)

pca = PCA()
pca.fit(scaled_X)

scores = pca.transform(scaled_X)
col = np.where(class2==1,'b','r')

corr_mat = X.corr()
print(corr_mat)

sns.heatmap(corr_mat, cmap='YlGnBu')

i, j = 0, 1 # which components
scale_arrow = s_ = 50
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(scores[:,0], scores[:,1], color=col)
ax.set_xlabel('PC%d' % (i+1))
ax.set_ylabel('PC%d' % (j+1))
for k in range(pca.components_.shape[1]):
    ax.arrow(0, 0, s_*pca.components_[i,k], s_*pca.components_[j,k])
    ax.text(s_*pca.components_[i,k],
        s_*pca.components_[j,k],
        X.columns[k])

plt.show()
