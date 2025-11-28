import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

trainDF = pd.read_csv('data/train.csv')

scaler = StandardScaler()
X = scaler.fit_transform(trainDF.filter(regex=r'\.mean$'))

kmeans = KMeans(n_clusters=4, random_state=2, n_init=20).fit(X)

classes = np.unique(trainDF['class4'])

confusionDF = pd.DataFrame({'class': [], '1': [], '2': [], '3': [], '4': []})

for cl in classes:
    sums = [0,0,0,0]
    for row in trainDF[trainDF['class4'] == cl].index.values:
        sums[kmeans.labels_[row]]+=1

    confusionDF = pd.concat(
        [
            confusionDF,
            pd.DataFrame([[cl, sums[0], sums[1], sums[2], sums[3]]], columns=confusionDF.columns)
        ], 
        ignore_index=True
    )
print(confusionDF)

