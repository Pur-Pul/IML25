import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

trainDF = pd.read_csv('data/train.csv')

scaler = StandardScaler(with_mean=True, with_std=True)

X = scaler.fit_transform(trainDF.filter(regex=r'\.mean$'))

kmeans = KMeans(n_clusters=4, random_state=2, n_init=10, init='random').fit(X)

contingency = pd.DataFrame({ 'class': trainDF['class4'], 'cluster': kmeans.labels_})
contingency = contingency.groupby(['class', 'cluster']).size().unstack()
contingency = contingency.fillna(0)

lsa = linear_sum_assignment(contingency, maximize = True)

print(contingency[lsa[1]])

