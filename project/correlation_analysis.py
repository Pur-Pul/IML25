import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

trainDF = pd.read_csv('data/train.csv')
X = trainDF.drop(columns=['date', 'id', 'class4'])

def correlationFilter(X, corr_thresh, max_corr_num):
    mean_corr = X.filter(regex=r'\.mean$').corr()
    drop_list = []
    while True:
        feature_n = len(mean_corr.columns.values)
        corr_sum = pd.DataFrame(np.zeros((1, feature_n)), columns=mean_corr.columns)
        for i in range(0, feature_n):
            feature1 = mean_corr.columns.values[i]
            for j in range(i, feature_n):
                feature2 = mean_corr.columns.values[j]
                if feature1 == feature2:
                    continue
                if abs(mean_corr.loc[feature1, feature2]) > corr_thresh:
                    corr_sum[feature1]+=1
        
        most_correlative = corr_sum.max().idxmax()
        if corr_sum[most_correlative][0] > max_corr_num:
            drop_list.append(most_correlative)
            mean_corr = mean_corr.drop(columns=[most_correlative])
        else:
            print(corr_sum)
            return drop_list
            
print(correlationFilter(X, 0.9, 8))
