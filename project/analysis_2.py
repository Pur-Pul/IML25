import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

scaler = StandardScaler(with_mean=True, with_std=True)
le = LabelEncoder()

trainDF = pd.read_csv('data/train.csv')
class2 = (trainDF['class4'] != 'nonevent').astype(int)
class4 = le.fit_transform(trainDF['class4'])
X = trainDF.drop(columns=['date', 'id', 'class4'])
scaledX = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

class Model(SVC):
    def fit(self, X, y, sample_weight=None):
        temp = np.bincount(y)
        temp[np.where(le.classes_ == 'nonevent')[0][0]] = -1
        self.most_common = temp.argmax()
        return super().fit(X, y, sample_weight)

    def predict(self, X_test):
        results = super().predict(X_test)
        for i, cls in enumerate(results):
            if cls == np.where(le.classes_ == 'nonevent')[0][0]:
                continue

            results[i] = self.most_common

        print(results)
        return results

def plot_decision_boundary(model, X, y):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.fit(X, y).predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)

def plot_pca(X, label, p_cols, p_i):
    p_rows = 1
    pca = PCA().fit(X)
    scores = pca.transform(X)
    PCs = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1, scores.shape[1] + 1)])
    PVE = pca.explained_variance_ratio_

    lf = LR(max_iter=500)
    svc = SVC()

    svc.fit(PCs[['PC1', 'PC2', 'PC3', 'PC4']], class2)
    
    print(f'10-fold cv on {label} with SVC (4 first PC):', cross_val_score(Model(), PCs[['PC1', 'PC2', 'PC3', 'PC4']], class4, cv=10).mean())

    plt.subplot(p_rows, p_cols, p_i)
    plt.title(f'PCA on {label}')

    plot_decision_boundary(svc, PCs[['PC1', 'PC2']], class2)

    for i, label in enumerate(le.classes_):
        plt.scatter(PCs[class4==i]['PC1'], PCs[class4==i]['PC2'], label=label, color=plt.cm.tab10(i))
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')


plot_pca(scaledX.filter(regex=r'\.mean$'), 'mean values', 1, 1)

plt.show()
