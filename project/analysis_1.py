import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)

trainDF = pd.read_csv('data/train.csv')
class2 = (trainDF['class4'] != 'nonevent').astype(int)
X = trainDF.drop(columns=['date', 'id', 'class4'])
scaledX = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.fit(X, y).predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)

def plot_pca(X, label, p_cols, p_i):
    p_rows = 3
    pca = PCA().fit(X)
    scores = pca.transform(X)
    PCs = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1, scores.shape[1] + 1)])
    PVE = pca.explained_variance_ratio_

    lf = LR(max_iter=500)
    svc = SVC()
  
    print(f'10-fold cv on {label} with LR (2 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2']], class2, cv=10).mean())
    print(f'10-fold cv on {label} with LR (4 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class2, cv=10).mean())
    print(f'10-fold cv on {label} with SVC (4 first PC):', cross_val_score(svc, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class2, cv=10).mean())

    plt.subplot(p_rows, p_cols, p_i)
    plt.title(f'PCA on {label}')
    plt.scatter(PCs[class2==1]['PC1'], PCs[class2==1]['PC2'], label='event')
    plt.scatter(PCs[class2==0]['PC1'], PCs[class2==0]['PC2'], label='nonevent')
    plot_decision_boundary(lf, PCs[['PC1', 'PC2']], class2)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(p_rows, p_cols, p_i + p_cols)
    plt.title(f'PCA on {label}')
    plt.scatter(PCs[class2==1]['PC1'], PCs[class2==1]['PC2'], label='event')
    plt.scatter(PCs[class2==0]['PC1'], PCs[class2==0]['PC2'], label='nonevent')
    plot_decision_boundary(svc, PCs[['PC1', 'PC2']], class2)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(p_rows, p_cols, p_i + 2*p_cols)
    plt.title(f'PVE of the {label}')
    plt.plot(np.linspace(1, len(PVE), len(PVE)), PVE, label='PVE')
    plt.xlabel('Principal component')
    plt.ylabel('Prop. Variance Explained')
    plt.legend()


plot_pca(scaledX, 'complete set', 3,1)
plot_pca(scaledX.filter(regex=r'\.mean$'), 'mean values', 3,2)
plot_pca(scaledX.filter(regex=r'\.std$'), 'std values', 3,3)


plt.show()
