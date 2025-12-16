import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)
le = LabelEncoder()

trainDF = pd.read_csv('data/train.csv')
class4 = le.fit_transform(trainDF['class4'])

X = trainDF.drop(columns=['date', 'id', 'class4'])
scaledX = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(
        np.linspace(
            X.iloc[:, 0].min() - 1,
            X.iloc[:, 0].max() + 1,
            500
        ),
        np.linspace(
            X.iloc[:, 1].min() - 1,
            X.iloc[:, 1].max() + 1,
            500
        )
    )

    Z = model.fit(X, y).predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns))
    print(np.unique(Z))
    Z = Z.reshape(xx.shape)
        
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.tab10)

def plot_pca(X, label, p_cols, p_i):
    p_rows = 3
    pca = PCA().fit(X)
    scores = pca.transform(X)
    PCs = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1, scores.shape[1] + 1)])
    PVE = pca.explained_variance_ratio_

    lf = LR(max_iter=500)
    svc = SVC(C=0.6)
    nn1 = KNeighborsClassifier(n_neighbors=10)
    lda = LinearDiscriminantAnalysis()
  
    print(f'10-fold cv on {label} with LR (2 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2']], class4, cv=10).mean())
    print(f'10-fold cv on {label} with LR (4 first PC):', cross_val_score(lf, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class4, cv=10).mean())
    print(f'10-fold cv on {label} with SVC (4 first PC):', cross_val_score(svc, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class4, cv=10).mean())
    print(f'10-fold cv on {label} with KNN (4 first PC):', cross_val_score(nn1, PCs[['PC1', 'PC2', 'PC3', 'PC4']], class4, cv=10).mean())
    print(f'10-fold cv on {label} with LDA (4 first PC):', cross_val_score(lda, PCs, class4, cv=10).mean())

    plt.subplot(p_rows, p_cols, p_i)
    plt.title(f'PCA on {label}')
    plot_decision_boundary(lf, PCs[['PC1', 'PC2']], class4)
    
    for i, label in enumerate(le.classes_):
        plt.scatter(PCs[class4==i]['PC1'], PCs[class4==i]['PC2'], label=label, color=plt.cm.tab10(i))
    plt.legend()

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.subplot(p_rows, p_cols, p_i + p_cols)
    plt.title(f'PCA on {label}')
    plot_decision_boundary(svc, PCs[['PC1', 'PC2']], class4)
    
    for i, label in enumerate(le.classes_):
        plt.scatter(PCs[class4==i]['PC1'], PCs[class4==i]['PC2'], label=label, color=plt.cm.tab10(i))

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
