import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector

le = LabelEncoder()

trainDF = pd.read_csv('data/train.csv')
testDF = pd.read_csv('data/test.csv')
class2 = (trainDF['class4'] != 'nonevent').astype(int)
class4 = le.fit_transform(trainDF['class4'])

classMap = {
    'nonevent' : np.where(le.classes_ == 'nonevent')[0][0],
    'II' : np.where(le.classes_ == 'II')[0][0],
    'Ia' : np.where(le.classes_ == 'Ia')[0][0],
    'Ib' : np.where(le.classes_ == 'Ib')[0][0]
}

X = trainDF.drop(columns=['date', 'id', 'class4'])

class Model(SVC):
    def __init__(self, PCA_depth, C=1, kernel='rbf', degree=3):
        self.PCA_depth = PCA_depth
        self.pca = PCA()
        self.scaler = StandardScaler(with_mean=True, with_std=True)

        super().__init__(C=C, kernel=kernel, degree=degree, probability=True)

    def fit_pca(self, X):
        scaledX = pd.DataFrame(self.scaler.fit_transform(X), columns = X.columns)
        self.fit_cols = X.columns
        self.pca.fit(scaledX)

    def PCA(self, X):
        scaledX = pd.DataFrame(self.scaler.transform(X[self.fit_cols]), columns = self.fit_cols)
        scores = self.pca.transform(scaledX)
        PCs = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1, scores.shape[1] + 1)])
        return PCs[[f'PC{i}' for i in range(1, self.PCA_depth+1)]]

    def fit(self, X, y, sample_weight=None):
        self.fit_pca(X)
        self.PCs = self.PCA(X)
        self.y = y
        
        event_count = np.bincount(y)
        event_count[classMap['nonevent']] = -1
        self.most_common_event = event_count.argmax()

        class2 = np.copy(y)
        class2[class2 != classMap['nonevent']] = event_count.argmax()

        eventX = self.PCs[y != classMap['nonevent']]
        event_y = y[y != classMap['nonevent']]

        self.event_model = LR(max_iter=500).fit(
            self.PCs,
            y
        )
        return super().fit(self.PCs, class2, sample_weight)

    def predict(self, X_test, pca=True, return_probs=False):
        if pca:
            testPCs = self.PCA(X_test)
        else:
            testPCs = X_test

        event_preds = self.event_model.predict(testPCs)
        class2_preds = super().predict(testPCs)
        preds = np.array([])
        probs = np.array([])

        event_probs = self.event_model.predict_proba(testPCs)
        class2_probs = super().predict_proba(testPCs)

        print(event_preds)

        for i, cls in enumerate(class2_preds):
            if cls != classMap['nonevent']:
                preds = np.append(preds, event_preds[i])
                probs = np.append(probs, event_probs[i][np.where(self.event_model.classes_ == cls)[0][0]])
            else:
                preds = np.append(preds, class2_preds[i])
                probs = np.append(probs, class2_probs[i][np.where(self.classes_ == cls)[0][0]])

        if return_probs:
            return [preds, probs]

        return preds
    
    def predict_and_write(self, X_test, ids):
        preds, probs = self.predict(X_test, return_probs=True)

        

        reverseMap = { v: k for k, v in classMap.items() }
        preds_mapped = [reverseMap[pred] for pred in preds]

        output = pd.DataFrame({ 'id': ids, 'class4': preds_mapped, 'p': probs })
        output.to_csv('models/pca_svc_lr.csv', index=False)

    def plot_scatter(self, axes):
        for i, label in enumerate(le.classes_):
            #if i == classMap['nonevent']:
            #    continue
            plt.scatter(self.PCs[self.y==i][f'PC{axes[0]+1}'], self.PCs[self.y==i][f'PC{axes[1]+1}'], label=label, color=plt.cm.tab10(i))
    
    def plot_decision_boundary(self, axes):
        xx, yy = np.meshgrid(
            np.linspace(
                self.PCs.iloc[:, axes[0]].min() - 1,
                self.PCs.iloc[:, axes[0]].max() + 1,
                500
            ),
            np.linspace(
                self.PCs.iloc[:, axes[1]].min() - 1,
                self.PCs.iloc[:, axes[1]].max() + 1,
                500
            )
        )

        spaceslice = []
        for i in range(0, self.PCA_depth):
            spaceslice.append(np.zeros(xx.shape).ravel())
        
        spaceslice[axes[0]] = xx.ravel()
        spaceslice[axes[1]] = yy.ravel()

        Z = self.predict(pd.DataFrame(np.c_[spaceslice].T, columns=self.PCs.columns), pca=False)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.5)
    
    def plot_PVE(self):
        PVE = self.pca.explained_variance_ratio_
        plt.plot(np.linspace(1, len(PVE), len(PVE)), PVE, label='PVE')
        plt.xlabel('Principal component')
        plt.ylabel('Prop. Variance Explained')
        plt.legend()




def plot_pca(X, y, label):
    p_rows = 1
    p_cols = 1

    model = Model(PCA_depth=4)
    model2 = Model(PCA_depth=2)

    print(f'10-fold cv on {label} with SVC (2 first PC):', cross_val_score(model2, X, y, cv=10).mean())
    print(f'10-fold cv on {label} with SVC (4 first PC):', cross_val_score(model, X, y, cv=10).mean())


    model.fit(X, y)

    p1 = 0
    p2 = 1

    #for p1 in range(0, 4):
    #    for p2 in range(p1+1, 4):

    plt.subplot(p_rows, p_cols, p1+p2)
    plt.title(f'PCA on {label}')
    model.plot_decision_boundary([p1, p2])
    model.plot_scatter([p1, p2])
    plt.legend()
    plt.xlabel(f'PC{p1+1}')
    plt.ylabel(f'PC{p2+1}')




#plot_pca(
#    X.filter(regex=r'\.mean$'),
#    class4,
#    'mean values'
#)

#plt.show()

Model(PCA_depth=4).fit(trainDF.filter(regex=r'\.mean$'), class4).predict_and_write(testDF, ids= testDF['id'])
