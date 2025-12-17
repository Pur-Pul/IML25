import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector
from kaggle_scoring_metric import score

trainDF = pd.read_csv('data/train.csv')
testDF = pd.read_csv('data/test.csv')

class Model():
    def __init__(self, PCA_depth, class2Model, eventModel):
        self.PCA_depth = PCA_depth
        self.pca = PCA()
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.class2Model = class2Model
        self.eventModel = eventModel
        self.eventLE = LabelEncoder()
        self.class2LE = LabelEncoder()

    def fit_pca(self, X):
        scaledX = pd.DataFrame(self.scaler.fit_transform(X), columns = X.columns)
        self.fit_cols = X.columns
        self.pca.fit(scaledX)

    def PCA(self, X):
        scaledX = pd.DataFrame(self.scaler.transform(X[self.fit_cols]), columns = self.fit_cols)
        scores = self.pca.transform(scaledX)
        PCs = pd.DataFrame(scores, columns=[f'PC{i}' for i in range(1, scores.shape[1] + 1)], index=X.index)
        return PCs[[f'PC{i}' for i in range(1, self.PCA_depth+1)]]

    def fit(self, X, y):
        self.fit_pca(X)
        self.PCs = self.PCA(X)
        self.y = y

        self.class2_y = np.copy(y)
        self.class2_y[y != 'nonevent'] = 'event'
        self.class2LE.fit(self.class2_y)

        self.event_y = self.y[self.y != 'nonevent']
        self.eventX = self.PCs[self.y != 'nonevent']
        self.eventLE.fit(self.event_y)
        
        self.class2Map = {
            'nonevent' : np.where(self.class2LE.classes_ == 'nonevent')[0][0],
            'event' : np.where(self.class2LE.classes_ == 'event')[0][0]
        }
        
        self.eventMap = {
            'II' : np.where(self.eventLE.classes_ == 'II')[0][0],
            'Ia' : np.where(self.eventLE.classes_ == 'Ia')[0][0],
            'Ib' : np.where(self.eventLE.classes_ == 'Ib')[0][0]
        }


        self.class2Model.fit(self.PCs, self.class2LE.transform(self.class2_y))
        self.eventModel.fit(self.eventX, self.eventLE.transform(self.event_y))
        

    def predict(self, X_test, pca=True, return_probs=False):
        if pca:
            testPCs = self.PCA(X_test)
        else:
            testPCs = X_test

        event_preds = self.eventModel.predict(testPCs)
        event_labels = self.eventLE.inverse_transform(event_preds)

        class2_preds = self.class2Model.predict(testPCs)
        class2_labels = self.class2LE.inverse_transform(class2_preds)

        preds = np.array([])
        probs = np.array([])

        event_probs = self.eventModel.predict_proba(testPCs)
        class2_probs = self.class2Model.predict_proba(testPCs)
        

        for i, cls in enumerate(class2_labels):
            if cls == 'event':
                preds = np.append(preds, event_labels[i])
                probs = np.append(probs, class2_probs[i][class2_preds[i]])
            else:
                preds = np.append(preds, class2_labels[i])
                probs = np.append(probs, 1 - class2_probs[i][class2_preds[i]])

        if return_probs:
            return [preds, probs]

        return preds
    
    def predict_and_write(self, X_test, ids):
        preds, probs = self.predict(X_test, return_probs=True)
        reverseMap = { v: k for k, v in self.classMap.items() }
        preds_mapped = [reverseMap[pred] for pred in preds]

        output = pd.DataFrame({ 'id': ids, 'class4': preds_mapped, 'p': probs })
        output.to_csv('models/pca_svc_lr.csv', index=False)


data = pd.DataFrame(np.random.rand(10, 3))
binary_accuracy = np.array([])
perplexity = np.array([])
multi_class_accuracy = np.array([])
combined_score = np.array([])

models1 = {
    'SVC': SVC(probability=True),
    'LR': LR(max_iter=500),
    'KNN': KNN(n_neighbors=8),
    'RF': RF()
}

models2 = {
    'SVC': SVC(probability=True),
    'LR': LR(max_iter=500),
    'KNN': KNN(n_neighbors=8),
    'RF': RF()
}
print(
    '| class2 model',
    'event model',
    'Binary Accuracy',
    'Perplexity',
    'Multi-Class Accuracy',
    'Combined Score',
    sep=' | ',
    end=' |\n'
)
print('|---|---|---|---|---|---|')
for class2ModelName, class2Model in models1.items():
    for eventModelName, eventModel in models2.items():
        for validChunkIndex in np.array_split(trainDF.index, 10):
            validChunk = trainDF.loc[validChunkIndex]

            trainChunk = trainDF.drop(validChunk.index.values)
            model = Model(4, class2Model, eventModel)
            model.fit(trainChunk.drop(columns=['date', 'id', 'class4']), trainChunk['class4'])

            preds, probs = model.predict(validChunk, return_probs=True)

            submission = pd.DataFrame({
                'id': validChunk['id'].values,
                'class4': preds,
                'p': probs
            })
            solution = pd.DataFrame({
                'id': validChunk['id'].values,
                'class4': validChunk['class4'].values
            })
            comb_score, bin_acc, perp, mult_c_acc = score(solution, submission, 'id')
            binary_accuracy = np.append(binary_accuracy, bin_acc)
            perplexity = np.append(perplexity, perp)
            multi_class_accuracy = np.append(multi_class_accuracy, mult_c_acc)
            combined_score = np.append(combined_score, comb_score)
        
        print(
            f'| {class2ModelName}',
            eventModelName,
            f'{binary_accuracy.mean():.5f}',
            f'{perplexity.mean():.5f}',
            f'{multi_class_accuracy.mean():.5f}',
            f'{combined_score.mean():.5f}',
            sep=' | ',
            end=' |\n'
        )

#model = Model(PCA_depth=4).fit(trainDF.filter(regex=r'\.mean$'), class4).predict_and_write(testDF, ids= testDF['id'])
#print(cross_val_score(model, trainDF.drop(columns=['date', 'id', 'class4']), class4, cv=10).mean())
