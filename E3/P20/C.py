import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

scaler = StandardScaler(with_mean=True, with_std=True)

df = pd.read_csv('data/train.csv')
X = df.filter(regex=r'\.mean$')
y = df['class4']

trainX, validX, train_y, valid_y = train_test_split(X, y, test_size=0.5, random_state=1)
scaledX = scaler.fit_transform(X)

knn = KNN(n_neighbors=1)
knn.fit(trainX, train_y)

print('1-NN accuracy', accuracy_score(valid_y, knn.predict(validX)))

accuracies = np.array([])



for d in range(2, 51):
    pca = PCA(n_components=d)
    scores = pca.fit_transform(scaledX)
    pcaX = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])

    pcaTrainX, pcaValidX, pcaTrain_y, pcaValid_y = train_test_split(pcaX, y, test_size=0.5, random_state=1)

    knn.fit(pcaTrainX, pcaTrain_y)
    accuracies= np.append(accuracies, accuracy_score(pcaValid_y, knn.predict(pcaValidX)))
    if d == 4:
        print('1-NN accuracy with PCA', accuracy_score(pcaValid_y, knn.predict(pcaValidX)))

plt.plot(np.linspace(2, 50, 49), accuracies)
plt.show()

