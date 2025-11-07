import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#Task f
trainDF = pd.read_csv("data/train.csv")
event_count = (trainDF['class4'] != 'nonevent').sum()
total = len(trainDF)
common_class = trainDF['class4'].value_counts().idxmax()
probability = event_count / total

testDF = pd.read_csv("data/test.csv")

dummy = pd.DataFrame({
    'id': testDF['id'],
    'class4': common_class,
    'p': probability
})

dummy.to_csv('models/dummy.csv', index=False)