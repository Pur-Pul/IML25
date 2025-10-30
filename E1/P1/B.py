import pandas as pd

#Task b
df = pd.read_csv("data/train.csv")
print(df[["T84.mean","UV_A.mean","CS.mean"]].describe())
