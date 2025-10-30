import pandas as pd
import matplotlib.pyplot as plt

#Task d
df = pd.read_csv("data/train.csv")
plt.title('Task d')
plt.subplot(1,2,1)
plt.bar(df["class4"].value_counts().index.values, df["class4"].value_counts().values)
plt.xlabel("Events")
plt.ylabel("Counts")
plt.subplot(1,2,2)
plt.hist(df["CO242.mean"].values, bins=30)
plt.xlabel("CO242 Mean")
plt.show()
