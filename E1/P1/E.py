import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#Task e
df = pd.read_csv("data/train.csv")
g = seaborn.pairplot(df[["UV_A.mean","T84.mean","H2O84.mean"]])
g.fig.suptitle("Task e")
plt.show()
