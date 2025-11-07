# Problem 1
## Task a
The csv is read with pandas and `.drop()` is called to remove the columns `id` and `partlybad`.
```python
import pandas as pd

#Task a
df = pd.read_csv("data/train.csv")
df = df.drop(columns=["id", "partlybad"])
```
## Task b
A subsection of the data frame that includes the the columns `T84.mean`, `UV_A.mean` and `CS.mean` is selected and then `.describe()` is called to generate the summary.
```python
import pandas as pd

#Task b
df = pd.read_csv("data/train.csv")
print(df[["T84.mean","UV_A.mean","CS.mean"]].describe())

```
## Task c
The `T84.mean` column is selected and its values are obtained as Numpy array. `.mean()` is called on the Numpy array to calculate the mean and `.std()` to calculate the standard deviation.
```python
import pandas as pd

#Task c
df = pd.read_csv("data/train.csv")
t84Arr = df["T84.mean"].values
print("t84 numpy mean:", t84Arr.mean())
print("t84 numpy std:", t84Arr.std())
```
## Task d
To make the charts `matplotlib.pyplot` is used. The two charts are independently created and `pyplot.subplot()` is used to place them next to eachother. After both charts are created `pyplot.show()` is called to display the charts.
### Bar chart
For the bar chart the occurances of events in `class4` column are obtained with `value_counts()`. Then `.index.values` is used to get an array containing the unique events and `.values` to obtain the occurances as an array in the same order as the unique event array. These arrays are the fed to `pyplot.bar()`
### Histogram
For the histogram the values in the `CO242.mean` column are obtained with `.values`, which are then fed to the `pyplot.hist()` function. The `hist()` function also recieves `bins=30` in order to increase the number of bars in the histogram.
```python
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
plt.show(a)
```
### Plot
![Bar chart and histogram](../static/E1P1D.png)
## Task e
The `UV_A.mean`, `T84.mean` and `H2O84.mean` columns are selected and fed to `seaborn.pairplot()`. `pyplot.show()` is then called to display the plot.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

#Task e
df = pd.read_csv("data/train.csv")
g = seaborn.pairplot(df[["UV_A.mean","T84.mean","H2O84.mean"]])
g.fig.suptitle("Task e")
plt.show()
```
### Plot
![Scatter plot](../static/E1P1E.png)
## Task f
The number of events that are not `nonevent` are counted and the the total number of evetns are obtained. The probability is then calculated as the not `nonevent` count dividec by the total event count. To get the most common event the number of occurances of each event are calculated with `value_counts()` and then the event with the most occurances is obtained using `.idsmax()`. Then a new data frame if created by making a row for each id in the test data and inserting the most common class and probability on each row. This new data frame is then exported as a .csv file. This dummy model obtained a score of 0.36269.
```python
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
```

# Problem 2
## Task a
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from ISLP.models import (ModelSpec as MS, poly, sklearn_sm)
from sklearn.model_selection import (cross_validate, KFold, ShuffleSplit)

def MSE(model, x, y):
    return ((y - model.predict(x))**2).sum() / len(x)

trainDF =   pd.read_csv("data/train_syn.csv")
testDF =    pd.read_csv("data/test_syn.csv")
validDF =   pd.read_csv("data/valid_syn.csv")
trvaDF =    pd.concat([trainDF, validDF], ignore_index=True)

H = np.array(trvaDF['x'])
M = sklearn_sm(sm.OLS)

print('| Degree\t| Train\t\t\t| Validation\t\t| Test\t\t\t| TestTRVA\t\t| CV\t\t\t|')
print('|---------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|')
for p in range(0, 9):
    if p == 0:
        trainPolyDF = pd.DataFrame({ 'const': np.ones(len(trainDF)) })
        testPolyDF = pd.DataFrame({ 'const': np.ones(len(testDF)) })
        validPolyDF = pd.DataFrame({ 'const': np.ones(len(validDF)) })
        trvaPolyDF = pd.DataFrame({ 'const': np.ones(len(trvaDF)) })
    else:
        trainPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(trainDF)
        testPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(testDF)
        validPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(validDF)
        trvaPolyDF = MS([poly('x', degree=p, raw=True)]).fit_transform(trvaDF)
    
    model = sm.OLS(trainDF['y'], trainPolyDF).fit()
    trvaModel = sm.OLS(trvaDF['y'], trvaPolyDF).fit()
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
   
    X = np.power.outer(H, np.arange((p+1)))
    print(
        f'| {p}\t\t|',
        f'{MSE(model,       trainPolyDF, trainDF['y'])}\t|',
        f'{MSE(model,       validPolyDF, validDF['y'])}\t|',
        f'{MSE(model,       testPolyDF, testDF['y'])}\t|',
        f'{MSE(trvaModel,   testPolyDF, testDF['y'])}\t|',
        f'{np.mean(cross_validate(
            M,
            X,
            trvaDF['y'],
            cv=cv
        )['test_score'])}\t|'
    )
```

| Degree        | Train                     | Validation                | Test                      | TestTRVA                  | CV                        |
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| 0             | 18.45958471543744         | 32.3423633890777          | 22.100658138803382        | 21.62022122494303         | 29.47972529619117         |
| 1             | 4.08853509977745          | 7.127843632518778         | 8.87630669045679          | 9.934942592846511         | 7.274738817571967         |
| 2             | 0.21858585824966098       | 0.2937318827499832        | **0.24584672986724493**   | **0.21401610603668839**   | **0.3404256220538727**    |
| 3             | 0.21681898850133235       | **0.2834473943397284**    | 0.29007994916779434       | 0.27511057667345545       | 0.3751058882199807        |
| 4             | 0.11879553073504794       | 0.6247258729089251        | 0.9690780194043266        | 0.22399288732056818       | 0.4662675207669075        |
| 5             | 0.09653224688035783       | 0.5734799830959056        | 4.894836220386019         | 1.0390889971463342        | 0.3952163525272134        |
| 6             | 0.007574103844994087      | 3.416787014798486         | 213.29713998109577        | 0.881470788346449         | 0.39025762755317706       |
| 7             | 0.0049993794743657        | 6.862992665537132         | 1261.988069542342         | 0.27171864062603185       | 0.42803527993883517       |
| 8             | **0.0020824670831305877** | 401.65178036684495        | 154266.87136128993        | 11.223554149572486        | 1.7825998446160007        |

By analyzing the various losses for each polynomial degree, the optimal polynomial degree can be selected as the one that minimizes the loss. I have bolded the minimum MSE for each type of loss in the table above. For the test loss the best polynomial degree is p=2. Without the test set the options are the training loss, validation loss and tenfold cross-validation. The training loss is generally unrelated to the test loss, which can also be observed in the results I obtained. While the validation loss comes close to the same answer as the test loss with polynomial degree p=3 instead of p=2 tenfold CV is a better option due to it acutally resulting in p=2. In other words I would use cross-validation if I did not have a test set.

## b 

# Problem 3

## Task a
### Traing error and test error
Low training error rate does not mean low test error rate. Very flexible methods may have a low training error rate, but a high test error rate whilte the opposite is true for very inflexible methods. Its is however not a linear 