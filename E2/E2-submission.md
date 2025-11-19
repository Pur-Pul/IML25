# Problem 8
## Task a
1. The 'species' column in both the training and testing data is replaced with the 'Adelie' column where 1 means Adelie and 0 is notAdelie. 
2. The intercept term is added to both the training X values and testing X values. 
3. `sm.Logit` is used to train a logistin regression model with the training data.
4. The accuracy is carculated by calling `results.predict()` on both data sets.
    - Probabilities over 0.5 are classified as Adelie.
5. A scatter plot is used to best show the distribution of the data.

The red data points are Adelie and the blu are not.

### Output
```bash
Coefficients
const                179.558317
bill_length_mm       -23.201690
bill_depth_mm         38.308170
flipper_length_mm     -0.039799
body_mass_g            0.036815
dtype: float64

Train accuracy: 1.0
Test accuracy: 0.9866666666666667
```

### Plot
![Probability plot](../../static/E2P8A.png)

### code
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

trainDF = pd.read_csv('data/penguins_train.csv')
trainDF['Adelie'] = (trainDF['species'] == 'Adelie').astype(int)
trainDF = trainDF.drop(columns = ['species'])

testDF = pd.read_csv('data/penguins_test.csv')
testDF['Adelie'] = (testDF['species'] == 'Adelie').astype(int)
testDF = testDF.drop(columns = ['species'])

trainX = trainDF[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
trainX = sm.add_constant(trainX)

testX = testDF[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
testX = sm.add_constant(testX)

results = sm.Logit(trainDF['Adelie'], trainX).fit()

trainAccuracy = ((results.predict(trainX) > 0.5) == trainDF['Adelie']).mean()
testAccuracy = ((results.predict(testX) > 0.5) == testDF['Adelie']).mean()

print('Coefficients')
print(results.params)
print()
print('Train accuracy:', trainAccuracy)
print('Test accuracy:', testAccuracy)

linear_response = np.dot(trainX, results.params)
probs = results.predict(trainX)

col = np.where(trainDF['Adelie']==0,'b','r')

plt.scatter(linear_response, probs, color=col)
plt.show()
```

## Task b

### Output
```bash
Coefficients
const                0.000000
bill_length_mm      -0.084968
bill_depth_mm        0.557587
flipper_length_mm   -0.018480
body_mass_g         -0.000746
dtype: float64

Train accuracy: 0.9333333333333333
Test accuracy: 0.8933333333333333
```

### Plot
![Probability plot](../../static/E2P8B.png)

- The code works the same as in task a, but `GLM` is used instead of `Logit`.
- The model is fitted with the `fit_regularized` function with the parameters `alpha=0.1` and `L1_wt=1.0`.
- `L1_wt=1.0` makes sure that lasso fit is used instead of ridge fit.
- `alpha=0.1` was tweaked until one coefficient (the intercept) became 0.

### code
```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

trainDF = pd.read_csv('data/penguins_train.csv')
trainDF['Adelie'] = (trainDF['species'] == 'Adelie').astype(int)
trainDF = trainDF.drop(columns = ['species'])
testDF = pd.read_csv('data/penguins_test.csv')
testDF['Adelie'] = (testDF['species'] == 'Adelie').astype(int)
testDF = testDF.drop(columns = ['species'])

trainX = trainDF[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
trainX = sm.add_constant(trainX)

testX = testDF[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
testX = sm.add_constant(testX)

model = sm.GLM(trainDF['Adelie'], trainX, family=sm.families.Binomial())
results = model.fit_regularized(alpha=0.1, L1_wt=1.0)

print('Coefficients')
print(results.params)
print()

trainAccuracy = ((results.predict(trainX) > 0.5) == trainDF['Adelie']).mean()
testAccuracy = ((results.predict(testX) > 0.5) == testDF['Adelie']).mean()


print('Train accuracy:', trainAccuracy)
print('Test accuracy:', testAccuracy)

linear_response = np.dot(trainX, results.params)
probs = results.predict(trainX)

col = np.where(trainDF['Adelie']==0,'b','r')

plt.scatter(linear_response, probs, color=col)
plt.show()
```

## Task c
I got the following warnings in task a:
```bash
warnings.warn(msg, category=PerfectSeparationWarning)
... : PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
Warning: Maximum number of iterations has been exceeded.
         Current function value: 0.000000
         Iterations: 35
ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
```
The first warning apperares multiple times each time the code is run, which is due to the likelyhood optimization being iterative.

The problem is that there is a a quasi-complete separation in some of the variables between the two classes. For example if we look at the dataset `penguins_train.csv` we see that the `bill_length_mm` variable has only a small overlap between the two species.
| Species   | bill_length_mm range |
|-----------|----------------------|
| Adelie    |        33 - 44       |
| NotAdelie |        43 - 60       |
This means the bill length can predict the species perfectly except for the range 43 - 44. This casues the maximum likelihood estimation to fail for the variable.

# Problem 9


$$
\Pr(Y=k|X=x)=\frac{\pi_k f_k(x)}{\sum^2_{l=1}\pi_l f_l(x)}
$$

$$
f_k(x)=\frac{1}{\sqrt{2\pi}\sigma_k}\exp(-\frac{1}{2\sigma^2_k}(x-\mu_k)^2)
$$

If $p_1(x)>p_2(x)$ then class 1 is selected. The denominators for $p_1(x)$ and $p_2(x)$ are the same and cancel out.

$$
\pi_1 f_1(x)>\pi_2 f_2(x)
$$
Logs are monotone which means that the current comparasion 
$\pi_1 f_1(x) > \pi_2 f_2(x)$ is essentially the same as $log{\pi_1} + log{f_1(x)} > log{\pi_2} + log{f_2(x)}$ or in other words $\delta_1(x) > \delta_2(x)$

Plugging in the function $f_1(x)$ and $f_2(x)$ results in

$$
\log{\pi_1} - \log{(\sqrt{2\pi}\sigma_1)} -\frac{1}{2\sigma^2_1}(x-\mu_1)^2 > \log{\pi_2} - \log{(\sqrt{2\pi}\sigma_2)} -\frac{1}{2\sigma^2_2}(x-\mu_2)^2
$$

$\log{(\sqrt{2\pi}\sigma_k)}=\frac{1}{2}\log(2\pi)+\log\sigma_k$ and the $\frac{1}{2}\log(2\pi)$ is a common factor and cancels out. Furthermore $\log\sigma_k$ can be rewritten as $\frac{1}{2}\log{\sigma_k^2}$. Which results in:

$$
\log{\pi_1} - \frac{1}{2}\log{\sigma_1^2} -\frac{1}{2\sigma^2_1}(x-\mu_1)^2 > \log{\pi_2} - \frac{1}{2}\log{\sigma_2^2} -\frac{1}{2\sigma^2_2}(x-\mu_2)^2
$$

This can be turned into the difference between the discriminants
$$
\log{\pi_1} - \log{\pi_2} - (\frac{1}{2}\log{\sigma_1^2} - \frac{1}{2}\log{\sigma_2^2}) - (\frac{1}{2\sigma^2_1}(x-\mu_1)^2 - \frac{1}{2\sigma^2_2}(x-\mu_2)^2)
$$
Which can be rewritten as:
$$
\log{\frac{\pi_1}{\pi_2}} - \frac{1}{2}(\log{\sigma_1^2} - \log{\sigma_2^2})

- \frac{1}{2}(\frac{(x-\mu_1)^2}{\sigma^2_1} - \frac{(x-\mu_2)^2}{\sigma^2_2})
$$
The decision boundary between class 1 and class 2 is when the difference between their discriminants is 0. 
$$
\log{\frac{\pi_1}{\pi_2}} - \frac{1}{2}(\log{\sigma_1^2} - \log{\sigma_2^2}) - \frac{1}{2}(\frac{(x-\mu_1)^2}{\sigma^2_1} - \frac{(x-\mu_2)^2}{\sigma^2_2})=0
$$
By doubling to remove the $\frac{1}{2}$ and then rearranging:
$$
2\log{\frac{\pi_1}{\pi_2}} - \log{\frac{\sigma_1^2}{\sigma_2^2}} = (\frac{(x-\mu_1)^2}{\sigma^2_1} - \frac{(x-\mu_2)^2}{\sigma^2_2})
$$
By expanding the $(x-\mu_k)^2$ lefthand side of the equation becomes:
$$
(\frac{1}{\sigma^2_1} - \frac{1}{\sigma^2_2})x^2+(\frac{2\mu_1}{\sigma^2_1}+\frac{2\mu_2}{\sigma^2_2})x+\frac{\mu^2_1}{\sigma^2_1}-\frac{\mu^2_2}{\sigma^2_2}
$$
And then the equation can be rearranged into:
$$
(\frac{1}{\sigma^2_1} - \frac{1}{\sigma^2_2})x^2+(\frac{2\mu_1}{\sigma^2_1}+\frac{2\mu_2}{\sigma^2_2})x+\frac{\mu^2_1}{\sigma^2_1}-\frac{\mu^2_2}{\sigma^2_2}-2\log{\frac{\pi_1}{\pi_2}} + \log{\frac{\sigma_1^2}{\sigma_2^2}} = 0
$$

We can see that the quadratic coefficient is $\frac{1}{\sigma^2_1} - \frac{1}{\sigma^2_2}$. If we assume that $\sigma_1 \neq \sigma_2$ then the quadratic coefficient is non-zero, which means that the decision boundary equation is quadratic and not linear.

# Problem 10
## Task a
### Output
Adelie
   stat  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
0  mean        38.12400       18.33600         188.880000  3576.000000
1   std         2.72533        1.17979           6.192382   452.022123
prob: 0.33766233766233766

not Adelie
   stat  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g
0  mean       47.818000      15.890000         211.300000  4657.000000
1   std        3.563296       1.945688          11.674331   779.615931
prob: 0.6623376623376623
### Code
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import ceil, sqrt, pi, exp

trainDF = pd.read_csv('data/penguins_train.csv')
testDF = pd.read_csv('data/penguins_test.csv')

#Task a
def getAttributeStats(df, column, y):
    classDF = df[df[column] == y].drop(columns=[column])
    stats = pd.DataFrame({'stat': ['mean', 'std']})
    for attr in classDF.columns.values:
        stats[attr] = [np.mean(classDF[attr]), np.std(classDF[attr])] 
        
    return stats

def laplaceSmooth(df, column, y):
    classDF = df[df[column] == y].drop(columns=[column])
    return (len(classDF) + 1) / (len(df) + 2)

print('Adelie')
print(getAttributeStats(trainDF, 'species', 'Adelie'))

print('prob:', laplaceSmooth(trainDF, 'species', 'Adelie'))

print()
print('not Adelie')
print(getAttributeStats(trainDF, 'species', 'notAdelie'))

print('prob:', laplaceSmooth(trainDF, 'species', 'notAdelie'))
print()
```

## Task b
The 4 features in data (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) are quantitative, which means that a normal distribution can be assumed. The density function becomes:
$$
f_k(x_j)=\frac{1}{\sqrt{2\pi}\sigma_{jk}}\exp(-\frac{1}{2\sigma_{jk}^2}(x-\mu_{jk})^2)
$$
where $\sigma_k$, $\sigma_k^2$ and $\mu_k$ are the standard deviation, variance and mean for the jth feature in the kth class.

The formula for the posterior probability is:
$$
\Pr(Y=k|X=x)=\frac{\pi_k \times f_{k1}(x_1) \times f_{k2}(x_2) \times ... \times f_{kp}(x_p) }{\sum_{l=1}^K \pi_l \times f_{l1}(x_1) \times f_{l2}(x_2) \times ... \times f_{lp}(x_p)}
$$

The the density function can be plugged into the posterior probability function, which results in:
$$
\Pr(Y=k|X=x)=\frac{\pi_k \times \prod_{j=1}^p \frac{1}{\sqrt{2\pi}\sigma_{jk}}\exp(-\frac{1}{2\sigma_{jk}^2}(x-\mu_{jk})^2)}{\sum_{l=1}^K \pi_l \times \prod_{j=1}^p \frac{1}{\sqrt{2\pi}\sigma_{jl}}\exp(-\frac{1}{2\sigma_{jl}^2}(x-\mu_{jl})^2)}
$$

## Task c
### Code
The following code continues from the code in task a.
```python
#Task c
def density(x, mean, std):
    return exp(-((x - mean) ** 2) / (2 * std**2)) / (sqrt(2*pi) * std)

def postProb(df, column, y, X):
    numerator = laplaceSmooth(df, column, y)
    for attr in X.columns.values:
        stats = getAttributeStats(df, column, y)
        mean = stats[stats['stat'] == 'mean'][attr].iloc[0]
        std = stats[stats['stat'] == 'std'][attr].iloc[0]
        x = X[attr].iloc[0]

        numerator *= density(x, mean, std)
    
    classes = df[column].unique()
    denominator = 0
    for c in classes:
        product = laplaceSmooth(df, column, c)
        stats = getAttributeStats(df, column, c)
        for attr in X.columns.values:
            mean = stats[stats['stat'] == 'mean'][attr].iloc[0]
            std = stats[stats['stat'] == 'std'][attr].iloc[0]
            x = X[attr].iloc[0]
            product *= density(x, mean, std)
        denominator+=product
    
    return numerator / denominator

X = testDF.drop(columns=['species'])
Y = testDF['species']
preds = []
for i in range(0,len(testDF)):
    prob = postProb(trainDF, 'species', 'Adelie', X.iloc[[i]])

    preds.append('Adelie' if prob >= 0.5 else 'notAdelie')
    if i < 3:
        print(f'p^(y = Adelie | x_{i}) =', prob)
print('Accuracy:', np.mean(testDF['species'].values == np.array(preds)))
```

### Output
p^(y = Adelie | x_0) = 0.9967896646386412
p^(y = Adelie | x_1) = 0.7953470279474658
p^(y = Adelie | x_2) = 0.998919564529168
Accuracy: 0.9333333333333333

# Problem 11
## Task a
Discriminative learning is considered to be better for larger datasets while generative ones may be better when the dataset is small. This is because while discriminative learning has generally has a lower asymptotic error, generative learning reaches its asymptotic error earlier. This means that as the dataset increases, generative classifiers reach their final performance before discriminative ones do, which may mean that they are better for smaller sets.

## Task b
Two different parametric families are discussed 'Continuous inputs' and 'Discrete inputs'. The following models are chosen for $h_{Gen}$ and $h_{Dis}$ for the families:
| Family            | $h_{Gen}$                     | $h_{Dis}$           |
|-------------------|-------------------------------|---------------------|
| Continuous inputs | Normal discriminant analysis  | Logistic regression |
| Discrete inputs   | Naive Bayes                   | Logistic regression |

$h_{Gen}$ is fit to optimize the joint likelyhood of the input/features $p(\textnormal{features}, \textnormal{targets})$ and labels/targets. $h_{Dis}$ is fit to either optimize the conditinal likelyhood $p(\textnormal{target} | \textnormal{features})$ or to minimize the training error.

## Task c
For discrete inputs the Naive Bayes performs better most of the time with a few exceptions where the training dataset has grown enough for logistic regression to overtake it. Continuous datasets in comparasion show more scenarios where logistic regression overtakes Naive Bayes as the training dataset grows. 