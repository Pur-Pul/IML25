import matplotlib.pyplot as plt
import numpy as np

def testError(x):
    return 1/4 + ((x * 2 - 1.2) ** 2)/3

def trainError(x):
    return 0.6 * (1 - x) ** 2

def bias(x):
    return 0.6 * (1 - x) ** 8

def variance(x):
    return 0.6 * x ** 8

def irreducible(x):
    return np.full(100, 0.2)

x = np.linspace(0, 100, 100)

plt.plot(x, testError(x/100), label='Testing error')
plt.plot(x, trainError(x/100), label='Training error')
plt.plot(x, bias(x/100), label='Bias')
plt.plot(x, variance(x/100), label='Variance')
plt.plot(x, irreducible(x/100), label='Irreducible error')

plt.xlabel('Flexibility')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticks([])
frame1.axes.yaxis.set_ticks([])
plt.legend()
plt.show()
