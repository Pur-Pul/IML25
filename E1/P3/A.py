import matplotlib.pyplot as plt
import numpy as np

def testError(x):
    return 1/4 + ((x * 2 - 1.5) ** 2)/3

def trainError(x):
    return 0.8 * (1 - x) ** 6



x = np.linspace(0, 100, 100)

plt.plot(x, trainError(x/100), label='Training error')
plt.plot(x, testError(x/100), label='Testing error')
plt.xlabel('Flexibility (%)')
plt.legend()
plt.show()