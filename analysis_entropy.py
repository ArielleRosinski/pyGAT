import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt

h_1 = np.array([-2.2, -1.5, 0.4, 1.6, 2.8])
max_h = np.max(np.abs(h_1))+1


shift = 0


def leakyRelu(h, alpha=0.2):
    return np.where(h > 0, h, h * alpha)

xs = np.linspace(-max_h, max_h, num=101)
ys = [entropy(softmax(leakyRelu(h_1 + x))) for x in xs]

plt.plot(xs, ys)
plt.show()

xs = np.linspace(-max_h, max_h+0.1, 0.1)
for i in range(len(h_1)):
    plt.axvline(x=h_1[i], color='r', linestyle='--')
    ys = [softmax(leakyRelu(h_1 + x))[i] for x in xs]
    plt.plot(xs, ys, label=f'{i}')

plt.show()



