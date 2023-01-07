import numpy as np
import matplotlib.pyplot as plt


X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9*X + 2 + np.random.randn(200, 1)
# y = 0.8x^2 + 0.9x + 2

plt.plot(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()