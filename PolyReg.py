import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X = 6 * np.random.rand(200, 1) - 3
y = 0.8 * X**2 + 0.9*X + 2 + np.random.randn(200, 1)
# y = 0.8x^2 + 0.9x + 2

plt.plot(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

plt.plot(X_train, lr.predict(X_train), color="r")
plt.plot(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.show()

poly = PolynomialFeatures(degree=2)
X_train_trans = poly.fit_transform(X_train)
X_test_trans = poly.transform(X_test)

print(X_train)
print(X_train_trans[0])

lr.fit(X_train_trans, y_train)
y_pred = lr.predict(X_test_trans)

print(r2_score(y_test, y_pred))
print(lr.coef_)
print(lr.intercept_)

X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
y_new = lr.predict(X_new_poly)

plt.plot(X_new, y_new, "r", linewidth=2, label="Prediction")
# plt.plot(X_train, y_train, "b", label="Training Points")
# plt.plot(X_test, y_test, "g", label="Testing Points")
# plt.plot(X, y, "black")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
