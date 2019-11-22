import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples = 300, n_features=1, noise=8, bias=2)
y2 = y**2
model = LinearRegression()
model.fit(X, y2)
poly_features = PolynomialFeatures(degree = 2)  
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()  
poly_model.fit(X_poly, y2)

pred = poly_model.predict(X_poly)
new_X, new_y = zip(*sorted(zip(X, pred))) # sort values for plotting
plt.plot(new_X, new_y)
plt.scatter(X,y2)
plt.title("Polynomial Degree = 2")
plt.show()