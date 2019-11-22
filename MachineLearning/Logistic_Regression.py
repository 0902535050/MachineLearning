import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import numpy as np
from scipy.optimize import fmin_tnc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


filePath = os.getcwd() + '\logisticR_data\dataset.csv'
# Load data từ file csv
data = pd.read_csv(filePath).values
N, d = data.shape
X = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
x_men = X[y[:,0]==1]
x_women = X[y[:,0]==0]

plt.axis([45, 85, 50, 300])
plt.scatter(x_men[:, 0], x_men[:, 1], marker='o', c='b')
plt.scatter(x_women[:, 0], x_women[:, 1], marker='x', c='r')
plt.xlabel('Cân nặng (kg)')
plt.ylabel('Chiều cao (cm)')
plt.legend(['Đàn ông', 'Phụ nữ'])

#preparing the data for building the model
X = np.c_[np.ones((X.shape[0], 1)), X]
#y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))



class LogisticRegressionUGD:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Tính weighted sum of inputs Similar to Linear Regression
        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, theta):
       
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient, args=(x, y.flatten()))
        self.w_ = opt_weights[0]
        return self

    def predict(self, x):
        theta = self.w_[:, np.newaxis]
        return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100


# Logistic Regression from scratch using Gradient Descent
model = LogisticRegressionUGD()
model.fit(X, y, theta)
accuracy = model.accuracy(X, y.flatten())
parameters = model.w_
print("\n******************************************")
print("The accuracy of the model is: {} %".format(accuracy))
print("The model parameters using Gradient descent")
print (parameters)
print("******************************************\n")

# plotting the decision boundary
# As there are two features
# wo + w1x1 + w2x2 = 0
# x2 = - (wo + w1x1)/(w2)
x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
plt.plot(x_values, y_values, label='Decision Boundary')


# dự đoán kết quả trả về
X1 = pd.read_csv(filePath).values
Nt, dt = X1.shape
X1 = np.c_[X1, np.ones((X1.shape[0], 1), dtype=int)]
a = X1[:, 0]
b = X1[:, 1]
for i in range(Nt):
    xx = model.sigmoid((parameters[0]+parameters[1]*a[i]+parameters[2]*b[i]))*100
    
    print('{0} {1} {2}%'.format(a[i], b[i], xx))
    if xx < 50:
        X1[i,2]=0
print (X1)


#Logistic Regression Of ScikitLearn
model = LogisticRegression()
model.fit(X, y)
parameters = model.coef_
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(),predicted_classes)*100

print("\n******************************************")
print('The accuracy score using scikit-learn is: {} %'.format(accuracy))
print("The model parameters using scikit learn")
print(parameters)
print("******************************************\n")

plt.show()