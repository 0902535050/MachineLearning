# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


# Importing the dataset
filePath = os.getcwd() + '\multiLR_data\home.csv'
feature = ['size', 'bedroom', 'price']
my_data = pd.read_csv(filePath, names=feature) #read the data
#chuẩn hóa dữ liệu các feature bằng chuẩn hóa trung bình 
my_data = (my_data - my_data.mean())/my_data.std()
my_data.head()
#thiết lập các ma trận
#gắn cột thứ 2 cho x
X = my_data.iloc[:,0:2]
#gắn 1 mảng cho cột x
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
#chuyển đổi từ dataframe sang numpy array
y = my_data.iloc[:,2:3].values 
#gắn cột thứ 3 cho y
theta = np.zeros([1,3])


#đặt các tham số
alpha = 0.01
iters = 1000 #số vòng lập
#tạo code function
#tính code function
def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))
#tạo gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost



#chạy hàm gradient descent và cost function
g,cost = gradientDescent(X,y,theta,iters,alpha)
print(g)
finalCost = computeCost(X,y,g)
print(finalCost)
#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  #Số lần lặp
ax.set_ylabel('Cost')  #Hàm chi phí
ax.set_title('Error vs. Training Epoch')  #Hàm chi phí và số lần lặp
plt.show()