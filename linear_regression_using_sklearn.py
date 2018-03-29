# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:52:42 2018

@author: vs49000
"""

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

#training data
trainData_x = np.array([1, 2, 3, 4, 5])
trainData_x = trainData_x.reshape(-1,1)
trainData_y = np.array([3, 6, 9, 12, 15])
trainData_y = trainData_y.reshape(-1,1)


#testing data
testData_x = np.array([0,2.5,3.5])
testData_x = testData_x.reshape(-1,1)

# create the linear regression object
regr = linear_model.LinearRegression()

#train the model
regr.fit(trainData_x,trainData_y)

#predict using the testing data
predict_y = regr.predict(testData_x)

#the coefficients
print('the coefficents : \n',regr.coef_)

#plot the outputs
plt.scatter(trainData_x, trainData_y, color= 'black')
plt.plot(testData_x, predict_y, color='blue')
plt.show()
