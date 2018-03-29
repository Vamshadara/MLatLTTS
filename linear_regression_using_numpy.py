# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:01:01 2018

@author: Vamshadara Solium
"""

import numpy as np
import matplotlib.pyplot as plt

#Train Data 
trainData_x = np.array([1,2,3,4,5])
trainData_y = np.array([2,4,6,8,10])



#Create the polyfit
z           = np.polyfit(trainData_x, trainData_y, 1)

# Polynomial
p           = np.poly1d(z)

#Float test Data Off-axis test data
testData_int   = np.arange(0,30,1)
testData_float = np.random.rand(30)
testData_x    =  testData_int + testData_float

testData_y = p(testData_x)

#plots 
plt.scatter(testData_x, testData_y, color = 'green')
plt.plot(testData_x, testData_y, color = 'black')
plt.show()



