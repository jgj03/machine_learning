#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('/home/........./Data/studentscores.csv',)
X=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values
#splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#simple linear regressor
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(X_train,y_train)
#predicting for a new value
linear_regressor.predict(X_test)

#Plotting: visualization
plt.scatter(X_train,y_train,c='r')
plt.plot(X_train,linear_regressor.predict(X_train),c='b')
plt.xlabel('yoe'),plt.ylabel('Sal')
#test results
plt.scatter(X_test,y_test,c='g')
plt.plot(X_test,linear_regressor.predict(X_test),c='b')
plt.xlabel('yoe'),plt.ylabel('Sal')
