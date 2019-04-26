#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ """

import pandas as pd

#Reading the data and splitting input and output
data=pd.read_csv('/home/...../Data/50-Startups.csv')
X=data.iloc[:,:4].values #or X=data.iloc[:,:-1].values
y=data.iloc[:,4:5].values #or y=data.iloc[:,:4].values

#Categorical variabel treatment 
import sklearn
help(sklearn) # and 
dir(sklearn)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dir(sklearn.preprocessing.LabelEncoder) # list all the functions
help(sklearn.preprocessing.LabelEncoder) # with details
le=LabelEncoder()
ohe=OneHotEncoder()
X[:,3]
X[:,3]=le.fit_transform(X[:,3])
X[:,3]
ohe=OneHotEncoder(categorical_features=[3])
ohe
X=ohe.fit_transform(X).toarray() #use the ColumnTransformer instead: Deprication
print(X)

#Avoiding nummy variable trap: since only 2 variables(2 countries) are there
X = X[: , 1:]

#Splitting dataset
from sklearn.model_selection import train_test_split # random_state=42  can be given
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Fitting Linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
#predict
y_pred=lr.predict(X_test)

