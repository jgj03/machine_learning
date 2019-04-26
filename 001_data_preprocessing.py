#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import numpy as np
import pandas as pd

# READING DATASET AND SPLITTING INPUT & OUTPUT
dataset=pd.read_csv('Data.csv',delimiter=',') #Reading dataset
X=dataset.iloc[:,:-1].values #Input dataset
y=dataset.iloc[:,3].values #Output dataset

#HANDLING MISSING DATA
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0) #axis=0:impute along columns, 1:rows
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
# Imputer replaced with SimpleImputer in newer versions
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy="mean") # missing values if any
#imputer=SimpleImputer(missing_values="NaN",strategy="mean") 
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#ENCODING CATEGORICAL DATA
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#TRAINING AND TESTING DATASET SPLITTING
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#FEATURE SCALLING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
