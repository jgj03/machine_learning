#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data and separating as input and output
data=pd.read_csv('/home/....../Data/D6Social_Network_Ads.csv')
X=data.iloc[:,2:4].values
y=data.iloc[:,4:5].values

# Splitting into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting the KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p = 2)
#p:Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
knn.fit(X_train,y_train)

#predicting for test set
y_pred = knn.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm) # 

#For visualization
age = data.iloc[:, 2].values
sal = data.iloc[:, 3].values
#exploratory plot
plt.scatter(age, sal, color = ['green' if i else 'red' for i in y])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#plotting
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min() - 1, 
                               X_set[:,0].max() + 1, 
                               step = 0.001),
                     np.arange(X_set[:,1].min() - 1, 
                               X_set[:,1].max() + 1, 
                               step = 0.001))
boundary = knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1, X2, boundary, alpha = 0.75, 
             cmap = ListedColormap(('#fc7a74', '#6ff785')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), 
                label = j, s = 8)
plt.title('K-NN classifier')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
