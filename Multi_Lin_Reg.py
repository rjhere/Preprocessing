# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:30:18 2018

@author: Rishil
"""

# Multiple Linear Regression

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#First change the current working directory to where the code and data is placed
print(os.getcwd())
path="C:\\Users\\rjpun\\Documents\\ML a-z\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression"
os.chdir(path)
#dataset = pd.read_csv("C:\\Users\\rjpun\\Documents\\ML a-z\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\50_Startups.csv")
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#print(dataset.columns)