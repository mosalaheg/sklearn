# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:11:34 2021

@author: Mohamed Salah
"""


# Import libraries and methods
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np



# Data Preparation
## Data Selection
dataset = pd.read_csv("houses.csv")
#print(dataset)
X = dataset.iloc[ : , :-1]
y = dataset.iloc[ : , -1]
# print(y)
# print(X)
## Data Cleaning
# imputed_module = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputed_X = imputed_module.fit(X)
# X = imputed_X.transform(X)
# imputed_module = SimpleImputer(missing_values='nan', strategy='mean')
# imputed_y = imputed_module.fit(y)
# y = imputed_y.transform(y)

## Data Scaling
X = StandardScaler(copy= True, with_mean=True, with_std=True).fit_transform(X)
# print(X)

## Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

# Model Building
ridge_model = Ridge(alpha = 1)

# Fit Model
ridge_model.fit(X_train, y_train)

# Get Attribute Values
train_score = ridge_model.score(X_train, y_train)
test_score = ridge_model.score(X_test, y_test)
thetas = ridge_model.coef_
y_intercept = ridge_model.intercept_

# Predict
y_pred = ridge_model.predict(X_test)

# Get Accuracy

MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values

MdSEValue = median_absolute_error(y_test, y_pred)





