"""
This file is about how to load data from sklearn.

Library : sklearn
Moduale : datasets
Class   : load_iris
Object  : iris_data
data    : Iris Data

"""

# Import important libraries
from sklearn.datasets import load_iris

# Object of load_iris
iris_data = load_iris()

# Get X data and its information as shape, features, features_names and printing them
X = iris_data.data
X_shape = X.shape
feature_names = iris_data.feature_names

print('X Data is \n' , X)
print('X shape is ' , X_shape)
print('X Features are \n' , feature_names)

# Get y data and its information as shape, features, features_names and printing them
y = iris_data.target
y_shape = y.shape
target_names = iris_data.target_names

print('y Data is \n' , y)
print('y shape is ' , y_shape)
print('y Columns are \n' , target_names)