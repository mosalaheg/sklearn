"""
This file is about how to load data from sklearn.

Library : sklearn
Moduale : datasets
Class   : load_boston
Object  : boston_data
data    : Boston Data For House Prices

"""

# Import important libraries
from sklearn.datasets import load_boston

# Object of load_boston
boston_data = load_boston()

# Get X data and its information as shape, features, features_names and printing them
X = boston_data.data
X_shape = X.shape
feature_names = boston_data.feature_names

print('X Data is \n' , X[:10])
print('X shape is ' , X_shape)
print('X Features are \n' , feature_names)


# Get y data and its information as shape, features, features_names and printing them
y = boston_data.target
y_shape = y.shape
#target_names = boston_data.target_names

print('y Data is \n' , y[:10])
print('y shape is ' , y_shape)
#print('y Columns are \n' , target_names)
