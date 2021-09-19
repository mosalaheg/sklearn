"""
This file is about how to load data from sklearn.

Library : sklearn
Moduale : datasets
Class   : load_wine
Object  : wine_data
data    : Wine Data

"""

# Import important libraries
from sklearn.datasets import load_wine

# Object of load_wine
wine_data = load_wine()

# Get X data and its information as shape, features, features_names and printing them
X = wine_data.data
X_shape = X.shape
feature_names = wine_data.feature_names

print('X Data is \n' , X[:10])
print('X shape is ' , X_shape)
print('X Features are \n' , feature_names)


# Get y data and its information as shape, features, features_names and printing them
y = wine_data.target
y_shape = y.shape
target_names = wine_data.target_names

print('y Data is \n' , y[:170])
print('y shape is ' , y_shape)
print('y Columns are \n' , target_names)
