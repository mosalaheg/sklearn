"""
This file is about how to load data from sklearn.

Library : sklearn
Moduale : datasets
Class   : load_digits
Object  : digits_data
data    : Digits Data

"""
# Import important libraries
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# Object of load_digits
digits_data = load_digits()

# Get X data and its information as shape, features, features_names and printing them
X = digits_data.data
X_shape = X.shape
feature_names = digits_data.feature_names

print('X Data is \n' , X[10])
print('X shape is ' , X_shape)
print('X Features are \n' , feature_names)


# Get y data and its information as shape, features, features_names and printing them
y = digits_data.target
y_shape = y.shape
target_names = digits_data.target_names

print('y Data is \n' , y)
print('y shape is ' , y_shape)
print('y Columns are \n' , target_names)

# Plotting the data
plt.gray()
for g in range(10):
    print('Images of Number : ' , g)
    plt.matshow(digits_data.images[g])
    print('------------------------------')
    plt.show()
