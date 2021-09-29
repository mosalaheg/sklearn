# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:16:23 2021

@author: Mohamed Salah
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("houses.csv")
#print(dataset.head(10))

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 28)
print(X_train, X_test, y_train, y_test )

linear = LinearRegression(fit_intercept= True,normalize=True, copy_X=True, n_jobs=-1)

# Get the best fit line and thetas 
linear.fit(X_train, y_train)

# Predict and get prediction data
y_predict = linear.predict(X_test) 

# Train score
train_score = linear.score(X_train, y_train)

# Test Score
test_score = linear.score(X_test, y_test)

print("Predictions", y_predict[:10])
print("Train Score", train_score)
print("Test Score", test_score)

# Plotting the data
plt.scatter(X.iloc[:, 0], y, color = 'red')
plt.scatter(X_test.iloc[:, 0], y_test, color = 'green')
plt.plot(X_train.iloc[:, 0], linear.predict(X_train), color = 'blue')
plt.title('SAT degrees')
plt.xlabel('high_GPA')
plt.ylabel('univ_GPA')
plt.show()
