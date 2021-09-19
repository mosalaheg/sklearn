"""
    Cleaning Data:
        1th Step : Import the needed modules.
            from sklearn.impute import SimpleImputer
        2th Step : Make object of the class.
            imputed_module = SimpleImputer(args)
            args:
                - missing_values: Which value that consider it is missing, as nan, 0, 10000000 etc... .
                - strategy: How will the imputer will handle this missing values, there are many ways:
                    - 'mean'
                    - 'mode'
                    - 'median', etc..
        3th Step : Fit the X data with SimpleImputer Object.
            imputedX = imputed_module.fit(X)
        4th Step : Transform X data.
            X = imputedX.transform(X)
"""

# Import the need modules.
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_breast_cancer
import numpy as np

# Cleaning the data

# load breast cancer data

BreastData = load_breast_cancer()

#X Data
X = BreastData.data

#y Data
y = BreastData.target


# SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)

# Make object of the class SimpleImputer. 
imputed_module = SimpleImputer(missing_values = np.nan, strategy='mean')
imputed_X = imputed_module.fit(X)
X = imputed_X.transform(X)

#X Data
print('X Data is \n' , X[:10])

#y Data
print('y Data is \n' , y[:10])