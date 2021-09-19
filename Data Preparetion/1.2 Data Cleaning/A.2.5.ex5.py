# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:04:24 2021

@author: Mohamed Salah
"""

from sklearn.datasets import load_boston
from sklearn.impute import  SimpleImputer as simp
import numpy as np


BOSTON_DATA = load_boston()
X = BOSTON_DATA.data
y = BOSTON_DATA.target

imp = simp(missing_values = np.nan, strategy= "mean")
imp_data = imp.fit(X)
X = imp_data.transform(X)


print(X)
