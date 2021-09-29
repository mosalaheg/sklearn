# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:41:49 2021

@author: Mohamed Salah
"""

from sklearn.feature_selection import chi2, f_classif, SelectPercentile, GenericUnivariateSelect, SelectKBest, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X, y = dataset.data, dataset.target


select_percentile = SelectPercentile(score_func = chi2, percentile = 20).fit_transform(X, y)
print(select_percentile)

print("=" * 20)

generic_select = GenericUnivariateSelect(score_func = f_classif, param = 20).fit_transform(X, y)
print(generic_select)

print("=" * 20)

select_kbest = SelectKBest(score_func = f_classif, k = 20).fit_transform(X, y)
print(select_kbest)

print("=" * 20)

linear = LinearRegression()
select_from_model = SelectFromModel(estimator = linear).fit_transform(X, y)
