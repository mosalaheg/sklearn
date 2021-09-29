# Import important libraries
from sklearn.datasets import load_breast_cancer

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif

# Load data
breast_data = load_breast_cancer()

# X data
X = breast_data.data

# y data
y = breast_data.target

# Print X and y
# print(X[:10])
# print(X.shape)
print(breast_data.feature_names)

# print(y[:10])
# print(y.shape)
print(breast_data.target_names)


#Feature Selection by Percentile
print("Original Shape", X.shape)
feature_selections = SelectPercentile(score_func= chi2, percentile=20)
X = feature_selections.fit_transform(X, y)
print(X.shape)

print('Selected Features are : ' , breast_data.feature_names[feature_selections.get_support()])

X = breast_data.data

print("Original Shape", X.shape)
feature_selections = SelectPercentile(score_func= f_classif, percentile=20)
X = feature_selections.fit_transform(X, y)
print(X.shape)

print('Selected Features are : ' , breast_data.feature_names[feature_selections.get_support()])
