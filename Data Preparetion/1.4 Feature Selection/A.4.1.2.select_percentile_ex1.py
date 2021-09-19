from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2, f_classif

X, y = load_digits(return_X_y=True)

print(X.shape)


X_new = SelectPercentile(score_func =chi2, percentile=10).fit_transform(X, y)

print(X_new.shape)
