from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler


X, y = load_breast_cancer(return_X_y = True)
X = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(X)
print("=" * 50)
print(X[0])

