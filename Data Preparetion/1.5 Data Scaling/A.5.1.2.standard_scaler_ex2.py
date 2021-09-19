from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
print(X[0])

# scaler_data = StandardScaler(copy= True, with_mean=True, with_std=True).fit_transform(X)

print("=" * 50)
X = StandardScaler().fit_transform(X)
print(X[0])
