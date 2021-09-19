import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(150).reshape((15, 10)), range(15)

print(X)
print("=" * 50)
print(list(y))
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, shuffle= False)

print(X_train)
print("=" * 50)
print(y_train)
print("=" * 50)
print(X_test)
print("=" * 50)
print(y_test)
print("=" * 50)

print(train_test_split(y, shuffle=True))
