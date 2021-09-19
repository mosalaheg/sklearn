import numpy as np
from sklearn.model_selection import RepeatedKFold
from random import randint
X = np.array([[randint(1, 100), randint(1, 100)] for i in range(20)])

random_state = 12883823
rkf = RepeatedKFold(n_splits=4, n_repeats=2, random_state=random_state)
i = 1
for train, test in rkf.split(X):
    print(i)
    print("=" * 20)
    print(X[train])
    print("+" * 20)
    print(X[test])
    i += 1