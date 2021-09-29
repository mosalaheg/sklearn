from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf1 = LogisticRegression(random_state=10, solver='lbfgs' , max_iter= 1000 , C = 0.5 , tol = 0.01)
clf2 = LogisticRegression(random_state=10, solver='liblinear')
clf3 = LogisticRegression(random_state=10, solver='saga')

clf1.fit(X, y)
clf1.predict(X[:2, :])
clf1.predict_proba(X[:2, :])


clf2.fit(X, y)
clf2.predict(X[:2, :])
clf2.predict_proba(X[:2, :])



clf3.fit(X, y)
clf3.predict(X[:2, :])
clf3.predict_proba(X[:2, :])


score1 = clf1.score(X, y)

print('score = ' , score1)
print('No of iterations = ' , clf1.n_iter_)
print('Classes = ' , clf1.classes_)

score2 = clf2.score(X, y)

print('score = ' , score2)
print('No of iterations = ' , clf2.n_iter_)
print('Classes = ' , clf2.classes_)

score3 = clf3.score(X, y)

print('score = ' , score3)
print('No of iterations = ' , clf3.n_iter_)
print('Classes = ' , clf3.classes_)

 
 