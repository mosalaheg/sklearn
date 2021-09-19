from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]


#transformer = Normalizer(norm='l1' ).fit_transform(X)

#transformer = Normalizer(norm='l2' ).fit_transform(X)

transformer = Normalizer(norm='max' ).fit_transform(X)

print(transformer)
