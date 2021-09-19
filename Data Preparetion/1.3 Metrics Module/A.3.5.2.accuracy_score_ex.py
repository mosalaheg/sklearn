from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,5,3]
y_true = [0, 1, 2, 3,5,3]
print(accuracy_score(y_true, y_pred))                   # devide True values over all
print(accuracy_score(y_true, y_pred, normalize=False))  # number of all Trues
