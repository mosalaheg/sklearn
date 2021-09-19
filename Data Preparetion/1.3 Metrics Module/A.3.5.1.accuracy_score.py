# Import important libraries
from sklearn.metrics import accuracy_score


# Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
acc_score = accuracy_score(y_true, y_pred)
