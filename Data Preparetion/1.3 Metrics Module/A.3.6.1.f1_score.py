# Import important libraries
from sklearn.metrics import f1_score
#----------------------------------------------------

# Precision = TP / (TP + FP), Recall = TP / (TP + FN)
# Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)

# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

f1_sc = f1_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
# print('F1 Score is : ', F1Score)