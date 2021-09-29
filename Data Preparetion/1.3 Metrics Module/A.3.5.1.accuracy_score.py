# Import important libraries
from sklearn.metrics import accuracy_score

"""

    TRUE / PRED :         Success            Faild
    Success 820 :       700 TP (TS)        120 FP (FF) 
    Failed  180 :       50  FN (FS)        130 TN (TF)

    TP, and TN must be big as possible
    
    Accuracy score: You get:
        
                    number of the right prediction / number of all predicions

"""

# Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
acc_score = accuracy_score(y_true, y_pred)
