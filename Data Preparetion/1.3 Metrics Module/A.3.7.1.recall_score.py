# Import important libraries
from sklearn.metrics import recall_score
#----------------------------------------------------
"""
    TRUE / PRED :         Success            Faild
    Success 820 :       700 TP (TS)        120 FP (FF) 
    Failed  180 :       50  FN (FS)        130 TN (TF)

    TP, and TN must be big as possible
    
    Recall Score: You get:
        
                    number of the True Positive / number of all Positive
"""


# Calculating Recall Score : (Sensitivity) = (TP / float(TP + FN))   1 / 1+2  
# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#print('Recall Score is : ', RecallScore)