"""
    Confusion Matrix
        from sklearn.metrices import confusion_matrix
        conf_matrix = confusion_matrix(y_true, t_prediction)
        args :
            1- y_true: The true values of y
            2- y_prediction: The predicted values of y
            
    TRUE / PRED :         Success            Faild
    Success 820 :       700 TP (TS)        120 FP (FF) 
    Failed  180 :       50  FN (FS)        130 TN (TF)

    TP, and TN must be big as possible
    
"""


# Import important libraries
from sklearn.metrics import confusion_matrix

# These are for plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------


# Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()