"""
    Mean Error --> Cost Function
    3 Types of errors
        1) Mean Absolute Error
            from sklearn.metrices import mean_absolute_error
            maer_value = mean_absolute_error(y_true, t_prediction, 
                                             multioutput = 'uniform_average')
                args :
                    1- y_true: The true values of y
                    2- y_prediction: The predicted values of y
                    3- multioutput:
                                - 'uniform_average' : Default Value
                                - 'raw_values'
"""


# Import important libraries
from sklearn.metrics import mean_absolute_error


# Calculating Mean Absolute Error
maer_value = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')