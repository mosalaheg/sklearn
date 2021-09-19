"""
    Mean Error --> Cost Function
    3 Types of errors
        1) Median Absolute Error
            from sklearn.metrices impor median_absolute_error
            maer_value = median_absolute_error(y_true, t_prediction)
                args :
                    1- y_true: The true values of y
                    2- y_prediction: The predicted values of y
                    
"""


from sklearn.metrics import median_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

maer_value = median_absolute_error(y_true, y_pred)

print(maer_value)
