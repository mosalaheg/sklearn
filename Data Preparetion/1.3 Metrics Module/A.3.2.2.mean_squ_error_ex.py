from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

maer_value = mean_squared_error(y_true, y_pred, multioutput="uniform_average" )

print(maer_value)


y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

maer_value = mean_squared_error(y_true, y_pred, multioutput="uniform_average" )

print(maer_value)

maer_value = mean_squared_error(y_true, y_pred)

print(maer_value)



maer_value = mean_squared_error(y_true, y_pred, multioutput="raw_values" )

print(maer_value)

