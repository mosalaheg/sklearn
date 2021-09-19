# Import important libraries
from sklearn.preprocessing import StandardScaler

# Standard Scaler for Data

scaler_data = StandardScaler(copy= True, with_mean=True, with_std=True).fit_transform(X)

# scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# X = scaler.fit_transform(X)

# Showing data
# print('X \n' , X[:10])
# print('y \n' , y[:10])