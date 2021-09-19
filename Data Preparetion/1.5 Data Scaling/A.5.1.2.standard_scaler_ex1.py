from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
scaler.fit(data)
print(scaler.mean_)
print("=" * 50)
newdata = scaler.transform(data)
print(newdata)
print("=" * 50)
newdata = scaler.fit_transform(data) 
print(newdata)


