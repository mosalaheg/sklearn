from sklearn.impute import SimpleImputer as sip
data = [[1,2,0],
        [3,0,1],
        [5,0,0],
        [0,4,6],
        [5,0,0],
        [4,5,5]]

ipmod = sip(missing_values = 0, strategy='median')
ipdata = ipmod.fit(data)
modified_data = ipdata.transform(ipdata)
print(modified_data)