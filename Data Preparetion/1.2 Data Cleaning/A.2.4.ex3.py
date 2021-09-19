from sklearn.impute import SimpleImputer
import numpy as np


data = [[1,2,np.nan],
        [3,np.nan,1],
        [5,np.nan,0],
        [np.nan,4,6 ],
        [5,0,np.nan],
        [4,5,5]]

imputed_module = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputed_data = imputed_module.fit(data)
modified_data = imputed_data.transform(data)
print(modified_data)