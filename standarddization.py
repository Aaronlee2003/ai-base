import numpy as np
from sklearn.preprocessing import MinMaxScaler ,StandardScaler

data = np.array([[10, 2], 
                 [20, 4], 
                 [30, 6], 
                 [40, 8]])

min_max_scaler=MinMaxScaler()
data_minmax=min_max_scaler.fit_transform(data)

std_scaler=StandardScaler()
data_std=std_scaler.fit_transform(data)

