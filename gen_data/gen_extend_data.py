import numpy as np
# import cf

# data_file = "uniform_6col_test.csv"
# data_file2 = "gauss_6col_test.csv"
# data_file3 = "data_gamma_2_3_6col_test.csv"

data_file = "tpc_h_train_plus_med.csv"
data_file2 = "fd_train_plus.csv"
data_file3 = "tpc_ds_train_plus.csv"
Data = np.genfromtxt(data_file, delimiter=",")
Data2 = np.genfromtxt(data_file2, delimiter=",")
Data3 = np.genfromtxt(data_file3, delimiter=",")
print(Data.shape)
print(Data2.shape)
n,_ = Data.shape
data2 = np.ones((n,6))
data2[:,0] = Data[:,0]
data2[:,1] = Data[:,1]
data2[:,2] = Data[:,2]
data2[:,3] = Data2[:n,0]
data2[:,4] = Data2[:n,1]
data2[:,5] = Data3[:n,1]
# data2[:,4] = Data[:,4]
# data2[:,5] = Data[:,5]
# data2[:,6] = Data2[:,0]
# data2[:,7] = Data2[:,1]
# data2[:,8] = Data3[:,0]

# Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

np.savetxt("tpc_h_train_plus_med_6col.csv", data2, delimiter=",",fmt='%.3f')
