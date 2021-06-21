import numpy as np
# import cf

data_file = "../data/tpc_h_test.csv"
Data = np.genfromtxt(data_file, delimiter=",")
print(Data.shape)
n,_ = Data.shape
data2 = np.ones((n,3))
data2[:,0] = Data[:,0]
data2[:,1] = Data[:,4]
data2[:,2] = Data[:,0] + Data[:,4]
# Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

np.savetxt("tpc_h_test_plus.csv", data2, delimiter=",",fmt='%.3f')
