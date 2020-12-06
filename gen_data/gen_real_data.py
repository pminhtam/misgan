import numpy as np
# import cf

data_file = "../data/uce_test.csv"
Data = np.genfromtxt(data_file, delimiter=",")
print(Data.shape)
n,_ = Data.shape
data2 = np.ones((n,3))
data2[:,0] = Data[:,0]
data2[:,1] = Data[:,1]
data2[:,2] = Data[:,0] * Data[:,1]
# Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

np.savetxt("uce_test_mul.csv", data2, delimiter=",",fmt='%.3f')
