import numpy as np
# import cf

data_file = "fd_test.csv"
Data = np.genfromtxt(data_file, delimiter=",")
# Data = np.genfromtxt("data_ori/" +data_file, delimiter=",",skip_header=1,usecols = (5, 6, 7, 14, 22), filling_values=0)[8192:,:]
# print(np.array([Data[:,0]+Data[:,1]]).T)

# Data = np.append(Data,np.array([Data[:,0]+Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]+Data[:,3]]).T,1)
# Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

np.savetxt("fd_test.csv", Data, delimiter=",",fmt='%.3f')
