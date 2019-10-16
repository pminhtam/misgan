import numpy as np
# import cf

data_file = "uce-results-by-school-2011-2015.csv"
Data = np.genfromtxt("data_ori/" +data_file, delimiter=",",skip_header=1,usecols = (5, 6, 7, 14, 22), filling_values=0)[8192:,:]
# print(np.array([Data[:,0]+Data[:,1]]).T)

Data = np.append(Data,np.array([Data[:,0]+Data[:,1]]).T,1)
Data = np.append(Data,np.array([Data[:,2]+Data[:,3]]).T,1)
# Data = np.append(Data,np.array([Data[:,0]*Data[:,1]]).T,1)
# Data = np.append(Data,np.array([Data[:,2]*Data[:,3]]).T,1)

np.savetxt("tpc_uce_test.csv", Data, delimiter=",")
