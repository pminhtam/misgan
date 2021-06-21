import numpy as np
# import cf
data_name = "fd"
for t in ['train','test']:
    data_file = "../data/"+data_name+"_"+t+".csv"
    Data = np.genfromtxt(data_file, delimiter=",")
    print(Data.shape)
    n,_ = Data.shape
    data2 = np.ones((n,4))
    data2[:,0] = Data[:,0]
    data2[:,1] = Data[:,1]
    data2[:,2] = Data[:,2]

    data2[:, 0] = (data2[:,0]-np.min(data2[:,0]))/(np.max(data2[:,0])-np.min(data2[:,0]))
    data2[:, 1] = (data2[:,1]-np.min(data2[:,1]))/(np.max(data2[:,1])-np.min(data2[:,1]))
    data2[:, 2] = (data2[:,2]-np.min(data2[:,2]))/(np.max(data2[:,2])-np.min(data2[:,2]))
    data2[:,3] = data2[:,0] * data2[:,1] * data2[:,2]
    np.savetxt(data_name+"_"+t+"_mul3.csv", data2, delimiter=",",fmt='%.3f')


data_name = "tpc_ds"
for t in ['train','test']:
    data_file = "../data/"+data_name+"_"+t+".csv"
    Data = np.genfromtxt(data_file, delimiter=",")
    print(Data.shape)
    n,_ = Data.shape
    data2 = np.ones((n,4))
    data2[:,0] = Data[:,0]
    data2[:,1] = Data[:,4]
    data2[:,2] = Data[:,3]

    data2[:, 0] = (data2[:,0]-np.min(data2[:,0]))/(np.max(data2[:,0])-np.min(data2[:,0]))
    data2[:, 1] = (data2[:,1]-np.min(data2[:,1]))/(np.max(data2[:,1])-np.min(data2[:,1]))
    data2[:, 2] = (data2[:,2]-np.min(data2[:,2]))/(np.max(data2[:,2])-np.min(data2[:,2]))
    data2[:,3] = data2[:,0] * data2[:,1] * data2[:,2]

    np.savetxt(data_name+"_"+t+"_mul3.csv", data2, delimiter=",",fmt='%.3f')

data_name = "tpc_h"
for t in ['train','test']:
    data_file = "../data/"+data_name+"_"+t+".csv"
    Data = np.genfromtxt(data_file, delimiter=",")
    print(Data.shape)
    n,_ = Data.shape
    data2 = np.ones((n,4))
    data2[:,0] = Data[:,0]
    data2[:,1] = Data[:,1]
    data2[:,2] = Data[:,4]

    data2[:, 0] = (data2[:,0]-np.min(data2[:,0]))/(np.max(data2[:,0])-np.min(data2[:,0]))
    data2[:, 1] = (data2[:,1]-np.min(data2[:,1]))/(np.max(data2[:,1])-np.min(data2[:,1]))
    data2[:, 2] = (data2[:,2]-np.min(data2[:,2]))/(np.max(data2[:,2])-np.min(data2[:,2]))
    data2[:,3] = data2[:,0] * data2[:,1] * data2[:,2]

    np.savetxt(data_name+"_"+t+"_mul3.csv", data2, delimiter=",",fmt='%.3f')

data_name = "uce"
for t in ['train','test']:
    data_file = "../data/"+data_name+"_"+t+".csv"
    Data = np.genfromtxt(data_file, delimiter=",")
    print(Data.shape)
    n,_ = Data.shape
    data2 = np.ones((n,4))
    data2[:,0] = Data[:,0]
    data2[:,1] = Data[:,1]
    data2[:,2] = Data[:,2]

    data2[:, 0] = (data2[:,0]-np.min(data2[:,0]))/(np.max(data2[:,0])-np.min(data2[:,0]))
    data2[:, 1] = (data2[:,1]-np.min(data2[:,1]))/(np.max(data2[:,1])-np.min(data2[:,1]))
    data2[:, 2] = (data2[:,2]-np.min(data2[:,2]))/(np.max(data2[:,2])-np.min(data2[:,2]))
    data2[:,3] = data2[:,0] * data2[:,1] * data2[:,2]

    np.savetxt(data_name+"_"+t+"_mul3.csv", data2, delimiter=",",fmt='%.3f')
