import numpy as np 

data = np.random.uniform(0, 100, size=(10000, 3))
data[:,1] = 2*data[:,0]
# data[:, 2] = np.random.normal(size=(10000,))

np.savetxt("data/y=2x.csv", data, delimiter=",",fmt="%.3f")
