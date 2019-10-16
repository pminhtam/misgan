import numpy as np 

data = np.random.uniform(0, 100, size=(10000, 6))
data[:,1] = 2*data[:,0]

np.savetxt("data/y=2x.csv", data, delimiter=",",fmt="%.3f")
