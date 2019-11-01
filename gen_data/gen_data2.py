import numpy as np

x0 = np.random.uniform(0,10,size=10000)
x1 = np.random.uniform(5,20,size=10000)
x2 = x1+x0

x3 = np.random.uniform(0,10,size=10000)
x4 = np.random.uniform(5,20,size=10000)
x5 = x4*x3


a = np.asarray([x0,x1,x2,x3,x4,x5],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data2_test.csv',a,delimiter=',',fmt='%.3f')