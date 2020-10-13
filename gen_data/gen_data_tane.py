import numpy as np

x0 = np.random.normal(5,10,size=30)
x1 = np.random.normal(5,20,size=30)
x2 = np.random.normal(10,10,size=30)
x3 = np.random.normal(10,5,size=30)


a = np.asarray([x0,x1,x2,x3],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data_tane.csv',a,delimiter=',',fmt='%.3f')