import numpy as np

x0 = np.random.uniform(0,1,size=10000)
x1 = 2*x0


a = np.asarray([x0,x1],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data6_test.csv',a,delimiter=',',fmt='%.3f')