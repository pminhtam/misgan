import numpy as np

x0 = np.random.uniform(0,1,size=10000)
x1 = 2*x0
# x1 = np.random.normal(5,3,size=10000)
# x2 = x1+x0
#
# x3 = np.random.uniform(0,2,size=10000)
# x4 = np.random.normal(10,5,size=10000)
# x5 = x4*x3
#
# x6 = np.random.uniform(0,5,size=10000)
# x7 = np.random.uniform(0,5,size=10000)
# x8 = np.abs(x7-x6)


a = np.asarray([x0,x1],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data6_test.csv',a,delimiter=',',fmt='%.3f')