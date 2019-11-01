import numpy as np

x1 = np.random.uniform(0,1,size=1000)
x2 = np.random.uniform(0,1,size=1000)



x3 = x1+x2
x4 = np.abs(x1-x2)
x5 = x1*x2
x6 = x1*1.0/(x2+1.0)
x7 = x1*2 + x2
x8 = x1+3*x2


a = np.asarray([x1,x2,x3,x4,x5,x6,x7,x8],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data3_test.csv',a,delimiter=',',fmt='%.3f')