
# https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.random.html
import numpy as np

shape, scale = 1, 3
x1 = np.random.gamma(shape, scale, 10000)
x2 = np.random.gamma(shape,scale,size=10000)
x3 = np.random.gamma(shape,scale,size=10000)
# x3 = x1+x2
x4 = np.random.gamma(shape, scale, 10000)
x5 = np.random.gamma(shape,scale,size=10000)
x6 = np.random.gamma(shape,scale,size=10000)
# x6 = x4*x5
a = np.asarray([x1,x2,x3,x4,x5,x6],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('data_gamma_1_3_raw_col.csv',a,delimiter=',',fmt='%.5f')

# s = x1
# print(x1)
# import matplotlib.pyplot as plt
# import scipy.special as sps
# count, bins, ignored = plt.hist(s, 50)
# y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
# plt.plot(bins, y, linewidth=2, color='r')
# plt.show()