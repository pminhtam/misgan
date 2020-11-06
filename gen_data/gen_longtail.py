
# https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.random.html
import numpy as np

shape, scale = 1., 3
# shape, scale = 1., 3
x1 = np.random.gamma(shape, scale, 100000)
x2 = np.random.gamma(shape,scale,size=100000)
x3 = x1+x2

x4 = np.random.gamma(shape, scale, 100000)
x5 = np.random.gamma(shape,scale,size=100000)
x6 = x4*x5

# Save file csv
# a = np.asarray([x1,x2,x3],dtype=np.float32)
# a = a.T
# print(a[0])
# np.savetxt('data_gamma_1_3_test.csv',a,delimiter=',',fmt='%.3f')


### PLot img
s = x1
# print(x1)
import matplotlib.pyplot as plt
import scipy.special as sps
fig = plt.figure(figsize = (20,5))
plt.subplot(1, 6, 1)

count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))

plt.title("x1")
plt.plot(bins, y, linewidth=2, color='r')

s = x2
plt.subplot(1, 6, 2)

count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.title("x2")
plt.plot(bins, y, linewidth=2, color='r')

s = x3
plt.subplot(1, 6, 3)
count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.title("x3=x1+x2")
plt.plot(bins, y, linewidth=1, color='r')

s = x4
plt.subplot(1, 6, 4)
count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.title("x4")
plt.plot(bins, y, linewidth=1, color='r')

s = x5
plt.subplot(1, 6, 5)
count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.title("x5")
plt.plot(bins, y, linewidth=1, color='r')

s = x6
plt.subplot(1, 6, 6)
count, bins, ignored = plt.hist(s, 50, density=False)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.title("x6=x4*x5")
plt.plot(bins, y, linewidth=1, color='r')

plt.savefig("gamma_1_3_6col.jpg")
plt.show()