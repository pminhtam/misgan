
# https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.random.html
import numpy as np

mu, sigma = 0, 10
x1 = np.random.normal(mu, sigma , 10000)
x2 = np.random.normal(mu, sigma ,size=10000)
x3 = x1+x2
a = np.asarray([x1,x2,x3],dtype=np.float32)
a = a.T
print(a[0])
np.savetxt('gauss_test.csv',a,delimiter=',',fmt='%.3f')

s = x1
print(x1)
import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50)
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()