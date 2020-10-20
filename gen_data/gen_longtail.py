
# https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.random.html
import numpy as np

shape, scale = 5., 5
s = np.random.gamma(shape, scale, 100000)
print(s)
import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()