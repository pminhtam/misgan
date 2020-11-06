
# https://docs.scipy.org/doc/numpy-1.10.1/reference/routines.random.html
import numpy as np

min, max = 0, 10
x1 = np.random.uniform(min, max, 100000)
x2 = np.random.uniform(min, max  ,size=100000)
x3 = x1+x2

x4 = np.random.uniform(min, max, 100000)
x5 = np.random.uniform(min, max  ,size=100000)
x6 = x4*x5

# a = np.asarray([x1,x2,x3],dtype=np.float32)
# a = a.T
# print(a[0])
# np.savetxt('uniform_test.csv',a,delimiter=',',fmt='%.3f')


s = x1
print(x1)
import matplotlib.pyplot as plt
import scipy.special as sps
fig = plt.figure(figsize = (20,5))
plt.subplot(1, 6, 1)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)

plt.title("x1")
plt.plot(bins, y, linewidth=2, color='r')
# plt.show()

s = x2
# import matplotlib.pyplot as plt
# import scipy.special as sps
plt.subplot(1, 6, 2)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)
plt.title("x2")
plt.plot(bins, y, linewidth=2, color='r')
# plt.show()

s = x3
# import matplotlib.pyplot as plt
# import scipy.special as sps
plt.subplot(1, 6, 3)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)
plt.title("x3=x1+x2")
# print(bins)
# print(y)
plt.plot(bins, y, linewidth=1, color='r')


s = x4
# import matplotlib.pyplot as plt
# import scipy.special as sps
plt.subplot(1, 6, 4)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)
plt.title("x4")
# print(bins)
# print(y)
plt.plot(bins, y, linewidth=1, color='r')


s = x5
# import matplotlib.pyplot as plt
# import scipy.special as sps
plt.subplot(1, 6, 5)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)
plt.title("x5")
# print(bins)
# print(y)
plt.plot(bins, y, linewidth=1, color='r')


s = x6
# import matplotlib.pyplot as plt
# import scipy.special as sps
plt.subplot(1, 6, 6)

count, bins, ignored = plt.hist(s, 50, density=False)
y = np.ones_like(bins)
plt.title("x6=x4*x5")
# print(bins)
# print(y)
plt.plot(bins, y, linewidth=1, color='r')
plt.savefig("uniform_6col.jpg")


plt.show()