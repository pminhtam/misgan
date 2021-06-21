import math

# def nCr(n,r):
#     f = math.factorial
#     return f(n) / f(r) / f(n-r)
#
# # print(nCr(4,2))
# # n = 5
# for n in [5,10,15,20,25,30,35]:
#     less = 0
#     for i in range(1,n):
#         less+=nCr(n,i) + n-i
#
#     print(less)

import numpy as np
# x = [5,10,15,20,25,30]
# y = np.array([40,	1067,	32871,	1048764,	33554730,	1073742257])
x = [26,	24,	22,	20,	18,	16,	14,	12,	10,	8,	6,	4,	2,	1]
y = [15344,	12410,	7855,	4402,	1476,	680	,320	,145,	58,	24,	7	,2,	0	,0]
# y = np.log(y)
import matplotlib.pyplot as plt

# plt.plot(x,y)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# line, = ax.plot(x,y, color='blue', lw=2)
# ax.set_yscale('log')

from matplotlib import pyplot

pyplot.grid()

# pyplot.subplot(1,1,1)
pyplot.plot(x,y,'-*', color='black', lw=2,markersize = 10)
pyplot.yscale('log')
pyplot.ylabel("Number of meaningless FDs",fontsize=12)
pyplot.xlabel("Nunber of attributes",fontsize = 12)
# pyplot.show()
plt.savefig("col2fd_log_2.png")
# plt.savefig("col2fd_log_all.png")
