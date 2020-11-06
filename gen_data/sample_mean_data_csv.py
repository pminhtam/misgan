import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
# Data = np.genfromtxt("data/tpc_ds_train_3col.csv", delimiter=",", filling_values=0,dtype=float)
Data = np.genfromtxt("../gen_data/data_gamma_1_3.csv", delimiter=",", filling_values=0,dtype=float)

df = pd.DataFrame(Data)
s_mu = [list(df.sample(100).mean()) for i in range(20000)]

# print(Data)
# print(df)
# print(s_mu)
ss = pd.DataFrame(s_mu)

s = ss[2].to_numpy()
# print(s)
shape = 2
scale =3
# print(x1)
import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50)
y = np.ones_like(bins)
plt.plot(bins, y, linewidth=2, color='r')
plt.show()

# print(ss)
# ss.to_csv("../gen_data/data_gamma_1_3_mean.csv",header=False,index=False)
# df.sa