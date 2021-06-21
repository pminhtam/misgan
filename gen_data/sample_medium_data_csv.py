import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
Data = np.genfromtxt("../gen_data/tpc_h_train_mul.csv", delimiter=",", filling_values=0,dtype=float)
# print(Data)
df = pd.DataFrame(Data)
# s_mu = [list(df.sample(100).median()) for i in range(20000)]
n_rand = int(len(Data)/3)

sam_med = lambda fd_sam: list(fd_sam[fd_sam[2]==fd_sam[2].quantile(interpolation='nearest')].mean())
s_mu = [sam_med(df.sample(100)) for i in range(n_rand)]
# s_mu = df[df[2]==df[2].quantile(interpolation='nearest')]

# print(Data)
# print(df)
# print(s_mu)
ss = pd.DataFrame(s_mu)
# print(ss)
ss.to_csv("../gen_data/tpc_h_train_mul_med.csv",header=False,index=False)
# df.sa