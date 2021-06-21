import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
Data = np.genfromtxt("./uce_train_plus.csv", delimiter=",", filling_values=0,dtype=float)

df = pd.DataFrame(Data)
n_rand = int(len(Data)/3)

# s_mu = [df.sample(1) for i in range(100000)]
s_mu = df.sample(n_rand)

# print(Data)
# print(df)
# print(s_mu)
ss = pd.DataFrame(s_mu)
# print(ss)
ss.to_csv("./uce_train_plus_uniform.csv",header=False,index=False)
# df.sa