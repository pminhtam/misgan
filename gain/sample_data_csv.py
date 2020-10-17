import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
Data = np.genfromtxt("data/tpc_ds_train_3col.csv", delimiter=",", filling_values=0)

df = pd.DataFrame(Data)
s_mu = [list(df.sample(100).mean()) for i in range(20000)]

# print(Data)
# print(df)
# print(s_mu)
ss = pd.DataFrame(s_mu)

ss.to_csv("data/tpc_ds_train_3col_sam.csv",header=False,index=False)
# df.sa