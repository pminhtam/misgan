import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
Data = np.genfromtxt("../gen_data/data_gamma_2_3.csv", delimiter=",", filling_values=0)
# print(Data)
df = pd.DataFrame(Data)
s_mu = [list(df.sample(100).median()) for i in range(20000)]

# print(Data)
# print(df)
# print(s_mu)
ss = pd.DataFrame(s_mu)
# print(ss)
ss.to_csv("../gen_data/data_gamma_2_3_med.csv",header=False,index=False)
# df.sa