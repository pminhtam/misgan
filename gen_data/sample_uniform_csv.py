import csv
# Import Libraries
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# Generate Population Data
# df = pd.DataFrame(np.random.randn(10000,1)) #normal
# Data = np.genfromtxt("./uniform_6col.csv", delimiter=",", filling_values=0,dtype=float)
#
# df = pd.DataFrame(Data)
# # s_mu = [df.sample(1) for i in range(100000)]
# s_mu = df.sample(20000)
#
# # print(Data)
# # print(df)
# # print(s_mu)
# ss = pd.DataFrame(s_mu)
# # print(ss)
# ss.to_csv("./uniform_6col_uniform.csv",header=False,index=False)
# # df.sa



data_ori = ['data_gamma_1_3','data_gamma_2_3','uniform','gauss']
# data_ori = ['gauss']
# aa = [0,1,2,3,4,5]
# 01 02 03 05 12 14 15 24 23 45

aa = ['01','02','03','05','12','14','15','24','23','45']
for i_data in [1000,3000,5000,8000]:
    for data_name in data_ori:

        for seq in aa:
            data_file = "../data_new/" + data_name + "_plus_3col_part" + seq + ".csv"
            Data = np.genfromtxt(data_file, delimiter=",")
            print(data_file, Data.shape)
            df = pd.DataFrame(Data)
            s_mu = df.sample(i_data)
            ss = pd.DataFrame(s_mu)
            ss.to_csv("../data_new/"+data_name +"_plus_3col_part"+seq+ "_rand_uniform"+str(int(i_data/100))+".csv",header=False,index=False)
            # np.savetxt("../data_new/"+data_name +"_mul_3col_part"+seq+ "_rand_uniform.csv", data2, delimiter=",", fmt='%.3f')
