import csv
# Import Libraries
import pandas as pd
import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # Generate Population Data
# # df = pd.DataFrame(np.random.randn(10000,1)) #normal
# Data = np.genfromtxt("../gen_data/uniform_6col.csv", delimiter=",", filling_values=0,dtype=float)
# # print(Data)
# df = pd.DataFrame(Data)
# # s_mu = [list(df.sample(100).median()) for i in range(20000)]
#
# sam_med = lambda fd_sam: list(fd_sam[fd_sam[2]==fd_sam[2].quantile(interpolation='nearest')].mean())
# s_mu = [sam_med(df.sample(100)) for i in range(20000)]
# # s_mu = df[df[2]==df[2].quantile(interpolation='nearest')]
#
# # print(Data)
# # print(df)
# # print(s_mu)
# ss = pd.DataFrame(s_mu)
# # print(ss)
# ss.to_csv("../gen_data/uniform_6col_med.csv",header=False,index=False)
# # df.sa

data_ori = ['data_gamma_1_3','data_gamma_2_3','uniform','gauss']
# data_ori = ['gauss']
# aa = [0,1,2,3,4,5]
# 01 02 03 05 12 14 15 24 23 45
sam_med = lambda fd_sam: list(fd_sam[fd_sam[2] == fd_sam[2].quantile(interpolation='nearest')].mean())

aa = ['01','02','03','05','12','14','15','24','23','45']
for i_data in [1000,3000,5000,8000]:
    for data_name in data_ori:

        for seq in aa:
            data_file = "../data_new/" + data_name + "_plus_3col_part" + seq + ".csv"
            Data = np.genfromtxt(data_file, delimiter=",")
            print(data_file, Data.shape)
            df = pd.DataFrame(Data)
            # s_mu = df.sample(5000)
            s_mu = [sam_med(df.sample(100)) for i in range(i_data)]
            ss = pd.DataFrame(s_mu)
            ss.to_csv("../data_new/"+data_name +"_plus_3col_part"+seq+ "_rand_medium"+str(int(i_data/100))+".csv",header=False,index=False)
            # np.savetxt("../data_new/"+data_name +"_mul_3col_part"+seq+ "_rand_uniform.csv", data2, delimiter=",", fmt='%.3f')
