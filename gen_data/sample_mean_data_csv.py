import csv
# Import Libraries
import pandas as pd
import numpy as np
# np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
#     formatter = dict( float = lambda x: "%.5g" % x ))
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # Generate Population Data
# # df = pd.DataFrame(np.random.randn(10000,1)) #normal
# # Data = np.genfromtxt("data/tpc_ds_train_3col.csv", delimiter=",", filling_values=0,dtype=float)
# Data = np.genfromtxt("../gen_data/data_gamma_1_3_6col.csv", delimiter=",", filling_values=0,dtype=float)
# print(np.mean((Data-np.min(Data,axis=0))/(np.max(Data,axis=0)-np.min(Data,axis=0)),axis=0))
# exit(0)
# df = pd.DataFrame(Data)
# s_mu = [list(df.sample(100).mean()) for i in range(20000)]
#
# # print(Data)
# # print(df)
# # print(s_mu)
# ss = pd.DataFrame(s_mu)
#
# # s = ss[2].to_numpy()
# # # print(s)
# # shape = 2
# # scale =3
# # # print(x1)
# # import matplotlib.pyplot as plt
# # import scipy.special as sps
# # count, bins, ignored = plt.hist(s, 50)
# # y = np.ones_like(bins)
# # plt.plot(bins, y, linewidth=2, color='r')
# # plt.show()
#
# # print(ss)
# ss.to_csv("../gen_data/uniform_6col_mean.csv",header=False,index=False)
# # ss.to_csv("../gen_data/data_gamma_1_3_mean.csv",header=False,index=False)
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
            # s_mu = df.sample(5000)
            s_mu = [list(df.sample(100).mean()) for i in range(i_data)]
            ss = pd.DataFrame(s_mu)
            ss.to_csv("../data_new/"+data_name +"_plus_3col_part"+seq+ "_rand_mean"+str(int(i_data/100))+".csv",header=False,index=False)
            # np.savetxt("../data_new/"+data_name +"_mul_3col_part"+seq+ "_rand_uniform.csv", data2, delimiter=",", fmt='%.3f')
