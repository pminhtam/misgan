import numpy as np


# data_ori = ['data_gamma_1_3','data_gamma_2_3','uniform','gauss']
data_ori = ['uniform']
# aa = [0,1,2,3,4,5]
# 01 02 03 05 12 14 15 24 23 45

aa = ['01','02','03','05','12','14','15','24','23','45']
for data_name in data_ori:
    # data_file = data_name + "_raw_col.csv"
    # Data = np.genfromtxt(data_file, delimiter=",")
    # print(data_file, Data.shape)
    for seq in aa:

        # n, _ = Data.shape
        data2 = np.ones((10000, 3))

        data2[:, 0] = np.random.uniform(0, 0.5,size=10000)
        data2[:, 1] = np.random.uniform(0, 0.5,size=10000)
        data2[:, 2] = np.round(data2[:, 0]+data2[:, 1])
        # med = np.median(temp)
        # data2[:, 2] = temp>med
        # print(data2)
        # exit(0)
        np.savetxt("../data_new/"+data_name +"_plus_3col_part"+seq+ "_binary.csv", data2, delimiter=",", fmt='%.3f')

