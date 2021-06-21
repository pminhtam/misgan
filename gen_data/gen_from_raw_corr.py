import numpy as np


data_ori = ['data_gamma_1_3','data_gamma_2_3','uniform','gauss']
# data_ori = ['gauss']
# aa = [0,1,2,3,4,5]
# 01 02 03 05 12 14 15 24 23 45
# "../data_new/"+data_name +"_mul_3col_part"+seq+ ".csv"
aa = ['01','02','03','05','12','14','15','24','23','45']
for data_name1 in data_ori:
    for data_name2 in data_ori:
        if data_name1 == data_name2:
            continue
        data_file1 = data_name1 + "_raw_col.csv"
        data_file2 = data_name2 + "_raw_col.csv"
        Data1 = np.genfromtxt(data_file1, delimiter=",")
        Data2 = np.genfromtxt(data_file2, delimiter=",")
        print(data_file1, Data1.shape)
        for seq in aa:

            # n, _ = Data1.shape
            n = 10000
            data2 = np.ones((n, 3))

            data2[:, 0] = Data1[:, int(seq[0])][:n]
            data2[:, 1] = Data2[:, int(seq[1])][:n]
            data2[:, 2] = Data1[:, int(seq[0])][:n]+Data2[:, int(seq[1])][:n]

            np.savetxt("../data_new/"+data_name1+ "_" + data_name2 +"_plus_3col_part"+seq+ "_corr.csv", data2, delimiter=",", fmt='%.3f')

# 012 023 034 053 123 140 152 241 235 452
# aa = ['012','023','034','053','123','140','152','241','235','452']
# for data_name in data_ori:
#     data_file = data_name + "_raw_col.csv"
#     Data = np.genfromtxt(data_file, delimiter=",")
#     print(data_file, Data.shape)
#     for seq in aa:
#
#         n, _ = Data.shape
#         data2 = np.ones((n, 4))
#
#         data2[:, 0] = Data[:, int(seq[0])]
#         data2[:, 1] = Data[:, int(seq[1])]
#         data2[:, 2] = Data[:, int(seq[2])]
#         data2[:, 3] = Data[:, int(seq[0])]*Data[:, int(seq[1])]*Data[:, int(seq[2])]
#
#         np.savetxt("../data_new/"+data_name +"_mul3_3col_part"+seq+ ".csv", data2, delimiter=",", fmt='%.3f')

