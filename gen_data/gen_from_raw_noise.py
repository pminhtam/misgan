import numpy as np


data_ori = ['data_gamma_1_3','data_gamma_2_3','uniform','gauss']
# data_ori = ['gauss']
# aa = [0,1,2,3,4,5]
# 01 02 03 05 12 14 15 24 23 45
# "../data_new/"+data_name +"_mul_3col_part"+seq+ ".csv"
aa = ['01','02','03','05','12','14','15','24','23','45']
for num_noise in [4,5,6]:

    for data_name in data_ori:
        for seq in aa:
            data_file = "../data_new/" + data_name + "_plus_3col_part" + seq + ".csv"
            Data = np.genfromtxt(data_file, delimiter=",")
            print(data_file, Data.shape)
            # n, _ = Data1.shape
            n = 10000
            data2 = np.ones((n, num_noise))

            data2[:, 0] = Data[:, 0][:n]
            data2[:, 1] = Data[:, 1][:n]
            data2[:, 2] = Data[:, 2][:n]
            if 'uniform' in data_file:
                for ii in range(3,num_noise):
                    data2[:,ii] = np.random.uniform(0, 1,size=n)
            elif 'gauss' in data_file:
                for ii in range(3,num_noise):
                    data2[:, 3] = np.random.normal(0.5, 0.5/3 , n)
            elif 'data_gamma_1_3' in data_file:
                for ii in range(3,num_noise):
                    data2[:, 3] = np.random.gamma(1,3,size=n)
            elif 'data_gamma_2_3' in data_file:
                for ii in range(3,num_noise):
                    data2[:, 3] = np.random.gamma(2,3,size=n)
            np.savetxt("../data_new/"+data_name +"_plus_3col_part"+seq+ "_"+str(num_noise)+"noise.csv", data2, delimiter=",", fmt='%.3f')

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

