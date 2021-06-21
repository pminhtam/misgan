import numpy as np
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
from sklearn.metrics import accuracy_score,f1_score

# thresh = 0.3
# thresh = 0.296
# thresh = {'plus':0.296,'mul':0.37,'mul3':0.615}
if __name__=="__main__":
    type = ['plus']
    # data = np.genfromtxt("../gen_data/data_gamma_1_3_6col.csv", delimiter=",", filling_values=0)
    # data = np.genfromtxt("../gen_data/uce_train_mul.csv", delimiter=",", filling_values=0)
    # data_ori = ['data_gamma_1_3', 'data_gamma_2_3', 'uniform', 'gauss']
    data_ori = ['uniform']
    name_data_real = ['01', '02', '03', '05', '12', '14', '15', '24', '23', '45']
    name_data_fake = ['012', '023', '034', '053', '123', '140', '152', '241', '235', '452']
    # name_data_fake = ['0153', '0234', '0341', '0532', '1235', '1402', '1524', '2413', '2354', '4530']
    for t in type:
        print(t,"+===================================================")
        for data_name in data_ori:
            print(data_name)
            # pos = []
            # neg = []
            y_pred = []
            y_true = []
            for seq in name_data_real:

                data = np.genfromtxt("../data_new/"+data_name +"_"+ t +"_3col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../data_new/"+data_name +"_fake_4col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                loss = 1- np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                # y_pred.extend((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][2]+ loss[1][2])/2)
                y_true.append(0)
            thresh = np.max(y_pred)
            for seq in name_data_fake:

                # data = np.genfromtxt("../data_new/"+data_name +"_plus_3col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                data = np.genfromtxt("../data_new/"+data_name +"_fake_3col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                loss = 1- np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][2]+ loss[1][2])/2)
                y_true.append(1)
            for seq in name_data_real:

                data = np.genfromtxt("../data_new/"+data_name +"_"+t+"_3col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../data_new/"+data_name +"_fake_4col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                aa = data[0:100, 1]
                np.random.shuffle(aa)
                data[0:100, 1] = aa
                loss = 1- np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                # y_pred.extend((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][2]+ loss[1][2])/2)
                y_true.append(1)
            y_pred = np.array(y_pred)
            print(y_pred)
            # thresh = np.mean(y_pred)

            y_true = np.array(y_true)
            y_pred = y_pred > thresh
            print(y_pred)
            print("acc : ",accuracy_score(y_true, y_pred))
            print("f1 : ",f1_score(y_true, y_pred))
    name_data_real = ['012', '023', '034', '053', '123', '140', '152', '241', '235', '452']
    name_data_fake = ['0153', '0234', '0341', '0532', '1235', '1402', '1524', '2413', '2354', '4530']
    type = ['mul3']
    for t in type:
        print(t,"+===================================================")

        for data_name in data_ori:
            print(data_name)
            # pos = []
            # neg = []
            y_pred = []
            y_true = []
            for seq in name_data_real:
                data = np.genfromtxt("../data_new/" + data_name + "_" + t + "_3col_part" + seq + ".csv",
                                     delimiter=",", filling_values=0)
                # data = np.genfromtxt("../data_new/"+data_name +"_fake_4col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                loss = 1 - np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                # y_pred.extend((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_true.append(0)
            # thresh = np.max(y_pred)
            for seq in name_data_fake:
                # data = np.genfromtxt("../data_new/"+data_name +"_plus_3col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                data = np.genfromtxt("../data_new/" + data_name + "_fake_4col_part" + seq + ".csv", delimiter=",",
                                     filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                loss = 1 - np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_true.append(1)
            for seq in name_data_real:
                data = np.genfromtxt("../data_new/" + data_name + "_" + t + "_3col_part" + seq + ".csv",
                                     delimiter=",", filling_values=0)
                # data = np.genfromtxt("../data_new/"+data_name +"_fake_4col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                aa = data[0:100, 1]
                np.random.shuffle(aa)
                data[0:100, 1] = aa
                loss = 1 - np.corrcoef(data.T)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                # y_pred.append((loss[0][2] + loss[1][2]) / 2)
                y_true.append(1)
            y_pred = np.array(y_pred)
            thresh = np.mean(y_pred)
            print(y_pred)
            y_true = np.array(y_true)
            y_pred = y_pred > thresh
            print(y_pred)
            print("acc : ",accuracy_score(y_true, y_pred))
            print("f1 : ",f1_score(y_true, y_pred))
