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
    data_ori = [ 'uniform']
    name_data_real = ['01', '02', '03', '05', '12', '14', '15', '24', '23', '45']
    for t in type:
        print(t,"+===================================================")
        for data_name1 in data_ori:
            print(data_name1)
            # pos = []
            # neg = []
            y_pred = []
            for seq in name_data_real:

                data = np.genfromtxt("../data_new/"+data_name1 +"_"+ t +"_3col_part"+seq+ "_category.csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../data_new/"+data_name +"_fake_4col_part"+seq+ ".csv", delimiter=",", filling_values=0)
                # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
                # print(data.shape)

                # corr, _ = pearsonr(data[0], data[1])
                # print('Pearsons correlation: %.3f' % corr)
                loss = 1- np.corrcoef(data.T)
                print(loss)
                # print((loss[0][3]+ loss[1][3]+loss[2][3])/3)
                y_pred.append((loss[0][2]+ loss[1][2])/2)
            print("f1 : ", y_pred)
