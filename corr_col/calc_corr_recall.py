import numpy as np
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
pos = []
neg = []
if __name__=="__main__":
    # data = np.genfromtxt("../gen_data/data_gamma_1_3_6col.csv", delimiter=",", filling_values=0)
    # data = np.genfromtxt("../gen_data/uce_train_mul.csv", delimiter=",", filling_values=0)
    data = np.genfromtxt("../gen_data/tpc_ds_train_mul3.csv", delimiter=",", filling_values=0)
    # data = np.genfromtxt("../gen_data/tpc_h_train_mul3.csv", delimiter=",", filling_values=0)
    print(data.shape)
    print(1- np.corrcoef(data.T))
    # np.shuffle(data[:1])
    # data.sample(frac=1)
    np.random.shuffle(data)
    # corr, _ = pearsonr(data[0], data[1])
    # print('Pearsons correlation: %.3f' % corr)
    print(1- np.corrcoef(data.T))
    # print(np.random.shuffle(data[:,0]))
    aa = data[:,1]
    np.random.shuffle(aa)
    data[:,1] = aa
    # print(data)
    print(1- np.corrcoef(data.T))
