import numpy as np
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

if __name__=="__main__":
    data = np.genfromtxt("../gen_data/tpc_h_train_mul.csv", delimiter=",", filling_values=0)
    print(data.shape)
    # corr, _ = pearsonr(data[0], data[1])
    # print('Pearsons correlation: %.3f' % corr)
    # np.set_printoptions(suppress=True)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    print(np.corrcoef(data.T))