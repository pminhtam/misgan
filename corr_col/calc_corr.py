import numpy as np
# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr

if __name__=="__main__":
    data = np.genfromtxt("../gen_data/gauss.csv", delimiter=",", filling_values=0)
    print(data.shape)
    # corr, _ = pearsonr(data[0], data[1])
    # print('Pearsons correlation: %.3f' % corr)
    print(np.corrcoef(data.T))