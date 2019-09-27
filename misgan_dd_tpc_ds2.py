import numpy as np
import pandas
import argparse
import random
import cf
from model_dd import *
from data_generate import GenerateData
from misgan_dd_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


"""## Structure dataset"""

data_file = "store_returns.dat"

data = pandas.read_csv(data_file, delimiter="|", header=None)[[0, 1, 2, 3, 4, 5]].head(cf.num_row).replace(np.nan, 0)
data['z1'] = data[0] + data[1]
data['z2'] = data[2] + data[3]
data = data.values.astype(np.float32)

data = GenerateData(data_file,data)

netG_imp,netD_imp = train(data)

torch.save(netG_imp,   './model/Gim_ds2.pt')
torch.save(netD_imp,  './model/Dim_ds2.pt')