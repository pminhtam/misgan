# -*- coding: utf-8 -*-
import numpy as np

import pandas
import argparse
import random
from model import *
from data_generate import GenerateData
from misgan_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device
import cf
"""## Structure dataset"""


data_file = "store_returns.dat"
data_ori = pandas.read_csv(data_file, delimiter="|", header=None)[[0,1,2,3,4,5]].head(cf.num_row).replace(np.nan, 0).values.astype(np.float32)

data = GenerateData(data_file,data_ori)

data_gen,mask_gen,imputer,loss = train(data)

torch.save(data_gen.state_dict(), './model/data_gen_tpc_ds.pt')
torch.save(mask_gen.state_dict(), './model/mask_gen_tpc_ds.pt')
torch.save(imputer.state_dict(),  './model/imputer_tpc_ds.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig("./img/"+data_file + ".jpg")
