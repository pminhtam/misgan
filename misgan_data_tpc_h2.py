# -*- coding: utf-8 -*-
import numpy as np

import pandas
import argparse
import random
import cf
from model import *
from data_generate import GenerateData
from misgan_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device

"""## Structure dataset"""

data_file = "lineitem.tbl.8"

data = pandas.read_csv(data_file, delimiter="|", header=None)[[0, 1, 2, 4, 5]].head(cf.num_row)

data['z1'] = data[0] + data[1]
data['z2'] = data[2] + data[4]

data_ori = data.values.astype(np.float32)

data = GenerateData(data_ori)

data_gen,mask_gen,imputer,loss = train(data)

torch.save(data_gen.state_dict(), './model/data_gen_tpc_h2.pt')
torch.save(mask_gen.state_dict(), './model/mask_gen_tpc_h2.pt')
torch.save(imputer.state_dict(),  './model/imputer_tpc_h2.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig("./img/"+data_file + "2.jpg")
