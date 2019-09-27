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

data_file = "fd-reduced-30.csv"

data_ori = pandas.read_csv(data_file, delimiter=",", header=None, skiprows=1)[[3, 4, 5, 6, 7]].head(
    cf.num_row).values.astype(np.float32)

data = GenerateData(data_file,data_ori)

data_gen,mask_gen,imputer,loss = train(data)

torch.save(data_gen.state_dict(), './model/data_gen_fd_reduced.pt')
torch.save(mask_gen.state_dict(), './model/mask_gen_fd_reduced.pt')
torch.save(imputer.state_dict(),  './model/imputer_fd_reduced.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig("./img/"+data_file + ".jpg")
