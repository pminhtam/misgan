# -*- coding: utf-8 -*-

import pandas
import cf
from misgan.model import *
from misgan.data_generate import GenerateData
from misgan.misgan_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device

"""## Structure dataset"""

data_file = "uce-results-by-school-2011-2015.csv"

data_ori = pandas.read_csv("data/" +data_file, delimiter=",", header=None, skiprows=1)[[3, 6, 7, 15, 22]].head(
    cf.num_row).replace(np.nan, 0).values.astype(np.float32)

data = GenerateData(data_ori)

data_gen,mask_gen,imputer,loss = train(data)

torch.save(data_gen.state_dict(), './model/data_gen_uce.pt')
torch.save(mask_gen.state_dict(), './model/mask_gen_uce.pt')
torch.save(imputer.state_dict(),  './model/imputer_uce.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig("./img/"+data_file + ".jpg")
