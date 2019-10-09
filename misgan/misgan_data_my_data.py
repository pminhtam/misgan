# -*- coding: utf-8 -*-

import pandas
from misgan.model import *
from misgan.data_generate import GenerateData
from misgan.misgan_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


data_file = "data3.csv"

data_ori = pandas.read_csv("data/" +data_file, delimiter=",", header=None).values.astype(np.float32)

data = GenerateData(data_ori)

data_gen,mask_gen,imputer,loss = train(data)

torch.save(data_gen.state_dict(), './model/data_gen_my_data3.pt')
torch.save(mask_gen.state_dict(), './model/mask_gen_my_data3.pt')
torch.save(imputer.state_dict(),  './model/imputer_my_data3.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig("./img/"+data_file + ".jpg")
