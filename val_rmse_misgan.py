
import numpy as np

import pandas
import argparse
import random
import cf
from model import *
from data_generate import GenerateDataVal

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device

data_file = "fd-reduced-30.csv"

data = pandas.read_csv(data_file, delimiter=",", header=None, skiprows=1)[[3, 4, 5, 6, 7]].head(cf.num_row)
data['z1'] = data[3] + data[4]
data['z2'] = data[5] + data[6]

data = data.values.astype(np.float32)

data = GenerateDataVal(data_file, data,[0,1])

batch_size = 1
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

nz = 128  # dimensionality of the latent code
n_critic = 5
alpha = .2
output_dim = data.n_labels
# batch_size = 1024

data_gen = FCDataGenerator(nz, output_dim).to(device)
mask_gen = FCMaskGenerator(nz, output_dim).to(device)



imputer = Imputer(output_dim).to(device)
impu_critic = FCCritic(output_dim).to(device)
impu_noise = torch.empty(batch_size, output_dim, device=device)




data_gen.load_state_dict(torch.load('./data_gen_tpc_ds2.pt'))
mask_gen.load_state_dict(torch.load('./mask_gen_tpc_ds2.pt'))
imputer.load_state_dict(torch.load('./imputer_tpc_ds2.pt'))

imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, data_loader,batch_size,output_dim)
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))

