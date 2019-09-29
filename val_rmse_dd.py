import numpy as np
import pandas
import argparse
import random
import cf
from model_dd import *
from data_generate import GenerateDataVal
from misgan_dd_train import train
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
from sklearn.metrics import mean_squared_error


def cal_loss_MSER(netG_imp, data_loader):
    np.random.seed(0)
    torch.manual_seed(100)
    imputed_data_mask = []
    origin_data_mask = []
    loss = []
    for real_data, real_mask, origin_data, _ in data_loader:
        origin_data = origin_data.float().to(device)
        real_mask = real_mask.float().to(device)
        imputed_data = netG_imp(origin_data, real_mask)

        imputed_data = imputed_data.detach().cpu().numpy()
        origin_data = origin_data.detach().cpu().numpy()
        masks = real_mask.detach().cpu().numpy()

        imputed_data_mask.extend(imputed_data * (1 - masks))
        origin_data_mask.extend(origin_data * (1 - masks))

    return imputed_data_mask, origin_data_mask

"""## Structure dataset"""

data_file = "lineitem.tbl.8"

data = pandas.read_csv(data_file, delimiter="|", header=None)[[0, 1, 2, 4, 5]].head(cf.num_row)

data['z1'] = data[0] + data[1]
data['z2'] = data[2] + data[4]

data_ori = data.values.astype(np.float32)

netG_imp = torch.load('./model/Gim_h2.pt',map_location='cpu')
netD_imp = torch.load('./model/Dim_h2.pt',map_location='cpu')

data = GenerateDataVal(data_file,data_ori,[2,3,4,5,6])
imputed_data_mask,origin_data_mask = cal_loss_MSER(netG_imp, DataLoader(data, batch_size=1, shuffle=False,drop_last=True))
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))

data = GenerateDataVal(data_file,data_ori,[0,1,4,5,6])
imputed_data_mask,origin_data_mask = cal_loss_MSER(netG_imp, DataLoader(data, batch_size=1, shuffle=False,drop_last=True))
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))