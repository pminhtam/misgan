import pandas
import cf
from misgan_dd.model_dd import *
from misgan.data_generate import GenerateDataVal

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


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

data_file = "data/uce-results-by-school-2011-2015.csv"
data = pandas.read_csv(data_file, delimiter=",", header=None, skiprows=1)[[3, 6, 7, 15, 22]].head(cf.num_row).replace(
    np.nan, 0)
data['z1'] = data[3] + data[6]
data['z2'] = data[7] + data[15]
data_ori = data.values.astype(np.float32)
netG_imp = torch.load('./model/Gim_uce2.pt',map_location='cpu')
netD_imp = torch.load('./model/Dim_uce2.pt',map_location='cpu')

data = GenerateDataVal(data_file,data_ori,[2,3,4,5,6])
imputed_data_mask,origin_data_mask = cal_loss_MSER(netG_imp, DataLoader(data, batch_size=1, shuffle=False,drop_last=True))
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))
# print(imputed_data_mask[0])
# print(origin_data_mask[0])
data = GenerateDataVal(data_file,data_ori,[0,1,4,5,6])
imputed_data_mask,origin_data_mask = cal_loss_MSER(netG_imp, DataLoader(data, batch_size=1, shuffle=False,drop_last=True))
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))

# print(imputed_data_mask[0])
# print(origin_data_mask[0])