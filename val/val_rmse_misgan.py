import pandas
import cf
from misgan.model import *
from misgan.data_generate import GenerateDataVal

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device

data_file = "data/fd-reduced-30.csv"

data = pandas.read_csv(data_file, delimiter=",", header=None, skiprows=1)[[3, 4, 5, 6, 7]].head(cf.num_row)
data['z1'] = data[3] + data[4]
data['z2'] = data[5] + data[6]

data_ori = data.values.astype(np.float32)


nz = 128  # dimensionality of the latent code
n_critic = 5
alpha = .2
output_dim = data.n_labels
# batch_size = 1024

data_gen = FCDataGenerator(nz, output_dim).to(device)
mask_gen = FCMaskGenerator(nz, output_dim).to(device)
imputer = Imputer(output_dim).to(device)

data_gen.load_state_dict(torch.load('./model/data_gen_tpc_ds2.pt'))
mask_gen.load_state_dict(torch.load('./model/mask_gen_tpc_ds2.pt'))
imputer.load_state_dict(torch.load('./model/imputer_tpc_ds2.pt'))


data = GenerateDataVal(data_file, data_ori,[0,1])
data_loader = DataLoader(data, batch_size=1, shuffle=False,drop_last=True)
imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, data_loader,1,output_dim)
print(np.mean(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=0))

