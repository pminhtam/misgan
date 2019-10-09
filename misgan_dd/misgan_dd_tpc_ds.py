import pandas
import cf
from misgan_dd.model_dd import *
from misgan.data_generate import GenerateData
from misgan_dd.misgan_dd_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


"""## Structure dataset"""

data_file = "store_returns.dat"
data_ori = pandas.read_csv("data/" +data_file, delimiter="|", header=None)[[0,1,2,3,4,5]].head(cf.num_row).replace(np.nan, 0).values.astype(np.float32)


data = GenerateData(data_ori)

netG_imp,netD_imp = train(data)

torch.save(netG_imp,   './model/Gim_ds.pt')
torch.save(netD_imp,  './model/Dim_ds.pt')