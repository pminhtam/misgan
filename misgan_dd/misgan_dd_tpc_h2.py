import pandas
import cf
from misgan_dd.model_dd import *
from misgan.data_generate import GenerateData
from misgan_dd.misgan_dd_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


"""## Structure dataset"""

data_file = "lineitem.tbl.8"

data = pandas.read_csv("data/" +data_file, delimiter="|", header=None)[[0, 1, 2, 4, 5]].head(cf.num_row)

data['z1'] = data[0] + data[1]
data['z2'] = data[2] + data[4]

data_ori = data.values.astype(np.float32)

data = GenerateData(data_ori)

netG_imp,netD_imp = train(data)

torch.save(netG_imp,   './model/Gim_h2.pt')
torch.save(netD_imp,  './model/Dim_h2.pt')