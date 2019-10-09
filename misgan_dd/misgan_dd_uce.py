import pandas
import cf
from misgan_dd.model_dd import *
from misgan.data_generate import GenerateData
from misgan_dd.misgan_dd_train import train

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


"""## Structure dataset"""

data_file = "uce-results-by-school-2011-2015.csv"

data_ori = pandas.read_csv("data/" +data_file, delimiter=",", header=None, skiprows=1)[[3, 6, 7, 15, 22]].head(
    cf.num_row).replace(np.nan, 0).values.astype(np.float32)

data = GenerateData(data_ori)

netG_imp,netD_imp = train(data)

torch.save(netG_imp,   './model/Gim_uce.pt')
torch.save(netD_imp,  './model/Dim_uce.pt')