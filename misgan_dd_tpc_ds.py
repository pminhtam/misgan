import numpy as np
import pandas
import argparse
import random
import cf
from model_dd import *
from data_generate import GenerateData

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


"""## Structure dataset"""

data_file = "store_returns.dat"
data = pandas.read_csv(data_file, delimiter="|", header=None)[[0,1,2,3,4,5]].head(cf.num_row).replace(np.nan, 0).values.astype(np.float32)


data = GenerateData(data_file,data)

batch_size = cf.batch_size
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

lambda_ = 10
lr = 5e-4
beta1 = 0.0001
beta2 = 0.999
n_iter_d = 2
n_iter_g = 1
n_iter = 100000

netG_imp = Generator_Imputer(data.n_labels)
netG_imp.apply(weights_init)

netD_imp = MultipleBranchesDiscriminator(data.n_labels)
netD_imp.apply(weights_init)

if use_cuda:
    netG_imp = netG_imp.cuda()
    netD_imp = netD_imp.cuda()
optimizerG_imp = optim.Adam(
    netG_imp.parameters(), lr=lr, betas=(beta1, beta2))
# Note: transfer learning with lr/10
optimizerD_imp = optim.Adam(
    netD_imp.parameters(), lr=lr, betas=(beta1, beta2))


for iter in range(n_iter):
    print(iter)
    for real_data, real_mask, origin_data, _ in data_loader:
        netG_imp.zero_grad()
        netD_imp.zero_grad()


        if use_cuda:
            real_data, real_mask = real_data.cuda(), real_mask.cuda()

            fake_imp = netG_imp(real_data, real_mask).detach()
            real_imp = real_data.detach()
        else:
            real_data, real_mask = real_data.float(), real_mask.float()

            fake_imp = netG_imp(real_data, real_mask)
            real_imp = real_data
        # train with real
        D_real_imp = netD_imp(real_data)
        D_real_imp = D_real_imp.mean()

        # train with fake
        D_fake_imp = netD_imp(fake_imp)
        D_fake_imp = D_fake_imp.mean()

        gradient_penalty = calc_gradient_penalty(
            netD_imp, real_imp, fake_imp, batch_size,use_cuda,lambda_)
        D_imp_cost = D_fake_imp - D_real_imp + gradient_penalty
        dist = D_real_imp - D_fake_imp
        D_imp_cost.backward()
        optimizerD_imp.step()

    ############################
    # (2) Update G_imp network
    ############################

    for real_data, real_mask, origin_data, _ in data_loader:
        netG_imp.zero_grad()
        netD_imp.zero_grad()

        if use_cuda:
            real_data, real_mask = real_data.cuda(), real_mask.cuda()
        else:
            real_data, real_mask = real_data.float(), real_mask.float()
        fake_imp = netG_imp(real_data, real_mask)

        G_imp_cost = netD_imp(fake_imp)
        # G_imp_cost = -G_imp_cost.mean()* 0.0 + 1.0 * torch.sqrt(torch.sum((real_data-fake_imp)**2))
        G_imp_cost = -G_imp_cost.mean()
        G_imp_cost.backward()
        optimizerG_imp.step()
torch.save(netG_imp,   './model/Gim_fd.pt')
torch.save(netD_imp,  './model/Dim_fd.pt')