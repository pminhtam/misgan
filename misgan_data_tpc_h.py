# -*- coding: utf-8 -*-
"""misgan_data_my_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T3zncu3EVQNR-6kJeHvepeVIOpF6ORy-

# MisGAN: Learning from Incomplete Data with GANs
"""

# !pip install --upgrade torch torchvision

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


import numpy as np

import pandas
import argparse
import random
import cf
from model import *
from data_generate import GenerateData
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device


data_file = "lineitem.tbl.8"

data = pandas.read_csv(data_file, delimiter="|", header=None)[[0, 1, 2, 4, 5]].head(cf.num_row).values.astype(
    np.float32)

data = GenerateData(data_file,data)

batch_size = 8192
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

data.data_size

data_samples, mask_samples, data_origin, _ = next(iter(data_loader))
#print(data_samples[0])
#print(mask_samples[0])
#print(data_origin[0])
#print(torch.norm(mask_samples[0] * data_origin[0] - data_samples[0]))


nz = 128   # dimensionality of the latent code
n_critic = 5
alpha = .2
output_dim = data.n_labels
# batch_size = 1024

data_gen = FCDataGenerator(nz, output_dim).to(device)
mask_gen = FCMaskGenerator(nz, output_dim).to(device)

data_critic = FCCritic(output_dim).to(device)
mask_critic = FCCritic(output_dim).to(device)

data_noise = torch.empty(batch_size, nz, device=device)
mask_noise = torch.empty(batch_size, nz, device=device)

lrate = 1e-4
data_gen_optimizer = optim.Adam(
    data_gen.parameters(), lr=lrate, betas=(.5, .9))
mask_gen_optimizer = optim.Adam(
    mask_gen.parameters(), lr=lrate, betas=(.5, .9))

data_critic_optimizer = optim.Adam(
    data_critic.parameters(), lr=lrate, betas=(.5, .9))
mask_critic_optimizer = optim.Adam(
    mask_critic.parameters(), lr=lrate, betas=(.5, .9))

update_data_critic = CriticUpdater(
    data_critic, data_critic_optimizer, batch_size)
update_mask_critic = CriticUpdater(
    mask_critic, mask_critic_optimizer, batch_size)

"""### Training MisGAN"""

#data_gen.load_state_dict(torch.load('./data_gen_my_data3_0.2.pt'))
#mask_gen.load_state_dict(torch.load('./mask_gen_my_data3_0.2.pt'))

plot_interval = 500
critic_updates = 0

for epoch in range(cf.epoch_1):
    for real_data, real_mask, origin_data, _ in data_loader:

        real_data = real_data.float().to(device)
        real_mask = real_mask.float().to(device)

        # Update discriminators' parameters
        data_noise.normal_()
        mask_noise.normal_()

        fake_data = data_gen(data_noise)
        fake_mask = mask_gen(mask_noise)

        masked_fake_data = mask_data(fake_data, fake_mask)
        masked_real_data = mask_data(real_data, real_mask)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)

        critic_updates += 1

        if critic_updates == n_critic:
            critic_updates = 0

            # Update generators' parameters
            for p in data_critic.parameters():
                p.requires_grad_(False)
            for p in mask_critic.parameters():
                p.requires_grad_(False)

            data_gen.zero_grad()
            mask_gen.zero_grad()

            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = mask_data(fake_data, fake_mask)

            data_loss = -data_critic(masked_fake_data).mean()
            data_loss.backward(retain_graph=True)
            data_gen_optimizer.step()

            mask_loss = -mask_critic(fake_mask).mean()
            (mask_loss + data_loss * alpha).backward()
            mask_gen_optimizer.step()

            for p in data_critic.parameters():
                p.requires_grad_(True)
            for p in mask_critic.parameters():
                p.requires_grad_(True)

    if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
        # Although it makes no difference setting eval() in this example, 
        # you will need those if you are going to use modules such as 
        # batch normalization or dropout in the generators.
        data_gen.eval()
        mask_gen.eval()

        with torch.no_grad():
            print('Epoch:', epoch)
            
            data_noise.normal_()
            data_samples = data_gen(data_noise)
            print(data_samples[0])
            
            mask_noise.normal_()
            mask_samples = mask_gen(mask_noise)
            print(mask_samples[0])
     
        data_gen.train()
        mask_gen.train()


imputer = Imputer(output_dim).to(device)
impu_critic = FCCritic(output_dim).to(device)
impu_noise = torch.empty(batch_size, output_dim, device=device)

imputer_lrate = 2e-4
imputer_optimizer = optim.Adam(
    imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))
impu_critic_optimizer = optim.Adam(
    impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))
update_impu_critic = CriticUpdater(
    impu_critic, impu_critic_optimizer, batch_size)



#imputer.load_state_dict(torch.load('./imputer_my_data3_0.2.pt'))

alpha = .2
beta = .2
plot_interval = 500
critic_updates = 0
loss = []
for epoch in range(cf.epoch_2):
#     print("Epoch %d " % epoch)
    data.suff()
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)
    for real_data, real_mask, origin_data, index in data_loader:

        real_data = real_data.float().to(device)
        real_mask = real_mask.float().to(device)

        masked_real_data = mask_data(real_data, real_mask)

        # Update discriminators' parameters
        data_noise.normal_()
        fake_data = data_gen(data_noise)

        mask_noise.normal_()
        fake_mask = mask_gen(mask_noise)
        masked_fake_data = mask_data(fake_data, fake_mask)

        impu_noise.uniform_()
        imputed_data = imputer(real_data, real_mask, impu_noise)

        update_data_critic(masked_real_data, masked_fake_data)
        update_mask_critic(real_mask, fake_mask)
        update_impu_critic(fake_data, imputed_data)

        critic_updates += 1

        if critic_updates == n_critic:
            critic_updates = 0

            # Update generators' parameters
            for p in data_critic.parameters():
                p.requires_grad_(False)
            for p in mask_critic.parameters():
                p.requires_grad_(False)
            for p in impu_critic.parameters():
                p.requires_grad_(False)

            data_noise.normal_()
            fake_data = data_gen(data_noise)

            mask_noise.normal_()
            fake_mask = mask_gen(mask_noise)
            masked_fake_data = mask_data(fake_data, fake_mask)

            impu_noise.uniform_()
            imputed_data = imputer(real_data, real_mask, impu_noise)

            data_loss = -data_critic(masked_fake_data).mean()
            mask_loss = -mask_critic(fake_mask).mean()
            impu_loss = -impu_critic(imputed_data).mean()

            mask_gen.zero_grad()
            (mask_loss + data_loss * alpha).backward(retain_graph=True)
            mask_gen_optimizer.step()

            data_gen.zero_grad()
            (data_loss + impu_loss * beta).backward(retain_graph=True)
            data_gen_optimizer.step()

            imputer.zero_grad()
            impu_loss.backward()
            imputer_optimizer.step()

            for p in data_critic.parameters():
                p.requires_grad_(True)
            for p in mask_critic.parameters():
                p.requires_grad_(True)
            for p in impu_critic.parameters():
                p.requires_grad_(True)

    if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
        with torch.no_grad():
            imputer.eval()
            
            imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, DataLoader(GenerateData(data_file), batch_size=batch_size, shuffle=False,drop_last=True),batch_size,output_dim)
            #print(np.sum(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=1).mean())
            loss.append(np.sum(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=1).mean())
            imputer.train()

#imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, DataLoader(GenerateData("data.csv"), batch_size=batch_size, shuffle=False,drop_last=True))

torch.save(data_gen.state_dict(), './data_gen_tpc_h.pt')
torch.save(mask_gen.state_dict(), './mask_gen_tpc_h.pt')
torch.save(imputer.state_dict(),  './imputer_tpc_h.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig(data_file + ".jpg")
