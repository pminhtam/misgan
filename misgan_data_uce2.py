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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from matplotlib.patches import Rectangle
import pylab as plt
import pandas
import argparse
import random

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

device

"""## Structure dataset"""

class GenerateData(Dataset):
    def print_info(self):        
        print(self.data_max)

    def __init__(self, data_file):
        data = pandas.read_csv(data_file, delimiter=",",header=None, skiprows=1)[[3,6,7,15,22]].head(20000).replace(np.nan, 0)
        data['z1'] = data[3]+ data[6]
        data['z2'] = data[7] + data[15]
        self.data = data.values.astype(np.float32)

        self.maxs = np.ones((self.data.shape[1]))
        self.cur_iter = 0
        self.data_size = self.data.shape[0]        
        self.data_dim = self.data.shape[1]        
        self.population = [i for i in range(self.data_dim)]
        
        # Change the value of the records from continous domain to binary domain
#         self.data = self.records2onehot(self.data)     
        self.data_max = np.max(self.data, axis=0) + 1e-6
        self.data_min = np.min(self.data, axis=0)

        self.n_labels = len(self.data_max)

        self.data = (self.data - self.data_min) / self.data_max
        self.data = self.data.astype('float64')
        self.generate_incomplete_data(self.data)
        
 #         np.random.seed(0)
        np.random.shuffle(self.data)  
               
        self.print_info()
    def suff(self):
        np.random.shuffle(self.data)
        self.generate_incomplete_data(self.data)
    def log2(self, val):
        if val == 0: return 0
        return int(np.ceil(np.log2(val)))
    def generate_incomplete_data(self, data):
        n_masks = self.data_size
        
        masks = np.random.choice(2, size=(n_masks, self.data_dim), p=[0.2, 0.8])

#         print(n_masks)
#         print(self.data_dim)


#         masks = np.ones((n_masks,self.data_dim))
#         for i in range(n_masks):
#           data_hint = random.sample(self.population,2)
#           for j in data_hint:
#             masks[i][j] = 0
        
        self.masks =masks.astype('float64')       
        
        # Mask out missing data by zero
        self.records = self.data * self.masks
        
    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.records[index], self.masks[index], self.data[index], index

    def __len__(self):
        return self.data_size
data_file = "uce-results-by-school-2011-2015.csv"
data = GenerateData(data_file)

batch_size = 8192
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

data.data_size

data_samples, mask_samples, data_origin, _ = next(iter(data_loader))
#print(data_samples[0])
#print(mask_samples[0])
#print(data_origin[0])
#print(torch.norm(mask_samples[0] * data_origin[0] - data_samples[0]))

"""### Masking operator"""

def mask_data(data, mask, tau=0):
    return mask * data + (1 - mask) * tau

"""## MisGAN

### Generator
"""

# Must sub-class ConvGenerator to provide transform()
class FCGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),


            nn.Linear(64, self.output_dim)
        )
        
    def forward(self, input):
        # Note: noise or without noise
        hidden = self.main(input)
        hidden = self.transform(hidden)        
        return hidden 

class FCDataGenerator(FCGenerator):
    def __init__(self, input_dim, output_dim, temperature=.66):
        super().__init__(input_dim, output_dim)
        #Transform from [0, 1] to [-1,1]        
#         self.transform = lambda x: 2 * torch.sigmoid(x / temperature) - 1
        self.transform = lambda x: torch.sigmoid(x / temperature)


class FCMaskGenerator(FCGenerator):
    def __init__(self, input_dim, output_dim, temperature=.66):
        super().__init__(input_dim, output_dim)
        self.transform = lambda x: torch.sigmoid(x / temperature)

"""### Discriminator"""

class FCCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.main = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),

            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),

            nn.Linear(32, 1),
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = self.main(input)
        return out.view(-1)

"""### Training Wasserstein GAN with gradient penalty"""

class CriticUpdater:
    def __init__(self, critic, critic_optimizer, batch_size=128, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, 1, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)

    def __call__(self, real, fake):
        
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()

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

for epoch in range(6000):
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

"""## Missing data imputation"""

class Imputer(nn.Module):
    def __init__(self, input_dim, arch=(512, 512)):
        super().__init__()        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, arch[0]),
            nn.ReLU(),
            nn.Linear(arch[0], arch[1]),
            nn.ReLU(),
            
            nn.Linear(arch[1], arch[0]),
            nn.ReLU(),

            nn.Linear(arch[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim),
        )

    def forward(self, data, mask, noise):
        net = data * mask + noise * (1 - mask)
        net = net.view(data.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net).view(data.shape)
        #Rescale from [0,1] to [-1,1]
#         net = 2 * net - 1
        return data * mask + net * (1 - mask)

output_dim = data.n_labels

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

"""### Training MisGAN imputer"""

from sklearn.metrics import mean_squared_error 

def cal_loss_MSER(imputer, data_loader):
  impu_noise = torch.empty(batch_size, output_dim, device=device)
  
  imputed_data_mask = []
  origin_data_mask = []
  loss = []
  for real_data, real_mask, origin_data, _ in data_loader:
    real_data = real_data.float().to(device)
    real_mask = real_mask.float().to(device)
    impu_noise.uniform_()
    imputed_data = imputer(real_data, real_mask, impu_noise)
    
    imputed_data = imputed_data.detach().cpu().numpy()
    origin_data = origin_data.detach().cpu().numpy()
    masks = real_mask.detach().cpu().numpy()
    
    imputed_data_mask.extend(imputed_data*(1-masks))
    origin_data_mask.extend(origin_data*(1-masks))

  return imputed_data_mask,origin_data_mask

#imputer.load_state_dict(torch.load('./imputer_my_data3_0.2.pt'))

alpha = .2
beta = .2
plot_interval = 500
critic_updates = 0
loss = []
for epoch in range(8000):
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
            
            imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, DataLoader(GenerateData(data_file), batch_size=batch_size, shuffle=False,drop_last=True))  
            #print(np.sum(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=1).mean())
            loss.append(np.sum(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=1).mean())
            imputer.train()

#imputed_data_mask,origin_data_mask = cal_loss_MSER(imputer, DataLoader(GenerateData("data.csv"), batch_size=batch_size, shuffle=False,drop_last=True))

torch.save(data_gen.state_dict(), './data_gen_uce2.pt')
torch.save(mask_gen.state_dict(), './mask_gen_uce2.pt')
torch.save(imputer.state_dict(),  './imputer_uce2.pt')


import matplotlib.pyplot as plt
plt.plot(loss)
plt.savefig(data_file + "2.jpg")
