
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad,Variable
from torch.utils.data import DataLoader

import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
class Discriminator(nn.Module):

    def __init__(self, n_labels):
        super(Discriminator, self).__init__()
        input_dim = n_labels
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class MultipleBranchesDiscriminator(nn.Module):
    def __init__(self, n_labels):
        super(MultipleBranchesDiscriminator, self).__init__()
        input_dim = n_labels
        branches = []
        for i in range(1):
            branch = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            branches.append(branch)
        self.branches = nn.Sequential(*branches)

    def forward(self, input):
        scores = []
        for branch in self.branches:
            score = branch(input)
            scores.append(score)
        return torch.cat(scores, dim=1)

class Divide(nn.Module):
    def __init__(self, lambda_):
        super(Divide, self).__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return x/self.lambda_

class Generator_Imputer(nn.Module):

    def __init__(self, n_labels):
        super(Generator_Imputer, self).__init__()
        input_dim = n_labels
        # Note: without branches
        self.main = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            # Note: performed better without Tanh activation
            nn.Sigmoid()
        )

    def forward(self, input, m):
        # Note: noise or without noise
        hidden = self.main(input * m)
        return input * m + (1 - m) * hidden

def calc_gradient_penalty(netD, real_data, fake_data, batch_size,use_cuda,lambda_):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * lambda_
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


from sklearn.metrics import mean_squared_error


def cal_loss_MSER(imputer, data_loader):
    imputed_data_mask = []
    origin_data_mask = []
    loss = []
    for real_data, real_mask, origin_data, _ in data_loader:
        origin_data = origin_data.float().to(device)
        real_mask = real_mask.float().to(device)
        imputed_data = imputer(origin_data, real_mask)

        imputed_data = imputed_data.detach().cpu().numpy()
        origin_data = origin_data.detach().cpu().numpy()
        masks = real_mask.detach().cpu().numpy()

        imputed_data_mask.extend(imputed_data * (1 - masks))
        origin_data_mask.extend(origin_data * (1 - masks))

    return imputed_data_mask, origin_data_mask


def train(data,batch_size,epochs,n_iter_d,n_iter_g,beta1,beta2,lr,lambda_):
    # batch_size = cf.batch_size
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                             drop_last=True)

    # lambda_ = 10
    # lr = 5e-4
    # beta1 = 0.0001
    # beta2 = 0.999
    # n_iter_d = 2
    # n_iter_g = 1
    # n_iter = cf.n_iter

    netG_imp = Generator_Imputer(data.n_labels)
    netG_imp.apply(weights_init)

    netD_imp = Discriminator(data.n_labels)
    netD_imp.apply(weights_init)

    if use_cuda:
        netG_imp = netG_imp.cuda()
        netD_imp = netD_imp.cuda()
    optimizerG_imp = optim.Adam(
        netG_imp.parameters(), lr=lr, betas=(beta1, beta2))
    # Note: transfer learning with lr/10
    optimizerD_imp = optim.Adam(
        netD_imp.parameters(), lr=lr, betas=(beta1, beta2))


    for _ in range(epochs):
    # for iter in range(5):
    #     print(iter)
    #     data.suff()
    #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
    #                          drop_last=True)
        for _, real_mask, real_data, _ in data_loader:
            for _ in range(n_iter_d):
            # for real_data, real_mask, origin_data, _ in data_loader:
            #     real_data, real_mask, origin_data, _ = next(iter(data_loader))
                netG_imp.zero_grad()
                netD_imp.zero_grad()
                if use_cuda:
                    real_data, real_mask = real_data.float().cuda(), real_mask.float().cuda()

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

            # for real_data, real_mask, origin_data, _ in data_loader:
            for _ in range(n_iter_g):
                # real_data, real_mask, origin_data, _ = next(iter(data_loader))

                netG_imp.zero_grad()
                netD_imp.zero_grad()

                if use_cuda:
                    real_data, real_mask = real_data.float().cuda(), real_mask.float().cuda()
                else:
                    real_data, real_mask = real_data.float(), real_mask.float()
                fake_imp = netG_imp(real_data, real_mask)

                G_imp_cost = netD_imp(fake_imp)
                # G_imp_cost = -G_imp_cost.mean()* 0.0 + 1.0 * torch.sqrt(torch.sum((real_data-fake_imp)**2))
                G_imp_cost = -G_imp_cost.mean()
                G_imp_cost.backward()
                optimizerG_imp.step()
    return netG_imp,netD_imp