
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad,Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


class Discriminator(nn.Module):

    def __init__(self, n_labels):
        super(Discriminator, self).__init__()
        input_dim = sum(n_labels)
        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        output = self.main(input)
        return output

class MultipleBranchesDiscriminator(nn.Module):
    def __init__(self, n_labels):
        super(MultipleBranchesDiscriminator, self).__init__()
        input_dim = n_labels
        branches = []
        for i in range(n_labels):
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
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            # Note: performed better without Tanh activation
            # nn.Tanh(), #
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