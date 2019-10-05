import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
# import pylab as plt
import pandas
import argparse
import random

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class GenerateData(Dataset):
    def print_info(self):
        print(self.data_max)

    def __init__(self, data_file):
        self.data = pandas.read_csv(data_file, delimiter=",", header=None).values.astype(np.float32)

        self.maxs = np.ones((self.data.shape[1]))
        self.cur_iter = 0
        self.data_size = self.data.shape[0]
        self.data_dim = self.data.shape[1]
        self.population = [i for i in range(self.data_dim)]

        # Change the value of the records from continous domain to binary domain
        #         self.data = self.records2onehot(self.data)
        self.data_max = np.max(self.data, axis=0)
        self.n_labels = len(self.data_max)

        self.data = self.data / self.data_max
        self.data = self.data.astype('float64')
        self.generate_incomplete_data(self.data)

        self.print_info()

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

        self.masks = masks.astype('float64')

        # Mask out missing data by zero
        self.records = self.data * self.masks

    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.records[index], self.masks[index], self.data[index], index

    def __len__(self):
        return self.data_size


data = GenerateData("data4.csv")

batch_size = 2048
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                         drop_last=True)

data_samples, mask_samples, data_origin, _ = next(iter(data_loader))


def mask_data(data, mask, tau=0):
    return mask * data + (1 - mask) * tau


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
        # Transform from [0, 1] to [-1,1]
        #         self.transform = lambda x: 2 * torch.sigmoid(x / temperature) - 1
        self.transform = lambda x: torch.sigmoid(x / temperature)


class FCMaskGenerator(FCGenerator):
    def __init__(self, input_dim, output_dim, temperature=.66):
        super().__init__(input_dim, output_dim)
        self.transform = lambda x: torch.sigmoid(x / temperature)

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
            nn.Linear(128, 1),
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        out = self.main(input)
        return out.view(-1)

nz = 128   # dimensionality of the latent code
n_critic = 5
alpha = .2
output_dim = data.n_labels
# batch_size = 1024

data_gen = FCDataGenerator(nz, output_dim).to(device)
mask_gen = FCMaskGenerator(nz, output_dim).to(device)

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


data_gen.load_state_dict(torch.load('./data_gen_my_data4.pt',map_location='cpu'))
mask_gen.load_state_dict(torch.load('./mask_gen_my_data4.pt',map_location='cpu'))
imputer.load_state_dict(torch.load('./imputer_my_data4.pt',map_location='cpu'))


class GenerateData2(Dataset):
    def print_info(self):
        print(self.data_max)

    def __init__(self, data_file, FD):
        dd = pandas.read_csv(data_file, delimiter=",", header=None)
        self.data = dd.values.astype(np.float32)
        # self.data2 = dd.head(10000).values.astype(np.float32)

        self.maxs = np.ones((self.data.shape[1]))
        self.cur_iter = 0
        self.data_size = self.data.shape[0]
        self.data_dim = self.data.shape[1]
        self.population = [i for i in range(self.data_dim)]
        self.FD = FD

        # Change the value of the records from continous domain to binary domain
        #         self.data = self.records2onehot(self.data)
        self.data_max = np.max(self.data, axis=0)
        self.n_labels = len(self.data_max)

        self.data = self.data / self.data_max
        self.data = self.data.astype('float64')
        self.generate_incomplete_data(self.data)

        np.random.seed(0)
        np.random.shuffle(self.data)
        self.print_info()

    def log2(self, val):
        if val == 0: return 0
        return int(np.ceil(np.log2(val)))

    def generate_incomplete_data(self, data):
        n_masks = self.data_size

        # masks = np.ones((n_masks, self.data_dim))
        # for i in self.FD:
        #     masks[:, i] = 0

        masks = np.zeros((n_masks, self.data_dim))
        for i in self.FD:
            masks[:, i] = 1

        self.masks = masks.astype('float64')

        # Mask out missing data by zero
        self.records = self.data * self.masks

    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.records[index], self.masks[index], self.data[index], index

    def __len__(self):
        return self.data_size


from sklearn.metrics import mean_squared_error

batch_size2 = 1


def cal_loss_MSER2(imputer, data_loader, batch_size):
    torch.manual_seed(0)
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

        imputed_data_mask.extend(imputed_data * (1 - masks))
        origin_data_mask.extend(origin_data * (1 - masks))

    return imputed_data_mask, origin_data_mask


def calc_loss_FD(vt,vp):
    data_file = "data4_test.csv"
    imputed_data_mask, origin_data_mask = cal_loss_MSER2(imputer, DataLoader(GenerateData2(data_file, vt),
                                                                             batch_size=batch_size2, shuffle=False,
                                                                             drop_last=True), batch_size2)
    print(np.mean(np.square(np.subtract(imputed_data_mask, origin_data_mask)), axis=0))
    return np.mean(np.square(np.subtract(imputed_data_mask, origin_data_mask)), axis=0)[vp]

# print(calc_loss_FD([0,1],2))