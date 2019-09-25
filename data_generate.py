

import numpy as np
from torch.utils.data import Dataset


class GenerateData(Dataset):
    def print_info(self):
        print(self.data_max)

    def __init__(self, data_file,data):
        self.data = data

        self.maxs = np.ones((self.data.shape[1]))
        self.cur_iter = 0
        self.data_size = self.data.shape[0]
        self.data_dim = self.data.shape[1]
        self.population = [i for i in range(self.data_dim)]

        # Change the value of the records from continous domain to binary domain
        #         self.data = self.records2onehot(self.data)
        self.data_max = np.max(self.data ,axis=0) + 1e-6
        self.data_min = np.min(self.data ,axis=0)

        self.n_labels = len(self.data_max)

        self.data = (self.data - self.data_min) /(self.data_max -self.data_min)
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

        self.masks =masks.astype('float64')

        # Mask out missing data by zero
        self.records = self.data * self.masks

    def __getitem__(self, index):
        # return index so we can retrieve the mask location from self.mask_loc
        return self.records[index], self.masks[index], self.data[index], index

    def __len__(self):
        return self.data_size