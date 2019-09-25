import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

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
        # Transform from [0, 1] to [-1,1]
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
        grad_penalty = ((grad_d.norm(dim=1) - 1 )**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()


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
        # Rescale from [0,1] to [-1,1]
        #         net = 2 * net - 1
        return data * mask + net * (1 - mask)


def cal_loss_MSER(imputer, data_loader,batch_size,output_dim):
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
