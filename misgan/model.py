import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader

import numpy as np

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
            nn.ReLU(False),
            nn.Linear(512, 256),
            nn.ReLU(False),
            nn.Linear(256, 128),
            nn.ReLU(False),

            nn.Linear(128, 64),
            nn.ReLU(False),
            nn.Linear(64, 32),
            nn.ReLU(False),

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
    np.random.seed(0)
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


def train(data,batch_size,nz,alpha,beta,epoch_1,epoch_2,lrate1,imputer_lrate):

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                             drop_last=True)


    n_critic = 5
    # alpha = .2
    output_dim = data.n_labels
    # batch_size = 1024

    data_gen = FCDataGenerator(nz, output_dim).to(device)
    mask_gen = FCMaskGenerator(nz, output_dim).to(device)

    data_critic = FCCritic(output_dim).to(device)
    mask_critic = FCCritic(output_dim).to(device)

    data_noise = torch.empty(batch_size, nz, device=device)
    mask_noise = torch.empty(batch_size, nz, device=device)

    # lrate1 = 1e-4
    data_gen_optimizer = optim.Adam(
        data_gen.parameters(), lr=lrate1, betas=(.5, .9))
    mask_gen_optimizer = optim.Adam(
        mask_gen.parameters(), lr=lrate1, betas=(.5, .9))

    data_critic_optimizer = optim.Adam(
        data_critic.parameters(), lr=lrate1, betas=(.5, .9))
    mask_critic_optimizer = optim.Adam(
        mask_critic.parameters(), lr=lrate1, betas=(.5, .9))

    update_data_critic = CriticUpdater(
        data_critic, data_critic_optimizer, batch_size)
    update_mask_critic = CriticUpdater(
        mask_critic, mask_critic_optimizer, batch_size)

    """### Training MisGAN"""

    # data_gen.load_state_dict(torch.load('./data_gen_my_data3_0.2.pt'))
    # mask_gen.load_state_dict(torch.load('./mask_gen_my_data3_0.2.pt'))

    plot_interval = 500
    critic_updates = 0

    for epoch in range(epoch_1):
    # for epoch in range(1):
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


                mask_loss = -mask_critic(fake_mask).mean()
                (mask_loss + data_loss * alpha).backward()

                data_gen_optimizer.step()
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

    # imputer_lrate = 2e-4
    imputer_optimizer = optim.Adam(
        imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))
    impu_critic_optimizer = optim.Adam(
        impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))
    update_impu_critic = CriticUpdater(
        impu_critic, impu_critic_optimizer, batch_size)

    """### Training MisGAN imputer"""

    # alpha = .2
    # beta = .2
    plot_interval = 500
    critic_updates = 0
    loss = []
    for epoch in range(epoch_2):
    # for epoch in range(1):
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

                imputed_data_mask, origin_data_mask = cal_loss_MSER(imputer, DataLoader(data,
                                                                                        batch_size=1,
                                                                                        shuffle=False, drop_last=True),1, output_dim)
                # print(np.sum(np.square(np.subtract(imputed_data_mask,origin_data_mask)),axis=1).mean())
                loss.append(np.sum(np.square(np.subtract(imputed_data_mask, origin_data_mask)), axis=1).mean())
                imputer.train()
    return data_gen,mask_gen,imputer,loss

