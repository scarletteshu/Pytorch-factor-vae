import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import configs as cfg


class FactorVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 gamma: float = 10.,
                 **kwargs) -> None:

        super(FactorVAE, self).__init__()

        self.latent_dim = latent_dim
        self.gamma = gamma

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=h_dim,
                              kernel_size=4,
                              stride=2,
                              padding=1
                              ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()  # [256, 128, 64, 32]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1
                               ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],
                      out_channels=1,
                      kernel_size=3,
                      padding=1
                      ),
            nn.Sigmoid())

    def encode(self, input: Tensor):

        result = self.encoder(input)  # 576, 256, 4, 4
        # batch_size holds, flatten dimensions after it
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 4)

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        std = torch.exp(0.5 * logvar)
        # Returns a tensor with the same size as std that is filled with random numbers from a normal distribution
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), mu, log_var, z]

    def permute_latent(self, z: Tensor)->Tensor:
        #z: B X D
        B, D = z.size()
        inds = torch.cat([D*i + torch.randperm(D) for i in range(B)])
        return z.view(-1)[inds].view(B, D)

    def test(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples


# Disicrinimator
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )

    def forward(self, z):
        return self.layer(z)
