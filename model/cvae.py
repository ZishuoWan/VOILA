import torch
from torch import nn
from torch.nn import functional as F

class ConditionalVAE(nn.Module):

    def __init__(self,
                 condition_shape,
                 image_shape,
                 latent_dim: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        condition_channel, condition_shape, _ = condition_shape
        self.image_shape = image_shape

        self.gap = nn.AdaptiveAvgPool3d((1,1,1))

        modules = []
        if hidden_dims is None:
            hidden_dims = [4, 8, 16, 32]

        mid_channel = 2
        if isinstance(condition_shape, (tuple, list)):
            condition_shape = condition_shape[0]
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(mid_channel, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.Tanh())
            )
            mid_channel = h_dim
            condition_shape = condition_shape//2

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv3d(h_dim, latent_dim, 1, 1)
        self.fc_var = nn.Conv3d(h_dim, latent_dim, 1, 1)
        self.final_shape = condition_shape
        self.final_dim = hidden_dims[-1]


        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(
                            nn.Conv3d(latent_dim + condition_channel, hidden_dims[-1], 1, 1),
                            nn.BatchNorm3d(hidden_dims[-1]),
                            nn.Tanh())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv3d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.Tanh())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv3d(hidden_dims[-1], 1, kernel_size=3, stride=1, padding=1),
                            nn.Sigmoid())

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, condition):
        x = torch.cat([input, condition], dim = 1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, condition], dim = 1)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               condition,
               **kwargs):
        with torch.no_grad():
            B, C, D, H, W = condition.shape
            z = torch.randn((B, self.latent_dim, D, H, W), device=condition.device)

            z = torch.cat([z, condition], dim=1)
            samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        return self.forward(x, **kwargs)[0]
