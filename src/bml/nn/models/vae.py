import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import lightning as L

from .resnet import resnet10_encoder, resnet10_decoder
from ..loader import InitFerroXDataset


class VAE(nn.Module):

    def __init__(self, device_shape, enc_out_dim=64, latent_dim=32):
        super().__init__()

        # encoder, decoder
        self.encoder = resnet10_encoder(True, False, layer1_channels=8)
        self.decoder = resnet10_decoder(latent_dim, device_shape, True, False, layer4_channels=latent_dim)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, batch):
        #x, _ = batch
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)


        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        return elbo, kl.mean(), recon_loss.mean()


class LightningModule(L.LightningModule):
    """Lightning Module aka Network model"""

    def __init__(self, max_gate_shape, latent_dim=32):
        """
        Args:
            n_channels: The number of values in each FerroX grid. By default,
                        3 channels are expected i.e. Pz, Ez, Phi
        """
        super().__init__()
        self.vae = VAE(max_gate_shape, latent_dim=latent_dim)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch = batch[0]
        elbo, kl, recon_loss = self.vae(batch)
        self.log('elbo', elbo)
        self.log('recon_loss', recon_loss)
        self.log('kl', kl)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch = batch[0]
        elbo, kl, recon_loss = self.vae(batch)
        self.log('elbo', elbo)
        self.log('recon_loss', recon_loss)
        self.log('kl', kl)
        return loss


class LightningDataModule(L.LightningDataModule):
    """Lightning Data Module"""

    def __init__(self, data_dir_glob, batch_size, max_gate_shape):
        """
        Args:
            data_dir_glob: The glob string for getting the FerroX run directories to
                           use for training
        """
        super().__init__()
        self.data_dir_glob = data_dir_glob
        self.batch_size = batch_size
        self.max_gate_shape = max_gate_shape

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dset = InitFerroXDataset(glob.glob(self.data_dir_glob), max_gate_shape=self.max_gate_shape) #
        self.train, self.val, self.test = random_split(
            dset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(31)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val)

    def test_dataloader(self):
        return DataLoader(self.test)

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass

