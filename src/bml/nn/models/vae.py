import glob
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import lightning.pytorch.callbacks as cb

from .resnet import resnet10_encoder, resnet10_decoder
from ..loader import FerroXDataset


class VAE(nn.Module):

    def __init__(self, device_shape, latent_dim=32):
        super().__init__()

        # encoder, decoder
        self.encoder = resnet10_encoder(True, False, layer1_channels=8)
        self.decoder = resnet10_decoder(latent_dim, device_shape, True, False, layer4_channels=latent_dim)

        enc_out_dim = 64

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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

    def forward(self, batch, annealer=None):
        #x, _ = batch
        if len(batch.shape) == 4:
            batch = batch[None, :]
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        if annealer is not None:
            kl = annealer(kl)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        return elbo, kl.mean(), recon_loss.mean()



class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape='cosine', baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = 'none'
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            y = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = ((self.total_steps / 2) - self.current_step)
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            y = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return


class LitVAE(L.LightningModule):
    """Lightning Module aka Network model"""

    def __init__(self, max_gate_shape, latent_dim=32, anneal_steps=20, cyclical_anneal=True):
        """
        Args:
            n_channels: The number of values in each FerroX grid. By default,
                        3 channels are expected i.e. Pz, Ez, Phi
        """
        super().__init__()
        self.vae = VAE(max_gate_shape, latent_dim=latent_dim)
        self.annealer = Annealer(total_steps=anneal_steps, cyclical=cyclical_anneal)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch = batch[0]
        elbo, kl, recon_loss = self.vae(batch, annealer=self.annealer)
        self.log('elbo', elbo)
        self.log('recon_loss', recon_loss)
        self.log('kl', kl)
        return elbo

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
        return elbo


class LightningDataModule(L.LightningDataModule):
    """Lightning Data Module"""

    def __init__(self, data_dir, max_gate_shape):
        """
        Args:
            data_dir: the directory containing FerroX run directories

        """
        super().__init__()
        self.data_dir = data_dir
        self.max_gate_shape = max_gate_shape

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dset = FerroXDataset(self.data_dir, max_gate_shape=self.max_gate_shape) #
        self.train, self.val, self.test = random_split(
            dset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(31)
        )

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=32,
                          shuffle=False,
                          drop_last=False,
                          num_workers=3)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=32,
                          shuffle=False,
                          drop_last=False,
                          num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test)

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass


def main(argv=None):
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='the directory with training data. should have directorie starting with "it"')
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument('-C', '--checkpoint_dir', help='the directory to save checkpoints to', default=None)
    #parser.add_argument('-s', '--seed', help='the seed to use for randomly splitting data', default=27, type=int)
    parser.add_argument('-S', '--sanity', help='the directory to save checkpoints to',
                        default=False, action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, help='you should not run this code if you need this explained')
    parser.add_argument('-e', '--epochs', type=int, help='you should not run this code if you need this explained')
    parser.add_argument('-n', '--num_nodes', type=int, help='the number of nodes to run on', default=1)
    parser.add_argument('-g', '--gpus', type=int, help='the number of GPUs to use', default=1)
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.data_dir


    max_shape = (200, 200, 52)
    model = LitVAE(max_shape)
    data = LightningDataModule(args.data_dir, max_shape)


    # Init ModelCheckpoint callback, monitoring 'val_loss'
    callbacks = [cb.ModelCheckpoint(monitor='elbo')]

    targs = dict(
            num_nodes=args.num_nodes,
            devices=args.gpus,
            default_root_dir=args.checkpoint_dir,
            enable_checkpointing=True,
            callbacks=callbacks,
            fast_dev_run=args.sanity,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=CSVLogger(args.outdir, name=args.experiment),
            #callbacks=[EarlyStopping(monitor=model.val_metric, min_delta=0.001, patience=3, mode="min")])
    )

    print("HELLO", targs)
    # train with both splits
    trainer = L.Trainer(**targs)
    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
