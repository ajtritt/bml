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

class AttnBlock(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        self.norm = nn.LayerNorm(n_features, bias=False)
        self.q = nn.Linear(n_features, n_features, bias=False)
        self.k = nn.Linear(n_features, n_features, bias=False)
        self.v = nn.Linear(n_features, n_features, bias=False)
        self.proj_out = nn.Linear(n_features, n_features, bias=False)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c = q.shape
        w_ = torch.einsum('bp,bq->bpq', q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        h_ = torch.einsum('bqk,bk->bq', w_, v)
        h_ = self.proj_out(h_)

        return x+h_


class AE(nn.Module):

    def __init__(self, device_shape):
        super().__init__()

        latent_dim = 64

        # encoder, decoder
        self.encoder = resnet10_encoder(True, False, layer1_channels=8)
        self.decoder = resnet10_decoder(latent_dim, device_shape, True, False, layer4_channels=latent_dim)

        self.attn = AttnBlock(latent_dim)

        self.mse = nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, batch):
        #x, _ = batch
        if len(batch.shape) == 4:
            batch = batch[None, :]
        x = batch


        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)

        x_encoded = self.attn(x_encoded)

        x_hat = self.decoder(x_encoded)

        # reconstruction loss
        recon_loss = self.mse(x_hat, x)

        return recon_loss


class LitAE(L.LightningModule):
    """Lightning Module aka Network model"""

    def __init__(self, max_gate_shape):
        """
        Args:
            n_channels: The number of values in each FerroX grid. By default,
                        3 channels are expected i.e. Pz, Ez, Phi
        """
        super().__init__()
        self.ae = AE(max_gate_shape)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch = batch[0]
        recon_loss = self.ae(batch)
        self.log('recon_loss', recon_loss)
        return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        batch = batch[0]
        recon_loss = self.ae(batch)
        return recon_loss


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
    model = LitAE(max_shape)
    data = LightningDataModule(args.data_dir, max_shape)


    # Init ModelCheckpoint callback, monitoring 'val_loss'
    callbacks = [cb.ModelCheckpoint(monitor='recon_loss')]

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

def test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='the directory with training data. should have directorie starting with "it"')
    args = parser.parse_args()

    mgs = (200, 200, 52)
    dset = FerroXDataset(args.data_dir, max_gate_shape=mgs) #
    ae = AE(mgs)

    dl = DataLoader(dset, batch_size=2, shuffle=False, drop_last=False)

    for batch in dl:
        ae(batch)
        break


if __name__ == '__main__':
    main()
