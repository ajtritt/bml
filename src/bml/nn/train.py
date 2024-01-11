import lightning as L
import lightning.pytorch.callbacks as cb
from lightning.pytorch.cli import LightningCLI

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .unet import UNet


class LitUNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(3)
        self.loss_fcn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        X, Y = batch
        output = self.unet(X)
        loss = self.loss_fcn(Y, output)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, Y = batch
        output = self.unet(X)
        val_loss = self.loss_fcn(Y, output)
        self.log("val_loss", val_loss)


class TestPlanarDataModule(L.LightningDataModule):

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        dset = FerroXDataset(glob.glob("/pscratch/sd/a/ajtritt/bml/test_planar/it*")) #
        self.train, self.val, self.test = data.random_split(
            dset, [80, 10, 10], generator=torch.Generator().manual_seed(31)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train)

    def val_dataloader(self):
        return data.DataLoader(self.val)

    def test_dataloader(self):
        return data.DataLoader(self.test)

    def teardown(self):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        pass


def main(argv=None):
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--checkpoint_dir", help="the directory to save checkpoints to", default=None)
    parser.add_argument("-s", "--seed", help="the seed to use for randomly splitting data", default=27, type=int)
    parser.add_argument("-S", "--sanity", help="the directory to save checkpoints to",
                        default=False, action='store_true')
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.data_dir


    model = LitUNet(UNet(3))


    train_set_size = int(len(dset) * 0.8)
    valid_set_size = len(dset) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(args.seed)
    train_set, valid_set = random_split(dset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set)
    valid_loader = DataLoader(valid_set)

    model = LitUNet(unet)


    # Init ModelCheckpoint callback, monitoring 'val_loss'
    callbacks = [cb.ModelCheckpoint(monitor="val_loss")]

    targs = dict(
            default_root_dir=args.checkpoint_dir,
            enable_checkpointing=True,
            callbacks=callbacks,

    )


    # train with both splits
    trainer = L.Trainer(**targs)
    trainer.fit(model, train_loader, valid_loader)


def lit_cli():
    cli = LightningCLI(LitUNet, TestPlanarDataModule)

if __name__ == '__main__':
    lit_cli()
