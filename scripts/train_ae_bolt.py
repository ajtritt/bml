import argparse
import glob
import warnings

from lightning.pytorch import Trainer, seed_everything
import lightning.pytorch.callbacks as cb
from lightning.pytorch.loggers import CSVLogger

from pl_bolts.models.autoencoders import AE
from pl_bolts.utils.stability import UnderReviewWarning

import torch
from torch.utils.data import DataLoader, random_split

from bml.nn.loader import FerroXDataset2D


def shape(string):
    try:
        return tuple(map(int, string.split(',')))
    except Exception:
        raise argparse.ArgumentTypeError('shape')


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='the directory with training data. should have directorie starting with "it"')
    parser.add_argument("experiment", type=str, help="the experiment name")
    parser.add_argument("-o", "--outdir", type=str, help="the directory to save output to", default='.')
    parser.add_argument("-C", "--checkpoint", type=str, help="the checkpoint file to start training from", default=None)
    parser.add_argument('-S', '--sanity', help='the directory to save checkpoints to',
                        default=False, action='store_true')
    parser.add_argument('-s', '--shape', help='the shape of the images', type=shape, default='200,200')
    parser.add_argument('-f', '--first_conv', help='first conv is 7x7', default=False, action='store_true')
    parser.add_argument('-m', '--model', type=str, help='model type to use', choices=['resnet18', 'resnet50'], default='resnet18')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size - you should not run this code if you need this explained', default=64)
    parser.add_argument('-r', '--lr', type=float, help='learning rate - you should not run this code if you need this explained', default=1e-2)
    parser.add_argument('-L', '--latent_dim', type=int, help='the size of the latent dimension', default=64)
    parser.add_argument('-e', '--epochs', type=int, help='you should not run this code if you need this explained')
    parser.add_argument('-n', '--num_nodes', type=int, help='the number of nodes to run on', default=1)
    parser.add_argument('-g', '--gpus', type=int, help='the number of GPUs to use', default=1)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UnderReviewWarning)

    seed_everything(1001)

    dset = FerroXDataset2D("training_data",  max_gate_shape=args.shape)
    subsets = random_split(dset, [0.8, 0.1, 0.1])

    train_dl = DataLoader(subsets[0], batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_dl = DataLoader(subsets[1], batch_size=args.batch_size, num_workers=4)


    model = AE(args.shape[0], latent_dim=args.latent_dim, channels=2, lr=args.lr, enc_type=args.model, first_conv=args.first_conv)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    callbacks = [
            cb.ModelCheckpoint(monitor='val_loss'),
            cb.LearningRateMonitor(logging_interval='epoch'),
   #         cb.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=3, mode="min"),
    ]

    targs = dict(
            precision=16,
            num_nodes=args.num_nodes,
            devices=args.gpus,
            ########################## default_root_dir=args.checkpoint_dir,
            enable_checkpointing=True,
            callbacks=callbacks,
            fast_dev_run=args.sanity,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=CSVLogger(args.outdir, name=args.experiment),
    )

    print("TRAINER ARGS:", targs)

    trainer = Trainer(**targs)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=args.checkpoint)

if __name__ == '__main__':
    main()
