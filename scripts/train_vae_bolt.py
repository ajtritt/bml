import warnings

from lightning.pytorch import Trainer, seed_everything
import lightning.pytorch.callbacks as cb
from lightning.pytorch.loggers import CSVLogger

from pl_bolts.models.autoencoders import VAE
from pl_bolts.utils.stability import UnderReviewWarning

import torch
from torch.utils.data import DataLoader, random_split

from bml.nn.loader import FerroXDataset2D



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

    warnings.filterwarnings("ignore", category=UnderReviewWarning)

    seed_everything(1001)

    dset = FerroXDataset2D("training_data")
    subsets = random_split(dset, [0.8, 0.1, 0.1])

    train_dl = DataLoader(subsets[0], batch_size=64, num_workers=4)
    val_dl = DataLoader(subsets[1], batch_size=64, num_workers=4)

    model = VAE(200, latent_dim=64, channels=2)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    callbacks = [cb.ModelCheckpoint(monitor='loss')]

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
    trainer = Trainer(**targs)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == '__main__':
    main()
