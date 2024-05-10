import importlib
import sys

import lightning as L
from lightning.pytorch.cli import LightningCLI

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=['vae', 'unet'], help='the model to run training for')
    args, extras = parser.parse_known_args()

    module = importlib.import_module(f'.models.{args.model}', package=__package__)
    litmod = None
    datamod = None
    for k in dir(module):
        obj = getattr(module, k)
        if isinstance(obj, type):
            if litmod is None:
                if issubclass(obj, L.LightningModule):
                    litmod = obj
            if datamod is None:
                if issubclass(obj, L.LightningDataModule):
                    datamod = obj

    sys.argv = sys.argv[1:]
    cli = LightningCLI(litmod, datamod)

if __name__ == '__main__':
    main()
