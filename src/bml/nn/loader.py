import os
import glob
import gzip
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import yt

from .utils import easy_pad
from ..bayes_opt import read_inputs, get_design_params


def load_timepoint(plt_path, use_cache=True, rm_gz=False):
    cell_path = os.path.join(plt_path, "Level_0/Cell_D_00000")
    to_remove = list()
    if not os.path.exists(cell_path):
        gz = cell_path + ".gz"
        if os.path.exists(gz):
            with gzip.open(gz, 'rb') as f_in:
                with open(cell_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            to_remove.append(cell_path)

            if rm_gz:
                to_remove.append(gz)
        else:
            if not use_cache:
                raise ValueError("No Cell_D file find. The only way to read this directory is with use_cache=True")

    cache_path = os.path.join(plt_path, 'tensor.pt')
    if use_cache and os.path.exists(cache_path):
        ret = torch.load(cache_path)
    else:
        ds = yt.load(plt_path)
        ad = ds.all_data()

        P_array = ad['Pz'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
        Phi_array = ad['Phi'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
        Ez_array = ad['Ez'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
        ret = torch.tensor(np.array([P_array, Phi_array, Ez_array]))
        torch.save(ret, cache_path)

    for path in to_remove:
        os.remove(path)
    return ret


class AutoregressionFerroXDataset(Dataset):

    def __init__(self, directories, max_gate_shape=(200, 200, 52)):
        self.X = list()
        self.Y = list()
        self.max_gate_shape = max_gate_shape
        for run_dir in directories:
            time_points = sorted(glob.glob(f"{run_dir}/plt*"))
            self.X.extend(time_points[:-1])
            self.Y.extend(time_points[1:])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, arg):
        X = easy_pad(load_timepoint(self.X[arg]), self.max_gate_shape)
        Y = easy_pad(load_timepoint(self.Y[arg]), self.max_gate_shape)
        return X, Y


class InitFerroXDataset(Dataset):

    def __init__(self, directories, max_gate_shape=(200, 200, 52), return_inputs=False):
        self.return_inputs = return_inputs
        self.X = list()
        self.Y = list()
        self.max_gate_shape = max_gate_shape
        for run_dir in directories:
            states = sorted(glob.glob(f"{run_dir}/plt*"))
            if len(states) > 1:
                initial_state = easy_pad(load_timepoint(states[1]), self.max_gate_shape)
            else:
                continue
            design_params = get_design_params(read_inputs(f"{run_dir}/inputs"))
            self.X.append(design_params)
            self.Y.append(initial_state)

        scaler = StandardScaler()

        self.X = pd.DataFrame(self.X).values

        self.X = torch.tensor(scaler.fit_transform((np.array(self.X))))
        # This is not concatenating all the tensors in Y to create a new dimension.

        self.Y = torch.stack(self.Y)

        self.X_mean = scaler.mean_
        self.X_std = np.sqrt(scaler.var_)


    def __len__(self):
        return len(self.X)


    def __getitem__(self, arg):
        ret = []
        if self.return_inputs:
            ret.append(self.X[arg])
        ret.append(self.Y[arg])
        return tuple(ret)


class FerroXDataset(Dataset):

    def __init__(self, directory, glob_pattern="it*/plt*", max_gate_shape=(200, 200, 52)):
        self.plt_dirs = glob.glob(os.path.join(directory, glob_pattern))
        self.max_gate_shape = max_gate_shape

    def __len__(self):
        return len(self.plt_dirs)

    def __getitem__(self, arg):
        pt = os.path.join(self.plt_dirs[arg], 'tensor.pt')
        if os.path.exists(pt):
            mesh = torch.load(pt)
        else:
            raise ValueError(f"Cannot find preprocessed tensor in {plt}. Please run preformat")
        if self.max_gate_shape is not None:
            mesh = easy_pad(mesh, self.max_gate_shape)
        return mesh


class FerroXDataset2D(Dataset):

    def __init__(self, directory, glob_pattern="it*/plt*", max_gate_shape=(200, 200)):
        self.plt_dirs = sorted(glob.glob(os.path.join(directory, glob_pattern)))
        self.max_gate_shape = max_gate_shape

    def __len__(self):
        return len(self.plt_dirs)

    def __getitem__(self, arg):
        pt = os.path.join(self.plt_dirs[arg], 'tensor.pt')
        if os.path.exists(pt):
            mesh = torch.load(pt)
            surface = mesh[:, :, :, -1][:2]
        else:
            raise ValueError(f"Cannot find preprocessed tensor in {plt}. Please run preformat")
        if self.max_gate_shape is not None:
            surface = easy_pad(surface, self.max_gate_shape)
        return surface
