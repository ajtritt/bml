import glob

import os

import yt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import lightning as L


def load_timepoint(plt_path, use_cache=True):
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
    return ret


def easy_pad(t, tgt_shape=(200, 200, 52)):
    if t.shape[-3:] == tgt_shape:
        return t
    diff = [tgt - st for (tgt, st) in zip(tgt_shape[::-1], t.shape[::-1])]
    pad_arg = list()
    for d in diff:
        q, r = divmod(d, 2)
        pad_arg.append(q + r)
        pad_arg.append(q)
    return F.pad(t, pad_arg, 'constant', 0)



class FerroXDataset(Dataset):
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
