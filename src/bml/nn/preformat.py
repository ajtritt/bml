import argparse
import glob
import os
import multiprocessing as mp
import shutil
import subprocess
import sys

import numpy as np
import torch
import tqdm
import yt

from .utils import easy_pad
from ..bayes_opt import read_inputs, get_design_params


def preformat(plt_path):
    cache_path = os.path.join(plt_path, 'tensor.pt')
    ds = yt.load(plt_path)
    ad = ds.all_data()

    P_array = ad['Pz'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
    Phi_array = ad['Phi'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
    Ez_array = ad['Ez'].to_ndarray().reshape(ad.ds.domain_dimensions).astype(np.float32)
    ret = torch.tensor(np.array([P_array, Phi_array, Ez_array]))
    torch.save(ret, cache_path)

def process_plt(plt_dir):
    try:
        preformat(plt_dir)
    except:
        return plt_dir
    shutil.rmtree(os.path.join(plt_dir, "Level_0"))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ferrox_base_dir", help="this should look something like 'it00000'")
    parser.add_argument("-H", "--hpss_loc", help="location on HPSS to get tar from",
                        default="/home/a/ajtritt/projects/bml/init_model/training_data")
    parser.add_argument("-N", "--no_backup", action='store_true', default=False,
                        help="do not backup to HPSS after preformatting")
    parser.add_argument("-q", "--quiet", action='store_true', default=False,
                        help="do not print progress information")
    parser.add_argument("-p", "--n_proc", help="the number of processes to use",
                        default=1, type=int)


    args = parser.parse_args()

    tar_path = os.path.join(args.hpss_loc, os.path.basename(args.ferrox_base_dir)) + ".tar"

    try:
        if not os.path.exists(args.ferrox_base_dir):
            cmd = f"htar -xf {tar_path}"
            if not args.quiet:
                print(f"Retrieving contents of {tar_path} from HPSS")
            output = subprocess.check_output(
                        cmd,
                        stderr=subprocess.STDOUT,
                        shell=True).decode('utf-8')
        else:
            if not args.quiet:
                print(f"Directory {args.ferrox_base_dir} already exists. Will not retrieve from HPSS")

        it = sorted(glob.glob(os.path.join(args.ferrox_base_dir, "plt*")))
        if not args.quiet:
            print(f"Found {len(it)} items in {args.ferrox_base_dir}")
            it = tqdm.tqdm(it, file=sys.stdout)

        map_func = map
        if args.n_proc > 1:
            pool = mp.Pool(args.n_proc)
            map_func = pool.map

        result = map_func(process_plt, it)

        if not args.no_backup:
            cmd = f"htar -cf {tar_path} {args.ferrox_base_dir}"
            if not args.quiet:
                print(f"Backing up contents of {args.ferrox_base_dir} to {tar_path} on HPSS")
            output = subprocess.check_output(
                        cmd,
                        stderr=subprocess.STDOUT,
                        shell=True).decode('utf-8')

    except Exception as e:
        print(f"FINISHED\t{args.ferrox_base_dir}\tFAILURE\t{e}")
        exit(13)

    result = list(filter(lambda x: x is not None, result))
    if len(result) == 0:
        print(f"FINISHED\t{args.ferrox_base_dir}\tSUCCESS\t")
    else:
        print(f"FINISHED\t{args.ferrox_base_dir}\tSUCCESS\t{','.join(result)}")

if __name__ == '__main__':
    main()
