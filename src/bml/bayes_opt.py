import glob
from importlib.resources import files
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import MaxAbsScaler
import scipy.stats as st
import tqdm
import yt

from .objective import DPhiSOverDVapp
from .job import ChainJobWriter
from .utils import get_input_params, load_config, read_inputs, get_default_inputs, parse_seed, round_sample, write_params, new_inputs, copy_keys

def write_job(config, f, base_outdir, job_name="ferroX", job_time=240, inputs_name="inputc"):
    exe_path = os.path.expandvars(config['exe_path'])
    project = config['slurm_project']

    SCRIPT=f"""#!/bin/bash
#SBATCH -A {project}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t {job_time}
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -e {base_outdir}.%j/error.txt
#SBATCH -o {base_outdir}.%j/output.txt
#SBATCH -J {job_name}
cd {base_outdir}.$SLURM_JOB_ID
export SLURM_CPU_BIND="cores"
srun {exe_path} {inputs_name}
cd {os.getcwd()}
{os.path.basename(sys.argv[0])} {' '.join(sys.argv[1:])}"""
    f.write(SCRIPT)


def write_array_job(config, f, base_outdir, n_tasks, job_name="ferroX", job_time=240, inputs_name="inputc"):
    exe_path = os.path.expandvars(config['exe_path'])
    project = config['slurm_project']

    SCRIPT=f"""#!/bin/bash
#SBATCH -A {project}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t {job_time}
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -n 1
#SBATCH -e {base_outdir}.%j/error.txt
#SBATCH -o {base_outdir}.%j/output.txt
#SBATCH -J {job_name}
#SBATCH -a 1-{n_tasks}
cd {base_outdir}.$SLURM_ARRAY_TASK_ID
export SLURM_CPU_BIND="cores"
srun {exe_path} {inputs_name}
cd {os.getcwd()}
{os.path.basename(sys.argv[0])} {' '.join(sys.argv[1:])}"""
    f.write(SCRIPT)


def calculate_values(plt_path, V_app, inputs, default_l=32e9, Phi_max=True):
    ds = yt.load(plt_path)
    ad = ds.all_data()

    shape = tuple(inputs['domain.n_cell'][:-1])

    P_array = ad['Pz'].to_ndarray().reshape(ad.ds.domain_dimensions)
    Phi_array = ad['Phi'].to_ndarray().reshape(ad.ds.domain_dimensions)
    Ez_array = ad['Ez'].to_ndarray().reshape(ad.ds.domain_dimensions)

    dz = inputs['FE_hi'][2] / inputs['domain.n_cell'][2]
    idx_fede_lo = int(inputs['FE_lo'][2] / dz)
    idx_fede_hi = idx_fede_lo + 1
    idx_sc_hi = int(inputs['SC_hi'][2] / dz) - 1
    lx = inputs['SC_hi'][0] - inputs['SC_lo'][0]
    ly = inputs['SC_hi'][1] - inputs['SC_lo'][1]

    epsilon_0 = inputs['epsilon_0']
    epsilon_de = inputs['epsilon_de']

    x = np.linspace(inputs['domain.prob_lo'][0], inputs['domain.prob_hi'][0], inputs['domain.n_cell'][0])
    y = np.linspace(inputs['domain.prob_lo'][1], inputs['domain.prob_hi'][1], inputs['domain.n_cell'][1])

    #Calculate V_fe_avg
    V_FeDe = 0.5 * (Phi_array[:, :, idx_fede_lo] + Phi_array[:, :, idx_fede_hi])
    integral_V = (1 / lx) * (1 / ly) * np.trapz(np.trapz(V_FeDe, y), x)
    V_fe_avg = V_app - integral_V

    #Calculate Q
    Ez_FeDe = 0.5 * (Ez_array[:, :, idx_fede_lo] + Ez_array[:, :, idx_fede_hi])
    P_FeDe = 0.5 * (P_array[:, :, idx_fede_lo] + P_array[:, :, idx_fede_hi])
    D_FeDe = epsilon_0 * epsilon_de * Ez_FeDe + P_FeDe
    Q = -1 * (1 / lx) * (1 / ly) * np.trapz(np.trapz(D_FeDe, y), x)

    #Calculate Surface potential
    Phi_S = Phi_array[:, :, idx_sc_hi]

    return V_fe_avg, Q, Phi_S


def get_design_params(inputs):
    """Extract design parameters from inputs"""
    ret = dict()
    ret['L_z_SC'] = inputs['SC_hi'][2] - inputs['SC_lo'][2]
    ret['L_z_DE'] = inputs['DE_hi'][2] - inputs['DE_lo'][2]
    ret['L_z_FE'] = inputs['FE_hi'][2] - inputs['FE_lo'][2]
    ret['L_x'] = inputs['domain.prob_hi'][0] - inputs['domain.prob_lo'][0]
    ret['L_y'] = inputs['domain.prob_hi'][1] - inputs['domain.prob_lo'][1]
    copy_keys(inputs, ret)
    return ret


def load_ferrox_dir(design_dir, inputs_name='inputs', min_perc_it=0.0, ignore_cache=False, Phi_max=True):
    inputs_path = f"{design_dir}/{inputs_name}"
    if not os.path.exists(inputs_path):
        raise ValueError(f"Could not find inputs file {inputs_name} in {design_dir}")
    inputs = read_inputs(inputs_path)
    inputs_tmpl = inputs.copy()

    # ^step = \d+
    V_app = np.arange(inputs['Phi_Bc_lo'],
                     inputs['Phi_Bc_hi_max'],
                     inputs['Phi_Bc_inc'])

    plots = list(sorted(glob.glob(f"{design_dir}/plt*")))[1:]
    if len(V_app) != len(plots):
        if (len(plots) / len(V_app)) < min_perc_it:
            msg = (f"Skipping {design_dir} - "
                   f"Expected {len(V_app)} plots, got {len(plots)}")
            warnings.warn(msg)
            return None, None, None, None, None
        else:
            V_app = V_app[:len(plots)]

    cache_path = f"{design_dir}/bayes-opt.npz"
    if os.path.exists(cache_path) and not ignore_cache:
        npz = np.load(cache_path)
        Q = npz['Q']
        V_fe_avg = npz['V_fe_avg']
        Phi_S = npz['Phi_S']
    else:
        Q = list()
        V_fe_avg = list()
        Phi_S = list()

        for _V_app, i in zip(V_app, plots):
            if 'old' in i:
                continue
            it = int(i.split('/')[-1][3:])

            ret = calculate_values(i, _V_app, inputs, Phi_max=Phi_max)

            _V_fe_avg, _Q, _Phi_S = ret

            Q.append(_Q)
            V_fe_avg.append(_V_fe_avg)
            Phi_S.append(_Phi_S)

    np.savez(cache_path, Q=Q, V_fe_avg=V_fe_avg, Phi_S=Phi_S)

    return inputs, V_app, Q, V_fe_avg, Phi_S


def read_ferroX_data(data_dir, **kwargs):
    """
    Read FerroX data

    Args:
        data_dir        : the directory containing the FerroX runs to use for training
        kwargs          : additional arguments for load_ferrox_dir
        ignore_cache    : ignore cached preprocessed FerroX data. This data is stored in each directory in the file
                          named 'bayes-opt.npz'

    Return:
        design_params   : the design params of each run found in data_dir
        V_app           : the applied voltage for each run foudn in data_dir
        Q               : the charge at each applied voltage in V_app
        V_fe_avg        : the average voltage of the ferroelectric for each applied voltage in V_app
        Phi_S           : the surface potential of the ferroelectric for each applied voltage in V_app
        inputs_tmpl     : a template for the inputs file found in the FerroX runs
    """

    yt.utilities.logger.set_log_level('warning')

    all_names = list()
    all_data = list()

    it = sorted(glob.glob(f"{data_dir}/it*"))
    it = tqdm.tqdm(it, file=sys.stderr)

    all_V_app = list()
    all_Q = list()
    all_V_fe_avg = list()
    all_Phi_S = list()

    design_params = list()

    inputs_tmpl = None

    for design_dir in it:

        inputs, V_app, Q, V_fe_avg, Phi_S = load_ferrox_dir(design_dir, **kwargs)
        if inputs is None:
            continue

        design_params.append(get_design_params(inputs))

        all_V_app.append(np.array(V_app))
        all_Q.append(np.array(Q))
        all_V_fe_avg.append(np.array(V_fe_avg))
        all_Phi_S.append(np.array(Phi_S))

        inputs_tmpl = inputs.copy()

    design_params = pd.DataFrame(data=design_params)

    V_app = tuple(all_V_app)
    Q = tuple(all_Q)
    V_fe_avg = tuple(all_V_fe_avg)
    Phi_S = tuple(all_Phi_S)

    return design_params, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl


def get_next_sample(design_params, response, bounds_df, maximize=True, ):
    """
    Fit GaussianProcess to data and return next set of design parameters based on expected improvement.

    Args:
        design_params       : the design parameters for each run of FerroX
        response            : the object function value for each run of FerroX. e.g. dPhi_S/dVapp
        bounds_df           : a pandas.DataFrame containing the min, max, and step for each parameter

    Return:
        new_sample          : a pandas.Series containing the value of each design paramter to use in the next run of
                              FerroX
    """
    bounds = bounds_df[['min', 'max']].values

    # 1 - Train initial Gaussian Process (GP) model
    X_train = design_params.values
    y_train = response # np.max(dPhiS_dVapp, axis=1)

    scaler = MaxAbsScaler().fit(X_train)
    kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5)) * gp.kernels.RBF(1.0, (1e-7, 1e3))
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        optimizer='fmin_l_bfgs_b',
                                        n_restarts_optimizer=30,
                                        alpha=1e-10,
                                        normalize_y=True)
    model.fit(scaler.transform(X_train), y_train)

    # 2 - Generate candidate samples
    candidates = np.array([np.random.uniform(low=bounds[:, 0], high=bounds[:, 1]) for _ in range(10)])
    #candidates = np.array([np.random.uniform(low=bounds[:, 0], high=bounds[:, 1]) for _ in range(100)])

    y_pred, pred_std = model.predict(scaler.transform(candidates), return_std=True)

    # 3 - Calculate the EI values of the candidate samples
    if maximize:
        current_objective = y_train.max()
        dev = y_pred - current_objective
    else:
        current_objective = y_train.min()
        dev = current_objective - y_pred

    pred_std = pred_std.reshape(pred_std.shape[0])
    cdf = st.norm.cdf(dev/pred_std)
    pdf = st.norm.pdf(dev/pred_std)
    EI =  dev * cdf + pred_std * pdf

    # 4 - Select a sample with highest EI
    new_sample = candidates[np.argmax(EI)]

    # 5 - Round params and convert to Series
    new_sample = dict(zip(design_params.columns, new_sample))
    round_sample(new_sample, bounds_df['step'])
    new_sample = pd.Series(new_sample)

    return new_sample


def run(config_path, data_dir, outdir, seed, job_prefix="ferroX_", inputs_name="inputc", debug=False, max_iter=100,
        min_perc_it=0.75, ignore_cache=False, **extra_kwargs):
    """A helper function to call from main"""
    print(f"Using seed {seed}", file=sys.stderr)
    print("Reading FerroX data", file=sys.stderr)
    design_params, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl = read_ferroX_data(data_dir, inputs_name,
                                                                             min_perc_it=min_perc_it,
                                                                             ignore_cache=ignore_cache)
    print("done", file=sys.stderr)

    obj = DPhiSOverDVapp()

    # Read the config file
    config = load_config(config_path)

    # The domain of the design parameters and FerroX constants
    range_df, constants = get_input_params(config)

    it = len(V_app)
    if it > 0:
        print(f"Found {it} iterations", file=sys.stderr)
        if len(design_params) >= max_iter:
            print("Finished -- the max number of iterations have been run", file=sys.stderr)
            return

        # Calculate the value of the objective function for each run of FerroX
        response = np.array([obj(_V_app, _Q, _V_fe_avg, _Phi_S)  for _V_app, _Q, _V_fe_avg, _Phi_S in zip(V_app, Q, V_fe_avg, Phi_S)])

        # Get min and max values for design parameters based on what has been run so far
        bounds_df = pd.DataFrame([design_params.min(axis=0), design_params.max(axis=0)], index=['min', 'max']).T

        # Update the bounds to sample from with those found in the config file
        bounds_df.update(range_df)
        bounds_df = bounds_df.join(range_df['step'])

        # Get the next sample, formatted as a pandas Series
        next_params = get_next_sample(design_params, response, bounds_df)
    else:
        print("No iterations found, starting from scratch", file=sys.stderr)
        inputs_tmpl = get_default_inputs()
        try:
            next_params = pd.Series(np.random.uniform(low=range_df['min'], high=range_df['max']), range_df.index)
            round_sample(next_params, range_df['step'])
        except OverflowError as e:
            if e.args[0] == 'Range exceeds valid bounds':
                missing = range_df.index[range_df[['min', 'max']].isna().any(axis=1)]
                print((f"The following parameters have no bounds: {', '.join(missing)}\n"
                       f"Since we are starting from scrach, please define bounds in {config_path}"), file=sys.stderr)
                exit(1)

    # Convert the design parametes to FerroX inputs, and copy to the template for submitting job
    inputs_tmpl.update(new_inputs(next_params, constants))

    # Write necessary files and submit job to Slurm
    writer = ChainJobWriter(config, job_time=240, inputs_name="inputc", job_prefix=job_prefix)

    outdir = writer.submit_workflow(inputs_tmpl, outdir, f"it{it:05d}", submit=not debug)

    # print new design parameters for the user
    dparams_path = os.path.join(outdir, 'design_params')
    with open(dparams_path, 'w') as f:
        write_params(next_params, f)
    with open(dparams_path, 'r') as f:
        print(f.read().strip(), file=sys.stderr)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='the config file to use for optimization')
    parser.add_argument('data_dir', help='the data directory with FerroX runs')
    parser.add_argument('-s', '--seed', type=parse_seed, help='the random number seed to use', default='')
    parser.add_argument('-o', '--outdir', type=str, help='the base directory for submitting job from', default=None)
    parser.add_argument('-d', '--debug', action='store_true', help='print inputs and sbatch script only', default=False)
    parser.add_argument('-i', '--inputs_name', type=str, help='the name of the inputs file', default='inputc')
    parser.add_argument('-j', '--job_prefix', type=str, help='the job name prefix to use', default='ferroX_')
    parser.add_argument('-I', '--max_iter', type=int, help='the maximum number of iterations to do', default=100)
    parser.add_argument('-c', '--ignore_cache', action='store_true', help='ignore processed FerroX data cache', default=False)
    parser.add_argument('-p', '--min_perc_it', type=float, default=0.75,
                        help='minimum complete iterations required to use a FerroX run')

    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = os.path.abspath(args.data_dir)

    kwargs = vars(args)
    config = kwargs.pop('config')
    data_dir = kwargs.pop('data_dir')
    outdir = kwargs.pop('outdir')
    seed = kwargs.pop('seed')
    run(config, data_dir, outdir, seed, **kwargs)
