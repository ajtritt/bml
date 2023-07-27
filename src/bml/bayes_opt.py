import glob
from importlib.resources import files
import os
import re
import shutil
import subprocess
import sys
import tempfile
from time import time
import tomllib
import warnings

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import MaxAbsScaler
import scipy.stats as st
import tqdm
import yt


def write_job(config, f, base_outdir, data_dir, job_name="ferroX", job_time=240, inputs_name="inputc", max_iter=100):
    with open(config, 'rb') as conf_f:
        conf = tomllib.load(conf_f)
    exe_path = os.path.expandvars(conf['exe_path'])
    project = conf['nersc_project']

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
bayes-opt -I {max_iter} {data_dir} {config}"""
    f.write(SCRIPT)


def parse_seed(string):
    if string:
        try:
            i = int(string)
            if i > 2**32 - 1:
                raise ValueError(string)
            return i
        except :
            raise argparse.ArgumentTypeError(f'{string} is not a valid seed')
    else:
        return int(int(time() * 1e6) % 1e6)


def calculate_values(ad, V_app, inputs):
    shape = tuple(inputs['domain.n_cell'][:-1])

    if not shape == (64, 64):
        raise ValueError(f"Expected xy dimensions of (64, 64), got {shape}")

    P_array = ad['Pz'].to_ndarray().reshape(ad.ds.domain_dimensions)
    Phi_array = ad['Phi'].to_ndarray().reshape(ad.ds.domain_dimensions)
    Ez_array = ad['Ez'].to_ndarray().reshape(ad.ds.domain_dimensions)

    idx_fede_lo = 20        # int(inputs['FE_lo'] / inputs['dz'])
    idx_fede_hi = 21        # idx_fede_lo + 1
    idx_sc_hi = 19          # int(inputs['SC_hi'] / inputs['dz']) - 1
    l = inputs['l']

    epsilon_0 = inputs['epsilon_0']
    epsilon_de = inputs['epsilon_de']

    x = np.linspace(inputs['domain.prob_lo'][0], inputs['domain.prob_hi'][0], inputs['domain.n_cell'][0])

    #Calculate V_fe_avg
    V_FeDe = 0.5 * (Phi_array[:, :, idx_fede_lo] + Phi_array[:, :, idx_fede_hi])
    # V_FeDe = Phi_array[:,:,index_fede_lo]
    integral_V = 1 / l * 1 / l * np.trapz(np.trapz(V_FeDe, x), x)
    V_fe_avg = V_app - integral_V

    #Calculate Q
    Ez_FeDe = 0.5 * (Ez_array[:, :, idx_fede_lo] + Ez_array[:, :, idx_fede_hi])
    P_FeDe = 0.5 * (P_array[:, :, idx_fede_lo] + P_array[:, :, idx_fede_hi])
    D_FeDe = epsilon_0 * epsilon_de * Ez_FeDe + P_FeDe
    Q = -1 / l * 1 / l * np.trapz(np.trapz(D_FeDe, x), x)

    #Calculate Surface potential
    V_Sc = Phi_array[:, :, idx_sc_hi]
    Phi_S = np.mean(V_Sc)     # Prabhat had this stored in variable xsi

    return V_fe_avg, Q, Phi_S


def parse_val(val):
    try:
        return float(val) if '.' in val or 'e' in val else int(val)
    except:
        return val

def read_inputs(inputs_path):
    ret = dict()
    with open(f'{inputs_path}', 'r') as f:
        for line in map(lambda x: x.strip(), f):
            if len(line) == 0:
                continue
            key, val = re.split('\s*=\s*', line)
            val = [parse_val(_) for _ in re.split('\s+', val)]
            if len(val) == 1:
                val = val[0]
            ret[key] = val
    return ret


def copy_keys(src, dest):
    for k in ('alpha', 'beta', 'gamma', 'g11', 'g44', 'epsilon_de', 'epsilon_si', 'epsilonZ_fe'):
        dest[k] = src[k]


def get_design_params(inputs):
    ret = dict()
    ret['L_z_SC'] = inputs['SC_hi'][2] - inputs['SC_lo'][2]
    ret['L_z_DE'] = inputs['DE_hi'][2] - inputs['DE_lo'][2]
    ret['L_z_FE'] = inputs['FE_hi'][2] - inputs['FE_lo'][2]
    ret['L_x'] = inputs['domain.prob_hi'][0] - inputs['domain.prob_lo'][0]
    ret['L_y'] = inputs['domain.prob_hi'][1] - inputs['domain.prob_lo'][1]
    copy_keys(inputs, ret)
    return ret


def new_inputs(dp, range_df):
    inputs = dict()

    dp = dp.copy()
    for k in ('L_z_SC', 'L_z_DE', 'L_z_FE', 'L_x', 'L_y'):
        res = 1 / range_df['step'][k]
        dp[k] = round(dp[k] * res) / res

    x = dp['L_x'] / 2
    y = dp['L_y'] / 2

    inputs['SC_lo'] = [ -x, -y, 0.0 ]
    inputs['SC_hi'] = [  x,  y, dp['L_z_SC'] ]
    inputs['DE_lo'] = [ -x, -y, inputs['SC_hi'][2] ]
    inputs['DE_hi'] = [  x,  y, inputs['DE_lo'][2] + dp['L_z_DE'] ]
    inputs['FE_lo'] = [ -x, -y, inputs['DE_hi'][2] ]
    inputs['FE_hi'] = [  x,  y, inputs['FE_lo'][2] + dp['L_z_FE'] ]


    inputs['domain.prob_lo'] = [ -x, -y, 0.0 ]
    inputs['domain.prob_hi'] = [  x,  y, inputs['FE_hi'][2] ]


    inputs['domain.n_cell'] = [int((inputs['domain.prob_hi'][i] - inputs['domain.prob_lo'][i]) / 0.5e-9) for i in range(len(inputs['domain.prob_lo']))]

    inputs['phi_tolerance'] = 5e-05

    copy_keys(dp, inputs)
    return inputs


def read_ferroX_data(data_dir, inputs_name, ignore_cache=False):
    yt.utilities.logger.set_log_level('warning')

    all_names = list()
    all_data = list()

    it = sorted(glob.glob(f"{data_dir}/*"))
    it = tqdm.tqdm(it, file=sys.stderr)

    all_V_app = list()
    all_Q = list()
    all_V_fe_avg = list()
    all_Phi_S = list()

    design_params = list()

    inputs_tmpl = None

    for design_dir in it:
        inputs_path = f"{design_dir}/{inputs_name}"
        if not os.path.exists(inputs_path):
            continue
        inputs = read_inputs(inputs_path)
        inputs_tmpl = inputs.copy()
        inputs.setdefault('l', 32e9)

        V_app = np.arange(inputs['Phi_Bc_lo'],
                          inputs['Phi_Bc_hi_max'],
                          inputs['Phi_Bc_inc'])

        plots = list(sorted(glob.glob(f"{design_dir}/plt*")))[1:]
        if len(V_app) != len(plots):
            msg = (f"Skipping {design_dir} - "
                   f"Expected {len(V_app)} plots, got {len(plots)}")
            warnings.warn(msg)
            continue


        design_params.append(get_design_params(inputs))
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

                ds = yt.load(i)
                ad = ds.all_data()

                _V_fe_avg, _Q, _Phi_S = calculate_values(ad, _V_app, inputs)

                Q.append(_Q)
                V_fe_avg.append(_V_fe_avg)
                Phi_S.append(_Phi_S)

            np.savez(cache_path, Q=Q, V_fe_avg=V_fe_avg, Phi_S=Phi_S)

        all_V_app.append(np.array(V_app))
        all_Q.append(np.array(Q))
        all_V_fe_avg.append(np.array(V_fe_avg))
        all_Phi_S.append(np.array(Phi_S))

    design_params = pd.DataFrame(data=design_params)
    V_app = np.array(all_V_app)
    Q = np.array(all_Q)
    V_fe_avg = np.array(all_V_fe_avg)
    Phi_S = np.array(all_Phi_S)

    return design_params, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl


def get_next_sample(design_params, response, bounds, increment=None):
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

    #print(candidates[:, 2])

    y_pred, pred_std = model.predict(scaler.transform(candidates), return_std=True)

    #print(y_pred)
    #print(pred_std)

    current_objective = y_pred[np.argmin(y_pred)]

    # 3 - Calculate the EI values of the candidate samples
    pred_std = pred_std.reshape(pred_std.shape[0])

    cdf = st.norm.cdf((current_objective-y_pred)/pred_std)
    dev = (current_objective - y_pred)
    pdf = st.norm.pdf((current_objective-y_pred)/pred_std)

    EI =  dev * cdf + pred_std * pdf

    # 4 - Select a sample with highest EI
    new_sample = candidates[np.argmax(EI)]

    return new_sample


def fmt_param(param):
    if isinstance(param, list):
        return " ".join(fmt_param(_) for _ in param)
    elif isinstance(param, (int, np.int64)):
        return str(param)
    elif isinstance(param, (float, np.float64)):
        if param == 0.0:
            return '0.0'
        return f'{param:0.3g}'
    elif isinstance(param, str):
        return param
    else:
        raise ValueError(f"Got value of type {type(param)}")


def submit_workflow(config, inputs, outdir, job_name, data_dir, job_name_prefix="ferroX_", submit=True, inputs_name="inputc",
                    max_iter=100):
    inputs_f = tempfile.NamedTemporaryFile('w', prefix='inputs')
    inputs_path = inputs_f.name
    for k in inputs:
        print(f"{k} = {fmt_param(inputs[k])}", file=inputs_f)
    inputs_f.flush()

    sh_f = tempfile.NamedTemporaryFile('w', suffix='.sh')
    sh_path = sh_f.name
    base_outdir = os.path.join(outdir, job_name)
    write_job(config, sh_f, base_outdir, data_dir, job_name=f"{job_name_prefix}{job_name}", inputs_name=inputs_name, max_iter=max_iter)
    sh_f.flush()

    outdir = sh_path

    jobid = None
    if submit:
        jobid = _submit_job(sh_path)
        if jobid == -1:
            exit()
        outdir = f"{base_outdir}.{jobid}"

        os.makedirs(outdir)
        shutil.copyfile(sh_path, f"{outdir}/run.sh")
        shutil.copyfile(inputs_path, f"{outdir}/{inputs_name}")
    else:
        with open(inputs_path, 'r') as f:
            print(f.read())
        print()
        with open(sh_path, 'r') as f:
            print(f.read())


def _submit_job(path):
    cmd = f'sbatch {path}'
    print(cmd)
    output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                shell=True).decode('utf-8')

    result = re.search('Submitted batch job (\d+)', output)
    if result is not None:
        ret = int(result.groups(0)[0])
    else:
        print(f'Job submission failed: {output}')
        ret = -1
    return ret


def run(config, data_dir, outdir, seed, job_prefix="ferroX_", inputs_name="inputc", debug=False, max_iter=100, ignore_cache=False):
    print(f"Using seed {seed}", file=sys.stderr)
    print("Reading FerroX data", file=sys.stderr)
    design_params, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl = read_ferroX_data(data_dir, inputs_name,
                                                                             ignore_cache=ignore_cache)
    print("done", file=sys.stderr)
    it = len(V_app)

    if len(design_params) >= max_iter:
        print("Finished -- the max number of iterations have been run", file=sys.stderr)
        return

    dPhiS_dVapp = (np.diff(Phi_S) / np.diff(V_app))

    response = np.max(dPhiS_dVapp, axis=1)

    bounds_df = pd.DataFrame([design_params.min(axis=0), design_params.max(axis=0)], index=['min', 'max']).T
    with open(config, 'rb') as f:
        range_df = pd.DataFrame(tomllib.load(f)['param']).T
    #range_df = pd.read_csv(files(__package__).joinpath('data/ranges.csv'), index_col=0)
    bounds_df.update(range_df)
    bounds = bounds_df.values

    next_sample = get_next_sample(design_params, response, bounds)

    inputs_tmpl.update(new_inputs(dict(zip(design_params.columns, next_sample)), range_df))

    submit_workflow(config, inputs_tmpl, outdir, f"it{it:05d}", data_dir, job_name_prefix=job_prefix, submit=not debug,
                    inputs_name=inputs_name, max_iter=max_iter)


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

    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = os.path.abspath(args.data_dir)

    run(args.config, args.data_dir, args.outdir, args.seed, job_prefix=args.job_prefix, inputs_name=args.inputs_name,
        debug=args.debug, max_iter=args.max_iter, ignore_cache=args.ignore_cache)
