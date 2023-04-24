import glob
import re
import sys
from time import time

import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.preprocessing import MaxAbsScaler
import scipy.stats as st
import tqdm
import yt


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
    shape = tuple(inputs['n_cell'][:-1])

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

    x = np.linspace(inputs['prob_lo'][0], inputs['prob_hi'][0], inputs['n_cell'][0])

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


def read_inputs(tfe_dir):
    ret = dict()
    with open(f'{tfe_dir}/inputs', 'r') as f:
        for line in map(lambda x: x.strip(), f):
            if len(line) == 0:
                continue
            key, val = re.split('\s*=\s*', line)
            val = [float(_) if '.' in _ or 'e' in _ else int(_) for _ in re.split('\s+', val)]
            if len(val) == 1:
                val = val[0]
            ret[key] = val
    return ret


def copy_keys(src, dest):
    for k in ('alpha', 'beta', 'gamma', 'g11', 'g44', 'epsilon_de', 'epsilon_si', 'epsilonZ_fe'):
        dest[k] = src[k]


def get_design_params(inputs):
    ret = dict()
    ret['L_z_SC'] = inputs['SC_hi'] - inputs['SC_lo']
    ret['L_z_DE'] = inputs['DE_hi'] - inputs['DE_lo']
    ret['L_z_FE'] = inputs['FE_hi'] - inputs['FE_lo']
    ret['L_x'] = inputs['prob_hi'][0] - inputs['prob_lo'][0]
    ret['L_y'] = inputs['prob_hi'][1] - inputs['prob_lo'][1]
    copy_keys(inputs, ret)
    return ret


def new_inputs(dp):
    inputs = dict()

    dp = dp.copy()
    for k in ('L_z_SC', 'L_z_DE', 'L_z_FE'):
        dp[k] = round(dp[k] * 2e9) / 2e9

    inputs['SC_lo'] = 0.0
    inputs['SC_hi'] = dp['L_z_SC']
    inputs['DE_lo'] = inputs['SC_hi']
    inputs['DE_hi'] = inputs['DE_lo'] + dp['L_z_DE']
    inputs['FE_lo'] = inputs['DE_hi']
    inputs['FE_hi'] = inputs['FE_lo'] + dp['L_z_FE']

    inputs['prob_lo'] = [-dp['L_x'] / 2, -dp['L_y'] / 2, 0.0]
    inputs['prob_hi'] = [ dp['L_x'] / 2,  dp['L_y'] / 2, inputs['FE_hi']]


    inputs['n_cell'] = [int((inputs['prob_hi'][i] - inputs['prob_lo'][i]) / 0.5e-9) for i in range(len(inputs['prob_lo']))]

    copy_keys(dp, inputs)
    return inputs


def read_ferroX_data(data_dir):
    yt.utilities.logger.set_log_level('warning')

    #data_dir = f"{os.environ['SCRATCH']}/bml/MD"

    all_names = list()
    all_data = list()
    tfes = list()

    it = sorted(glob.glob(f"{data_dir}/*nm"))
    it = tqdm.tqdm(it, file=sys.stderr)

    all_V_app = list()
    all_Q = list()
    all_V_fe_avg = list()
    all_Phi_S = list()

    design_params = list()

    inputs_tmpl = None

    for tfe_dir in it:
        tfes.append(float(tfe_dir[tfe_dir.rfind('/')+1:-2]))
        inputs = read_inputs(tfe_dir)
        inputs_tmpl = inputs.copy()
        inputs.setdefault('l', 32e9)

        design_params.append(get_design_params(inputs))
        names = list()
        data = {'Px': list(), 'Py': list(), 'Pz': list()}

        V_app = list()
        Q = list()
        V_fe_avg = list()
        Phi_S = list()

        for i in sorted(glob.glob(f"{tfe_dir}/plt*")):
            if 'old' in i:
                continue
            it = int(i.split('/')[-1][3:])
            if not (it > 0 and it % inputs['inc_step'] == 0):
                continue
            _V_app = ((it // inputs['inc_step']) - 1) / 10
            ds = yt.load(i)
            name = i[i.find('plt')+3:]
            names.append(name)

            ad = ds.all_data()

            _V_fe_avg, _Q, _Phi_S = calculate_values(ad, _V_app, inputs)

            V_app.append(_V_app)
            Q.append(_Q)
            V_fe_avg.append(_V_fe_avg)
            Phi_S.append(_Phi_S)

        order = np.argsort(V_app)
        all_V_app.append(np.array(V_app)[order])
        all_Q.append(np.array(Q)[order])
        all_V_fe_avg.append(np.array(V_fe_avg)[order])
        all_Phi_S.append(np.array(Phi_S)[order])

    order = np.argsort(tfes)

    design_params = pd.DataFrame(data=design_params).iloc[order]
    tfe = np.array(tfes)[order]
    V_app = np.array(all_V_app)[order]
    Q = np.array(all_Q)[order]
    V_fe_avg = np.array(all_V_fe_avg)[order]
    Phi_S = np.array(all_Phi_S)[order]

    return design_params, tfe, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl


def get_next_sample(design_params, response, bounds):
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

    y_pred, pred_std = model.predict(scaler.transform(candidates), return_std=True)

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


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='the data directory with FerroX runs')
    parser.add_argument('-s', '--seed', type=parse_seed, help='the random number seed to use', default='')
    args = parser.parse_args()
    data_dir = args.data_dir

    print(f"Using seed {args.seed}", file=sys.stderr)
    print("Reading FerroX data", file=sys.stderr)
    design_params, tfe, V_app, Q, V_fe_avg, Phi_S, inputs_tmpl = read_ferroX_data(data_dir)
    print("done", file=sys.stderr)

    dPhiS_dVapp = (np.diff(Phi_S) / np.diff(V_app))

    response = np.max(dPhiS_dVapp, axis=1)

    bounds = np.array([design_params.min(axis=0), design_params.max(axis=0)]).T

    next_sample = get_next_sample(design_params, response, bounds)

    inputs_tmpl.update(new_inputs(dict(zip(design_params.columns, next_sample))))

    for k in inputs_tmpl:
        print(f"{k} = {fmt_param(inputs_tmpl[k])}")
