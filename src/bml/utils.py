from importlib.resources import files
import re
from time import time
import tomllib

import pandas as pd


def load_config(config_path):
    # Read the config file
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config


def get_input_params(config):
    # The domain of the design parameters
    range_df = pd.DataFrame(config['param']).T
    # FerroX constants
    constants = config.get('constants', dict())
    return range_df, constants


def _parse_val(val):
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
            if line[0] == '#':
                continue
            key, val = re.split('\s*=\s*', line)
            val = [_parse_val(_) for _ in re.split('\s+', val)]
            if len(val) == 1:
                val = val[0]
            ret[key] = val
    return ret


def get_default_inputs():
    return read_inputs(files(__package__).joinpath("inputs"))


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


def round_sample(new_sample, resolutions):
    for k in ('L_z_SC', 'L_z_DE', 'L_z_FE', 'L_x', 'L_y'):
        res = 1 / resolutions[k]
        new_sample[k] = round(new_sample[k] * res) / res


def write_params(design_params, f):
    print("Design params:", file=f)
    for k, v in design_params.items():
        if k[0] == 'L':
            v = f"{v / 1e-9:0.4g} nm"
        else:
            v = f"{v:0.4g}"
        print(f"{k:<15}{v}", file=f)


def copy_keys(src, dest):
    for k in ('alpha', 'beta', 'gamma', 'g11', 'g44', 'epsilon_de', 'epsilon_si', 'epsilonZ_fe'):
        if k not in src:
            continue
        dest[k] = src[k]


def new_inputs(dp, constants):
    """Copy design parameters to inputs """
    inputs = dict()

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
    inputs['domain.n_cell'] = [int((inputs['domain.prob_hi'][0] - inputs['domain.prob_lo'][0]) / 0.5e-9),
                               int((inputs['domain.prob_hi'][1] - inputs['domain.prob_lo'][1]) / 0.5e-9),
                               int((inputs['domain.prob_hi'][2] - inputs['domain.prob_lo'][2]) / 0.5e-9)]

    inputs['domain.max_grid_size'] = inputs['domain.n_cell'].copy()
    inputs['domain.blocking_factor'] = inputs['domain.n_cell'].copy()

    inputs.update(constants)
    copy_keys(dp, inputs)

    return inputs
