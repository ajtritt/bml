import glob
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .job import JobWriter
from .utils import get_input_params, load_config, get_default_inputs, parse_seed, round_sample, write_params, new_inputs


def get_initial_samples(bounds_df, n_samples=10):
    bounds = bounds_df[['min', 'max']].values
    dims_to_sample = bounds[:, 0] != bounds[:, 1]

    candidates = np.zeros((n_samples, len(bounds_df)), dtype=float)
    sampler = qmc.LatinHypercube(d=dims_to_sample.sum())
    sample = sampler.random(n=n_samples)

    candidates[:, dims_to_sample] = qmc.scale(sample, bounds[dims_to_sample, 0], bounds[dims_to_sample, 1])
    candidates[:, ~dims_to_sample] = bounds_df['min'][~dims_to_sample]

    samples = list()
    for i in range(len(candidates)):
        samples.append(pd.Series(candidates[i], bounds_df.index))
        round_sample(samples[-1], bounds_df['step'])

    return pd.DataFrame(samples)


def run(config_path, n_samples, outdir, seed, job_prefix="ferroX_", inputs_name="inputs", debug=False,
        plot_int=None, **extra_kwargs):
    config = load_config(config_path)

    # The domain of the design parameters and FerroX constants
    range_df, constants = get_input_params(config)

    # set up the inputs file template
    inputs_tmpl = get_default_inputs()
    if plot_int is not None:
        inputs_tmpl['plot_int'] = plot_int

    # sample on the LHC - we will make one inputs file per sample
    samples = get_initial_samples(range_df, n_samples=n_samples)

    it_prefix = "it"

    # figure out how many iterations we already have
    start = 0
    if os.path.exists(outdir):
        start = len(glob.glob(os.path.join(outdir, f"{it_prefix}*")))
    else:
        os.mkdir(outdir)

    # write SLURM submission script
    writer = JobWriter(config, job_time=240, inputs_name=inputs_name, job_prefix=job_prefix)
    with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as sh_f:
        sh_path = sh_f.name
        writer.write_array_job(sh_f, outdir, job_prefix, n_samples, start_task=start)

    # create directories for each task and write respective inputs file to the directory
    iters = range(start, start + n_samples)
    for row_i, it_i in zip(range(n_samples), iters):
        sample = samples.iloc[row_i]
        inputs = inputs_tmpl.copy()
        inputs.update(new_inputs(sample, constants))

        run_dir  = os.path.join(outdir, f"{it_prefix}{it_i:06d}")
        os.mkdir(run_dir)
        with open(f"{run_dir}/{inputs_name}", "w") as inputs_f:
            writer.write_inputs(inputs_f, inputs)

        dparams_path = os.path.join(run_dir, 'design_params')
        with open(dparams_path, 'w') as f:
            write_params(sample, f)

    if not debug:
        jobid = writer.submit_job(sh_path)
        dest_sh = os.path.join(outdir, f"array.{jobid}.sh")
    else:
        dest_sh = os.path.join(outdir, "array.unsubmitted.sh")

    shutil.copyfile(sh_path, dest_sh)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='the config file to use for optimization')
    parser.add_argument('outdir', type=str, help='the base directory for submitting job from')
    parser.add_argument('-n', '--n_samples', type=int, help='the number of initial samples', default=10)
    parser.add_argument('-p', '--plot_int', type=int, default=None,
                        help='the plot interval. plot only steady states by default')
    parser.add_argument('-s', '--seed', type=parse_seed, help='the random number seed to use', default='')
    parser.add_argument('-d', '--debug', action='store_true', help='print inputs and sbatch script only', default=False)
    parser.add_argument('-i', '--inputs_name', type=str, help='the name of the inputs file', default='inputs')
    parser.add_argument('-j', '--job_prefix', type=str, help='the job name prefix to use', default='ferroX_')

    args = parser.parse_args(argv)
    kwargs = vars(args)
    config = kwargs.pop('config')
    outdir = kwargs.pop('outdir')
    seed = kwargs.pop('seed')
    n_samples = kwargs.pop('n_samples')
    run(config, n_samples, outdir, seed, **kwargs)
