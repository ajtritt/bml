# Device modeling analysis code

## Installation

The following commands will install the code in this repository in such a way
that will allow one to use the tools provided by said code. With that said, the
provided sequence of commands may not suit your specific needs. As this
repository follows PEP 517 style packaging, there are many ways to install the
software, so please use discretion and adapt as necessary.

```bash
git clone git@github.com:ajtritt/bml.git
cd bml
pip install -r requirements.txt
pip install .
```

## FerroX parameter sweeps

This package includes `ferrox-sweep`, a command for running FerroX parameter
sweeps. The first argument to this command is a config file, defining the
parameter ranges you want to sweep over. An example config file can be found in
`configs/example.toml`.

The following parameters are available for sweeping:
- `alpha` - `$\alpha$` Landau free energy coefficient
- `beta` - `$\beta$` Landau free energy coefficient
- `gamma` - `$\gamma$` Landau free energy coefficient
- `L_z_SC` - semiconductor thickness, in meters
- `L_z_DE` - dielectric thickness, in meters
- `L_z_FE` - ferroelectric thickness, in meters
- `L_x` - device width
- `L_y` - device length
- `g11` - `$g_11$` gradient energy coefficient
- `g44` - `$g_44$` gradient energy coefficient

To exclude a parameter from the sweep, set `min`, `max`, and `step` to `nan`.
If a parameter is not swept over, the default value will be calculated from the
input file found in `src/bml/inputs`.

Other parameters set by the config file are:
- `exe_path` - the path to the FerroX executable
- `slurm_project` - the project ID to use for SLURM jobs


```bash
usage: ferrox-sweep [-h] [-n N_SAMPLES] [-p PLOT_INT] [-s SEED] [-d]
                    [-i INPUTS_NAME] [-j JOB_PREFIX]
                    config outdir

positional arguments:
  config                the config file to use for optimization
  outdir                the base directory for submitting job from

options:
  -h, --help            show this help message and exit
  -n N_SAMPLES, --n_samples N_SAMPLES
                        the number of initial samples
  -p PLOT_INT, --plot_int PLOT_INT
                        the plot interval. plot only steady states by default
  -s SEED, --seed SEED  the random number seed to use
  -d, --debug           print inputs and sbatch script only
  -i INPUTS_NAME, --inputs_name INPUTS_NAME
                        the name of the inputs file
  -j JOB_PREFIX, --job_prefix JOB_PREFIX
                        the job name prefix to use
```

## Bayesian Optimization

```
usage: bayes-opt [-h] [-s SEED] [-o OUTDIR] [-d] [-i INPUTS_NAME]
                 [-j JOB_PREFIX] [-I MAX_ITER] [-c] [-p MIN_PERC_IT]
                 config data_dir

positional arguments:
  config                the config file to use for optimization
  data_dir              the data directory with FerroX runs

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  the random number seed to use
  -o OUTDIR, --outdir OUTDIR
                        the base directory for submitting job from
  -d, --debug           print inputs and sbatch script only
  -i INPUTS_NAME, --inputs_name INPUTS_NAME
                        the name of the inputs file
  -j JOB_PREFIX, --job_prefix JOB_PREFIX
                        the job name prefix to use
  -I MAX_ITER, --max_iter MAX_ITER
                        the maximum number of iterations to do
  -c, --ignore_cache    ignore processed FerroX data cache
  -p MIN_PERC_IT, --min_perc_it MIN_PERC_IT
                        minimum complete iterations required to use a FerroX
                        run
```
